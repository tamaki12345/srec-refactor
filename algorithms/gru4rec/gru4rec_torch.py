import numpy as np
import pandas as pd
import importlib
import time

try:
    tqdm = importlib.import_module('tqdm.auto').tqdm
except Exception:
    tqdm = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


_NN_MODULE_BASE = nn.Module if nn is not None else object


class _GRU4RecTorchModel(_NN_MODULE_BASE):
    def __init__(
        self,
        n_items,
        embedding_dim,
        hidden_size,
        dropout_p_hidden=0.0,
        constrained_embedding=False,
        hidden_act='tanh',
        sigma=0.0,
        init_as_normal=False,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.embedding_dim = int(embedding_dim)
        self.hidden_act = hidden_act
        self.sigma = float(sigma)
        self.init_as_normal = bool(init_as_normal)
        self.constrained_embedding = bool(constrained_embedding)
        if self.constrained_embedding:
            self.embedding = nn.Embedding(n_items, hidden_size)
            self.output_bias = nn.Parameter(torch.zeros(n_items))
            self.output = None
        else:
            self.embedding = nn.Embedding(n_items, embedding_dim)
            self.output_bias = None
            self.output_weight = nn.Parameter(torch.empty(n_items, hidden_size))
            self.output_bias = nn.Parameter(torch.zeros(n_items))
            self.output = None
        self.Wx = nn.Parameter(torch.empty(embedding_dim, hidden_size * 3))
        self.Wh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.Wrz = nn.Parameter(torch.empty(hidden_size, hidden_size * 2))
        self.Bh = nn.Parameter(torch.zeros(hidden_size * 3))
        self.dropout = nn.Dropout(dropout_p_hidden)
        self.reset_parameters()

    def _init_param(self, param):
        if param.ndim < 2:
            param.data.zero_()
            return
        if self.sigma != 0.0:
            scale = self.sigma
        else:
            scale = np.sqrt(6.0 / float(param.shape[0] + param.shape[1]))
        if self.init_as_normal:
            nn.init.normal_(param, mean=0.0, std=scale)
        else:
            nn.init.uniform_(param, a=-scale, b=scale)

    def reset_parameters(self):
        self._init_param(self.embedding.weight)
        self._init_param(self.Wx)
        self._init_param(self.Wh)
        self._init_param(self.Wrz)
        self.Bh.data.zero_()
        if self.constrained_embedding:
            self.output_bias.data.zero_()
        else:
            self._init_param(self.output_weight)
            self.output_bias.data.zero_()

    def _apply_hidden_activation(self, x):
        hidden_act = self.hidden_act.lower()
        if hidden_act == 'relu':
            return torch.relu(x)
        if hidden_act == 'tanh':
            return torch.tanh(x)
        if hidden_act == 'linear':
            return x
        if hidden_act.startswith('leaky-'):
            leak = float(hidden_act.split('-')[1])
            return F.leaky_relu(x, negative_slope=leak)
        if hidden_act.startswith('elu-'):
            alpha = float(hidden_act.split('-')[1])
            return F.elu(x, alpha=alpha)
        if hidden_act.startswith('selu-'):
            parts = hidden_act.split('-')[1:]
            if len(parts) != 2:
                raise ValueError('Invalid selu hidden_act: {}'.format(self.hidden_act))
            lmbd = float(parts[0])
            alpha = float(parts[1])
            return lmbd * torch.where(x >= 0, x, alpha * (torch.exp(x) - 1.0))
        raise NotImplementedError('Unsupported hidden_act for GRU4RecTorch: {}'.format(self.hidden_act))

    def output_logits(self, hidden):
        if self.constrained_embedding:
            return F.linear(self.dropout(hidden), self.embedding.weight, self.output_bias)
        return F.linear(self.dropout(hidden), self.output_weight, self.output_bias)

    def score_items(self, hidden, item_indices):
        hidden = self.dropout(hidden)
        if self.constrained_embedding:
            weight = self.embedding.weight[item_indices]
            bias = self.output_bias[item_indices]
            return F.linear(hidden, weight, bias)
        weight = self.output_weight[item_indices]
        bias = self.output_bias[item_indices]
        return F.linear(hidden, weight, bias)

    def forward_step(self, item_indices, hidden, dropout_p_embed=0.0):
        emb = self.embedding(item_indices)
        if dropout_p_embed > 0.0:
            emb = F.dropout(emb, p=dropout_p_embed, training=self.training)
        hs = self.hidden_size
        vec = torch.matmul(emb, self.Wx) + self.Bh
        rz = torch.sigmoid(vec[:, hs:] + torch.matmul(hidden, self.Wrz))
        h_tilde = self._apply_hidden_activation(torch.matmul(hidden * rz[:, :hs], self.Wh) + vec[:, :hs])
        z = rz[:, hs:]
        hidden_new = (1.0 - z) * hidden + z * h_tilde
        logits = self.output_logits(hidden_new)
        return hidden_new, logits


class GRU4RecTorch:
    """
    Code based on work by Hidasi et al., Recurrent Neural Networks with Top-k
    Gains for Session-based Recommendations, CoRR abs/1706.03847, 2017.

    GRU4RecTorch(loss='bpr-max', final_act='linear', hidden_act='tanh', layers=[100],
                 n_epochs=10, batch_size=32, dropout_p_hidden=0.0, dropout_p_embed=0.0,
                 learning_rate=0.1, momentum=0.0, lmbd=0.0, embedding=0, n_sample=2048,
                 sample_alpha=0.75, smoothing=0.0, constrained_embedding=False,
                 adapt='adagrad', adapt_params=[], grad_cap=0.0, bpreg=1.0,
                 sigma=0.0, init_as_normal=False, train_random_order=False, time_sort=True,
                 session_key='SessionId', item_key='ItemId', time_key='Time',
                 device='auto', seed=42, show_progress=True,
                 use_torch_compile=False, compile_mode='reduce-overhead', compile_backend='inductor',
                 progress_update_interval=128, profile_epoch=False,
                 fixed_candidate_size=False, use_cuda_graphs=False)
    Initializes the network.

    Parameters
    -----------
    loss : 'top1', 'bpr', 'cross-entropy', 'xe_logit', 'top1-max', 'bpr-max'
        selects the loss function (default: 'cross-entropy').
    final_act : 'softmax', 'linear', 'relu', 'tanh', 'softmax_logit'
        selects the activation function of the final layer (default: 'linear').
        NOTE: 'leaky-*', 'elu-*', 'selu-*' are not implemented in this Torch version.
    hidden_act : 'linear', 'relu', 'tanh', 'leaky-<X>', 'elu-<X>', 'selu-<X>-<Y>'
        kept for API compatibility; currently not used because nn.GRUCell uses tanh internally.
    layers : list of int values
        list of the number of GRU units in the layers (default: [100]).
        NOTE: currently exactly one GRU layer is supported.
    n_epochs : int
        number of training epochs (default: 10).
    batch_size : int
        size of the minibatch, also affects the number of in-batch negatives (default: 128).
    dropout_p_hidden : float
        probability of dropout of hidden units (default: 0.0).
    dropout_p_embed : float
        probability of dropout of the input embedding units (default: 0.0).
    learning_rate : float
        learning rate (default: 1e-3).
    momentum : float
        momentum strength for supported optimizers (default: 0.0).
    lmbd : float
        coefficient of L2 regularization (default: 0.0).
    embedding : int
        size of the embedding used; 0 means not to use a separate embedding size and to
        use hidden size instead (default: 0).
    n_sample : int
        number of additional negative samples to be used (besides in-batch negatives)
        (default: 0).
    sample_alpha : float
        probability of an item used as an additional negative sample is supp^sample_alpha
        (e.g. sample_alpha=1 -> popularity based sampling; sample_alpha=0 -> uniform)
        (default: 0.0).
    smoothing : float
        class-label smoothing factor for cross-entropy/xe_logit losses (default: 0.0).
    constrained_embedding : bool
        if True, the output weight matrix is also used as input embedding (default: False).
    adapt : None, 'adagrad', 'rmsprop', 'adam', 'adadelta'
        sets the learning-rate adaptation strategy (default: 'adam').
    adapt_params : list
        kept for API compatibility; currently not used by this Torch implementation.
    grad_cap : float
        clip gradients that exceed this value; 0 means no clipping (default: 0.0).
    bpreg : float
        score regularization coefficient for BPR-max loss (default: 1.0).
    sigma : float
        kept for API compatibility; currently not used by this Torch implementation.
    init_as_normal : bool
        kept for API compatibility; currently not used by this Torch implementation.
    train_random_order : bool
        whether to randomize the order of sessions in each epoch (default: False).
    time_sort : bool
        whether to ensure the order of sessions is chronological when
        train_random_order=False (default: True).
    session_key : string
        header of the session ID column in the input file (default: 'SessionId').
    item_key : string
        header of the item ID column in the input file (default: 'ItemId').
    time_key : string
        header of the timestamp column in the input file (default: 'Time').
    device : 'auto' or torch device string
        training/inference device (default: 'auto').
    seed : int
        random seed for NumPy and PyTorch (default: 42).
    show_progress : bool
        if True, display per-epoch tqdm progress with leave=False (default: True).
    use_torch_compile : bool
        if True and supported, use torch.compile on the forward/loss core step to reduce
        Python overhead during training (default: False).
    compile_mode : str
        torch.compile mode, e.g. 'default', 'reduce-overhead', 'max-autotune' (default: 'reduce-overhead').
    compile_backend : str
        torch.compile backend, e.g. 'inductor', 'aot_eager' (default: 'inductor').
    progress_update_interval : int
        number of processed pairs between tqdm updates; larger values reduce progress-bar overhead
        (default: 128).
    profile_epoch : bool
        if True, print per-epoch timing breakdown to identify bottlenecks (default: False).
    fixed_candidate_size : bool
        if True, keep the candidate-index tensor length fixed to (batch_size + n_sample)
        by padding tail steps; this can reduce shape-guard overhead in compile/graph paths
        (default: False).
    use_cuda_graphs : bool
        if True, try to use torch.cuda.make_graphed_callables for full-size steps on CUDA;
        automatically falls back to eager/compile path if unavailable (default: False).
    """

    def __init__(
        self,
        loss='cross-entropy',
        final_act='linear',
        hidden_act='tanh',
        layers=None,
        n_epochs=10,
        batch_size=128,
        dropout_p_hidden=0.0,
        dropout_p_embed=0.0,
        learning_rate=1e-3,
        momentum=0.0,
        lmbd=0.0,
        embedding=0,
        n_sample=0,
        sample_alpha=0.0,
        smoothing=0.0,
        constrained_embedding=False,
        adapt='adam',
        adapt_params=None,
        grad_cap=0.0,
        bpreg=1.0,
        sigma=0.0,
        init_as_normal=False,
        train_random_order=False,
        time_sort=True,
        session_key='SessionId',
        item_key='ItemId',
        time_key='Time',
        device='auto',
        seed=42,
        show_progress=True,
        use_torch_compile=False,
        compile_mode='reduce-overhead',
        compile_backend='inductor',
        progress_update_interval=128,
        profile_epoch=False,
        fixed_candidate_size=False,
        use_cuda_graphs=False,
    ):
        self._ensure_torch_available()

        self.loss = loss
        self.final_act = final_act
        self.hidden_act = hidden_act
        self.layers = [100] if layers is None else layers
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.dropout_p_hidden = float(dropout_p_hidden)
        self.dropout_p_embed = float(dropout_p_embed)
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.lmbd = float(lmbd)
        self.embedding = int(embedding)
        self.n_sample = int(n_sample)
        self.sample_alpha = float(sample_alpha)
        self.smoothing = float(smoothing)
        self.constrained_embedding = bool(constrained_embedding)
        self.adapt = adapt
        self.adapt_params = [] if adapt_params is None else adapt_params
        self.grad_cap = float(grad_cap)
        self.bpreg = float(bpreg)
        self.sigma = float(sigma)
        self.init_as_normal = bool(init_as_normal)
        self.train_random_order = bool(train_random_order)
        self.time_sort = bool(time_sort)
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.seed = int(seed)
        self.show_progress = bool(show_progress)
        self.use_torch_compile = bool(use_torch_compile)
        self.compile_mode = str(compile_mode)
        self.compile_backend = None if compile_backend in (None, '', 'none', 'None') else str(compile_backend)
        self.progress_update_interval = max(1, int(progress_update_interval))
        self.profile_epoch = bool(profile_epoch)
        self.fixed_candidate_size = bool(fixed_candidate_size)
        self.use_cuda_graphs = bool(use_cuda_graphs)

        if len(self.layers) != 1:
            raise NotImplementedError('GRU4RecTorch currently supports exactly one GRU layer.')

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model = None
        self.optimizer = None
        self._sampled_mode = False
        self._sampled_states = {}
        self.itemidmap = None
        self.n_items = 0
        self.predict = None
        self.error_during_train = False
        self.current_session = None
        self._pred_hidden = None
        self.predict_batch = None
        self._compiled_step = None
        self._compile_enabled = False
        self._cuda_graph_full_step = None

    @staticmethod
    def _ensure_torch_available():
        if _TORCH_IMPORT_ERROR is not None:
            raise ImportError(
                'GRU4RecTorch requires PyTorch, but import failed: {}'.format(_TORCH_IMPORT_ERROR)
            )

    def set_params(self, **kvargs):
        for key, value in kvargs.items():
            if not hasattr(self, key):
                raise NotImplementedError('Unknown attribute: {}'.format(key))
            current = getattr(self, key)
            if isinstance(current, bool):
                if value in ('True', '1', True):
                    value = True
                elif value in ('False', '0', False):
                    value = False
                else:
                    raise ValueError('Invalid boolean value for {}: {}'.format(key, value))
            elif isinstance(current, list):
                if isinstance(value, str):
                    value = [int(v) for v in value.split('/') if v]
                elif not isinstance(value, list):
                    value = [value]
            else:
                value = type(current)(value)
            setattr(self, key, value)

    def _init_data(self, data):
        data = data.sort_values([self.session_key, self.time_key]).copy()
        itemids = data[self.item_key].unique()
        self.n_items = len(itemids)
        self.itemidmap = pd.Series(data=np.arange(self.n_items, dtype=np.int64), index=itemids)
        data = pd.merge(
            data,
            pd.DataFrame({self.item_key: itemids, 'ItemIdx': self.itemidmap[itemids].values}),
            on=self.item_key,
            how='inner',
        )
        offset_sessions = np.zeros(data[self.session_key].nunique() + 1, dtype=np.int64)
        offset_sessions[1:] = data.groupby(self.session_key).size().cumsum().to_numpy(dtype=np.int64)
        if self.time_sort:
            base_order = np.argsort(data.groupby(self.session_key)[self.time_key].min().values)
        else:
            base_order = np.arange(len(offset_sessions) - 1, dtype=np.int64)
        data_items = data['ItemIdx'].to_numpy(dtype=np.int64)
        return data_items, offset_sessions, base_order

    def _generate_neg_samples(self, pop, length):
        if self.sample_alpha:
            sample = np.searchsorted(pop, np.random.rand(self.n_sample * length))
        else:
            sample = np.random.choice(self.n_items, size=self.n_sample * length)
        if length > 1:
            sample = sample.reshape((length, self.n_sample))
        return sample

    def _init_sampled_state(self, param, use_momentum):
        state = {'acc': torch.zeros_like(param)}
        if use_momentum:
            state['velocity'] = torch.zeros_like(param)
        return state

    def _rowwise_adagrad_step(self, param, row_idx):
        if row_idx.numel() == 0:
            return
        grad = param.grad
        if grad is None:
            return
        state = self._sampled_states[param]
        eps = 1e-6
        unique_idx = torch.unique(row_idx)

        with torch.no_grad():
            if param.ndim == 1:
                g = grad.index_select(0, unique_idx)
                acc = state['acc'].index_select(0, unique_idx) + g * g
                state['acc'].index_copy_(0, unique_idx, acc)
                scaled = g / torch.sqrt(acc + eps)
                if self.lmbd > 0:
                    scaled = scaled + self.lmbd * param.index_select(0, unique_idx)
                if 'velocity' in state:
                    vel = state['velocity'].index_select(0, unique_idx)
                    vel = self.momentum * vel - self.learning_rate * scaled
                    state['velocity'].index_copy_(0, unique_idx, vel)
                    param.index_add_(0, unique_idx, vel)
                else:
                    updates = -self.learning_rate * scaled
                    param.index_add_(0, unique_idx, updates)
            else:
                g = grad.index_select(0, unique_idx)
                acc = state['acc'].index_select(0, unique_idx) + g * g
                state['acc'].index_copy_(0, unique_idx, acc)
                scaled = g / torch.sqrt(acc + eps)
                if self.lmbd > 0:
                    scaled = scaled + self.lmbd * param.index_select(0, unique_idx)
                if 'velocity' in state:
                    vel = state['velocity'].index_select(0, unique_idx)
                    vel = self.momentum * vel - self.learning_rate * scaled
                    state['velocity'].index_copy_(0, unique_idx, vel)
                    param.index_add_(0, unique_idx, vel)
                else:
                    updates = -self.learning_rate * scaled
                    param.index_add_(0, unique_idx, updates)

    def _softmax_neg(self, scores, m):
        hm = torch.ones_like(scores)
        hm[:, :m] = hm[:, :m] - torch.eye(m, device=scores.device, dtype=scores.dtype)
        x = scores * hm
        x = x - x.max(dim=1, keepdim=True).values
        e_x = torch.exp(x) * hm
        return e_x / e_x.sum(dim=1, keepdim=True).clamp_min(1e-24)

    def _apply_final_activation(self, scores):
        final_act = self.final_act.lower()
        if final_act == 'linear':
            return scores
        if final_act == 'relu':
            return torch.relu(scores)
        if final_act == 'tanh':
            return torch.tanh(scores)
        if final_act == 'softmax':
            return torch.softmax(scores, dim=1)
        if final_act == 'softmax_logit':
            return torch.logsumexp(scores, dim=1, keepdim=True) - scores
        if final_act.startswith('leaky-'):
            leak = float(final_act.split('-')[1])
            return F.leaky_relu(scores, negative_slope=leak)
        if final_act.startswith('elu-'):
            alpha = float(final_act.split('-')[1])
            return F.elu(scores, alpha=alpha)
        if final_act.startswith('selu-'):
            parts = final_act.split('-')[1:]
            if len(parts) != 2:
                raise ValueError('Invalid selu final_act: {}'.format(self.final_act))
            lmbd = float(parts[0])
            alpha = float(parts[1])
            return lmbd * torch.where(scores >= 0, scores, alpha * (torch.exp(scores) - 1.0))
        raise NotImplementedError('Unsupported final_act for GRU4RecTorch: {}'.format(self.final_act))

    def _loss_fn(self, yhat, m):
        eps = 1e-24
        loss_name = self.loss.lower()
        diag = torch.diagonal(yhat[:, :m], 0)

        if loss_name == 'cross-entropy':
            if self.smoothing:
                n_out = m + self.n_sample
                term_main = (1.0 - (n_out / (n_out - 1.0)) * self.smoothing) * (-torch.log(diag + eps))
                term_smooth = (self.smoothing / (n_out - 1.0)) * torch.sum(-torch.log(yhat + eps), dim=1)
                return torch.mean(term_main + term_smooth)
            return torch.mean(-torch.log(diag + eps))

        if loss_name == 'xe_logit':
            if self.smoothing:
                n_out = m + self.n_sample
                term_main = (1.0 - (n_out / (n_out - 1.0)) * self.smoothing) * diag
                term_smooth = (self.smoothing / (n_out - 1.0)) * torch.sum(yhat, dim=1)
                return torch.mean(term_main + term_smooth)
            return torch.mean(diag)

        if loss_name == 'bpr':
            return torch.mean(-torch.log(torch.sigmoid(diag.unsqueeze(1) - yhat) + eps))

        if loss_name == 'bpr-max':
            softmax_scores = self._softmax_neg(yhat, m)
            bpr_terms = torch.sigmoid(diag.unsqueeze(1) - yhat) * softmax_scores
            reg_terms = self.bpreg * torch.sum((yhat ** 2) * softmax_scores, dim=1)
            return torch.mean(-torch.log(torch.sum(bpr_terms, dim=1) + eps) + reg_terms)

        if loss_name == 'top1':
            term = torch.mean(torch.sigmoid(-diag.unsqueeze(1) + yhat) + torch.sigmoid(yhat ** 2), dim=1)
            reg = torch.sigmoid(diag ** 2) / (m + self.n_sample)
            return torch.mean(term - reg)

        if loss_name == 'top1-max':
            softmax_scores = self._softmax_neg(yhat, m)
            y = softmax_scores * (torch.sigmoid(-diag.unsqueeze(1) + yhat) + torch.sigmoid(yhat ** 2))
            return torch.mean(torch.sum(y, dim=1))

        raise NotImplementedError('Unsupported loss for GRU4RecTorch: {}'.format(self.loss))

    def _step_forward_loss(self, h_active, xb, yb, m, scale):
        new_hidden, _ = self.model.forward_step(xb, h_active, self.dropout_p_embed)
        scores = self.model.score_items(new_hidden, yb)
        yhat = self._apply_final_activation(scores)
        loss = scale * self._loss_fn(yhat, m)
        return new_hidden, loss

    def _compile_step_if_enabled(self):
        self._compiled_step = self._step_forward_loss
        self._compile_enabled = False
        if not self.use_torch_compile:
            return
        compile_fn = getattr(torch, 'compile', None)
        if compile_fn is None:
            print('torch.compile is not available in this runtime. Falling back to eager mode.')
            return
        try:
            kwargs = {'mode': self.compile_mode}
            if self.compile_backend is not None:
                kwargs['backend'] = self.compile_backend
            self._compiled_step = compile_fn(self._step_forward_loss, **kwargs)
            self._compile_enabled = True
            print(
                'Enabled torch.compile for GRU4RecTorch step '
                '(mode={}, backend={}).'.format(self.compile_mode, self.compile_backend or 'default')
            )
        except Exception as exc:
            print('torch.compile failed ({}). Falling back to eager mode.'.format(exc))
            self._compiled_step = self._step_forward_loss

    def _build_cuda_graphed_full_step(self, full_m, full_y_len, full_scale):
        self._cuda_graph_full_step = None
        if not self.use_cuda_graphs:
            return
        if not str(self.device).startswith('cuda'):
            return
        if not torch.cuda.is_available():
            return
        graph_fn = getattr(torch.cuda, 'make_graphed_callables', None)
        if graph_fn is None:
            print('torch.cuda.make_graphed_callables is not available. Continuing without CUDA Graphs.')
            return

        def _full_step(h_active, xb, yb):
            return self._compiled_step(h_active, xb, yb, full_m, full_scale)

        static_h = torch.zeros((full_m, self.layers[0]), dtype=torch.float32, device=self.device)
        static_x = torch.zeros(full_m, dtype=torch.long, device=self.device)
        static_y = torch.zeros(full_y_len, dtype=torch.long, device=self.device)
        try:
            self._cuda_graph_full_step = graph_fn(_full_step, (static_h, static_x, static_y))
            print('Enabled CUDA Graphs for full-size GRU4RecTorch steps (m={}, y={}).'.format(full_m, full_y_len))
        except Exception as exc:
            print('CUDA Graph setup failed ({}). Falling back to non-graphed execution.'.format(exc))
            self._cuda_graph_full_step = None

    def fit(self, data, test=None, sample_store=10000000):
        """
        Trains the network.

        Parameters
        --------
        data : pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one
            column for session IDs, one for item IDs and one for the timestamp of
            the events (unix timestamps). It must have a header. Column names are
            arbitrary, but must correspond to the ones set during initialization
            (session_key, item_key, time_key properties).
        test : pandas.DataFrame, optional
            Present for API compatibility. Currently not used during training.
        sample_store : int
            If additional negative samples are used (n_sample > 0), GPU utilization
            can be improved by precomputing a large batch of negative samples and
            regenerating when necessary. Value is the maximum number of int values
            (IDs) to be stored in RAM for this cache.

        Notes
        --------
        This implementation follows the Theano training structure with session-parallel
        iteration and sampled candidate sets. Some optimizer internals differ from
        Theano due to framework differences.
        """
        self.predict = None
        self.error_during_train = False
        if len(data) == 0:
            raise ValueError('Training data is empty.')

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        data_items, offset_sessions, base_order = self._init_data(data)
        if len(data_items) < 2:
            raise ValueError('Not enough sequential events to build training pairs.')

        hidden_size = int(self.layers[0])
        embedding_dim = int(self.embedding) if self.embedding > 0 else hidden_size
        if self.constrained_embedding:
            embedding_dim = hidden_size
        self.model = _GRU4RecTorchModel(
            n_items=self.n_items,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            dropout_p_hidden=self.dropout_p_hidden,
            constrained_embedding=self.constrained_embedding,
            hidden_act=self.hidden_act,
            sigma=self.sigma,
            init_as_normal=self.init_as_normal,
        ).to(self.device)
        self._compile_step_if_enabled()

        self._sampled_mode = (self.adapt.lower() == 'adagrad')
        self._sampled_states = {}

        if self._sampled_mode:
            dense_params = [self.model.Wx, self.model.Wh, self.model.Wrz, self.model.Bh]
            if self.constrained_embedding:
                sampled_params = [self.model.embedding.weight, self.model.output_bias]
            else:
                sampled_params = [self.model.embedding.weight, self.model.output_weight, self.model.output_bias]
            for param in sampled_params:
                self._sampled_states[param] = self._init_sampled_state(param, use_momentum=(self.momentum > 0))
            self.optimizer = torch.optim.Adagrad(
                dense_params,
                lr=self.learning_rate,
                weight_decay=self.lmbd,
                foreach=False,
            )
        else:
            if self.adapt.lower() in ('adam', 'adagrad', 'rmsprop', 'adadelta'):
                if self.adapt.lower() == 'rmsprop':
                    alpha = float(self.adapt_params[0]) if len(self.adapt_params) >= 1 else 0.9
                    self.optimizer = torch.optim.RMSprop(
                        self.model.parameters(),
                        lr=self.learning_rate,
                        alpha=alpha,
                        momentum=self.momentum,
                        weight_decay=self.lmbd,
                    )
                elif self.adapt.lower() == 'adadelta':
                    rho = float(self.adapt_params[0]) if len(self.adapt_params) >= 1 else 0.95
                    self.optimizer = torch.optim.Adadelta(
                        self.model.parameters(),
                        lr=self.learning_rate,
                        rho=rho,
                        weight_decay=self.lmbd,
                    )
                elif self.adapt.lower() == 'adagrad':
                    self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.lmbd, foreach=False)
                else:
                    b1 = float(self.adapt_params[0]) if len(self.adapt_params) >= 1 else 0.9
                    b2 = float(self.adapt_params[1]) if len(self.adapt_params) >= 2 else 0.999
                    self.optimizer = torch.optim.Adam(
                        self.model.parameters(),
                        lr=self.learning_rate,
                        betas=(b1, b2),
                        weight_decay=self.lmbd,
                    )
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.lmbd)

        n_sessions = len(offset_sessions) - 1
        if n_sessions == 0:
            raise ValueError('No sessions were found in training data.')
        total_pairs = int(len(data_items) - n_sessions)
        active_batch = min(self.batch_size, n_sessions)
        full_y_len = active_batch + max(self.n_sample, 0)

        # Reuse device-side buffers to reduce per-step allocation overhead.
        xb_buf = torch.empty(active_batch, dtype=torch.long, device=self.device)
        yb_buf = torch.empty(full_y_len, dtype=torch.long, device=self.device)
        self._build_cuda_graphed_full_step(active_batch, full_y_len, active_batch / float(self.batch_size))

        pop = None
        sample_pointer = 0
        generate_length = 0
        neg_samples = None
        use_sample_store = False
        if self.n_sample > 0:
            pop = data.groupby(self.item_key).size()
            pop = pop[self.itemidmap.index.values].to_numpy(dtype=np.float64) ** self.sample_alpha
            pop = pop.cumsum() / pop.sum()
            pop[-1] = 1.0
            if sample_store:
                generate_length = sample_store // self.n_sample
                if generate_length > 1:
                    neg_samples = self._generate_neg_samples(pop, generate_length)
                    use_sample_store = True
                else:
                    print('No example store was used')
            else:
                print('No example store was used')

        for epoch in range(self.n_epochs):
            self.model.train()
            pbar = None
            if self.show_progress and tqdm is not None and total_pairs > 0:
                pbar = tqdm(total=total_pairs, desc='Epoch{}'.format(epoch), leave=False, unit='pair')
            session_idx_arr = np.random.permutation(n_sessions) if self.train_random_order else base_order
            iters = np.arange(active_batch, dtype=np.int64)
            maxiter = iters.max() if len(iters) else -1
            start = offset_sessions[session_idx_arr[iters]]
            end = offset_sessions[session_idx_arr[iters] + 1]

            hidden = torch.zeros((active_batch, self.layers[0]), dtype=torch.float32, device=self.device)
            total_loss = 0.0
            total_count = 0
            pending_pbar = 0
            timings = {
                'sample_prepare': 0.0,
                'transfer': 0.0,
                'forward_loss': 0.0,
                'backward': 0.0,
                'sampled_update': 0.0,
                'optimizer_step': 0.0,
                'hidden_update': 0.0,
            }
            n_steps = 0
            epoch_t0 = time.perf_counter()
            finished = False
            while not finished:
                minlen = (end - start).min()
                out_idx = data_items[start]
                for i in range(minlen - 1):
                    n_steps += 1
                    curr_m = len(iters)
                    in_idx = out_idx
                    out_idx = data_items[start + i + 1]
                    reset = (start + i + 1 == end - 1)

                    t0 = time.perf_counter()
                    if self.n_sample > 0:
                        if use_sample_store:
                            if sample_pointer == generate_length:
                                neg_samples = self._generate_neg_samples(pop, generate_length)
                                sample_pointer = 0
                            sample = neg_samples[sample_pointer]
                            sample_pointer += 1
                        else:
                            sample = self._generate_neg_samples(pop, 1)
                        if self.fixed_candidate_size:
                            y = np.empty(full_y_len, dtype=np.int64)
                            y[:curr_m] = out_idx
                            y[curr_m:curr_m + self.n_sample] = sample
                            if curr_m + self.n_sample < full_y_len:
                                y[curr_m + self.n_sample:] = out_idx[0]
                            y_len = full_y_len
                        else:
                            y = np.hstack([out_idx, sample])
                            y_len = curr_m + self.n_sample
                    else:
                        if self.fixed_candidate_size:
                            y = np.empty(full_y_len, dtype=np.int64)
                            y[:curr_m] = out_idx
                            if curr_m < full_y_len:
                                y[curr_m:] = out_idx[0]
                            y_len = full_y_len
                        else:
                            y = out_idx
                            y_len = curr_m
                    timings['sample_prepare'] += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    xb_buf[:curr_m].copy_(torch.from_numpy(in_idx), non_blocking=False)
                    yb_buf[:y_len].copy_(torch.from_numpy(y), non_blocking=False)
                    xb = xb_buf[:curr_m]
                    yb = yb_buf[:y_len]
                    timings['transfer'] += time.perf_counter() - t0

                    self.model.zero_grad(set_to_none=True)
                    self.optimizer.zero_grad(set_to_none=True)
                    h_active = hidden[:len(iters)]
                    step_scale = len(iters) / float(self.batch_size)
                    t0 = time.perf_counter()
                    use_graphed = (
                        self._cuda_graph_full_step is not None and
                        curr_m == active_batch and
                        y_len == full_y_len
                    )
                    if use_graphed:
                        try:
                            new_hidden, loss = self._cuda_graph_full_step(h_active, xb, yb)
                        except Exception as exc:
                            print('CUDA Graph runtime failure ({}). Disabling CUDA Graphs.'.format(exc))
                            self._cuda_graph_full_step = None
                            new_hidden, loss = self._compiled_step(h_active, xb, yb, len(iters), step_scale)
                    else:
                        try:
                            new_hidden, loss = self._compiled_step(h_active, xb, yb, len(iters), step_scale)
                        except Exception as exc:
                            if self._compile_enabled:
                                print('torch.compile runtime failure ({}). Falling back to eager mode.'.format(exc))
                                self._compiled_step = self._step_forward_loss
                                self._compile_enabled = False
                                new_hidden, loss = self._compiled_step(h_active, xb, yb, len(iters), step_scale)
                            else:
                                raise
                    timings['forward_loss'] += time.perf_counter() - t0
                    if torch.isnan(loss).item():
                        print(str(epoch) + ': NaN error!')
                        self.error_during_train = True
                        if pbar is not None:
                            pbar.close()
                        return
                    t0 = time.perf_counter()
                    loss.backward()
                    if self.grad_cap > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_cap)
                    timings['backward'] += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    if self._sampled_mode:
                        if self.constrained_embedding:
                            emb_rows = torch.cat([xb, yb], dim=0)
                            self._rowwise_adagrad_step(self.model.embedding.weight, emb_rows)
                            self._rowwise_adagrad_step(self.model.output_bias, yb)
                            self.model.embedding.weight.grad = None
                            self.model.output_bias.grad = None
                        else:
                            self._rowwise_adagrad_step(self.model.embedding.weight, xb)
                            self._rowwise_adagrad_step(self.model.output_weight, yb)
                            self._rowwise_adagrad_step(self.model.output_bias, yb)
                            self.model.embedding.weight.grad = None
                            self.model.output_weight.grad = None
                            self.model.output_bias.grad = None
                    timings['sampled_update'] += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    self.optimizer.step()
                    timings['optimizer_step'] += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    next_hidden = new_hidden.detach()
                    if reset.any():
                        next_hidden = next_hidden.clone()
                        next_hidden[torch.as_tensor(reset, dtype=torch.bool, device=self.device)] = 0.0
                    hidden[:len(iters)] = next_hidden
                    timings['hidden_update'] += time.perf_counter() - t0

                    bs = len(iters)
                    total_loss += float(loss.detach().cpu().item()) * bs
                    total_count += bs
                    if pbar is not None:
                        pending_pbar += bs
                        if pending_pbar >= self.progress_update_interval:
                            pbar.update(pending_pbar)
                            pending_pbar = 0

                start = start + minlen - 1
                finished_mask = (end - start <= 1)
                n_finished = finished_mask.sum()
                if n_finished > 0:
                    iters[finished_mask] = maxiter + np.arange(1, n_finished + 1, dtype=np.int64)
                    maxiter += n_finished
                valid_mask = (iters < n_sessions)
                n_valid = valid_mask.sum()
                if (n_valid == 0) or (n_valid < 2 and self.n_sample == 0):
                    finished = True
                    break

                mask = finished_mask & valid_mask
                if mask.any():
                    sessions = session_idx_arr[iters[mask]]
                    start[mask] = offset_sessions[sessions]
                    end[mask] = offset_sessions[sessions + 1]
                iters = iters[valid_mask]
                start = start[valid_mask]
                end = end[valid_mask]
                if n_valid < len(valid_mask):
                    hidden = hidden[torch.as_tensor(valid_mask, dtype=torch.bool, device=self.device)]

            if pbar is not None:
                if pending_pbar > 0:
                    pbar.update(pending_pbar)
                if total_count < total_pairs:
                    pbar.update(total_pairs - total_count)
                pbar.close()

            print('Epoch{}\tloss: {:.6f}'.format(epoch, total_loss / max(total_count, 1)))
            if self.profile_epoch:
                total_epoch_s = time.perf_counter() - epoch_t0
                print(
                    'Epoch{}\tprofile_s: total={:.3f}, sample_prepare={:.3f}, transfer={:.3f}, '
                    'forward_loss={:.3f}, backward={:.3f}, sampled_update={:.3f}, '
                    'optimizer_step={:.3f}, hidden_update={:.3f}, steps={}'.format(
                        epoch,
                        total_epoch_s,
                        timings['sample_prepare'],
                        timings['transfer'],
                        timings['forward_loss'],
                        timings['backward'],
                        timings['sampled_update'],
                        timings['optimizer_step'],
                        timings['hidden_update'],
                        n_steps,
                    )
                )

        self.predict = None
        self.current_session = None
        self._pred_hidden = None
        self.predict_batch = None
        return self

    def _ensure_predict_state(self, batch):
        if self.model is None:
            raise RuntimeError('Model is not trained. Call fit() first.')
        if self._pred_hidden is None or self.predict_batch != batch:
            self.predict_batch = batch
            self._pred_hidden = torch.zeros((batch, self.layers[0]), dtype=torch.float32, device=self.device)
            self.current_session = np.ones(batch, dtype=np.int64) * -1

    def _forward_predict(self, input_item_ids, batch):
        item_positions = self.itemidmap.reindex(input_item_ids)
        known_mask = ~item_positions.isna().to_numpy()

        if known_mask.any():
            known_pos = np.where(known_mask)[0]
            known_idx = torch.as_tensor(item_positions.to_numpy()[known_mask], dtype=torch.long, device=self.device)
            known_hidden = self._pred_hidden[known_pos]
            new_hidden, _ = self.model.forward_step(known_idx, known_hidden, 0.0)
            self._pred_hidden[known_pos] = new_hidden

        logits = self.model.output_logits(self._pred_hidden)
        return logits.detach().cpu().numpy()

    def predict_next_batch(self, session_ids, input_item_ids, predict_for_item_ids=None, batch=100):
        """
        Gives prediction scores for a selected set of items. Can be used in batch
        mode to predict for multiple independent events (events of different
        sessions) at once and thus speed up evaluation.

        If the session ID at a given coordinate of the session_ids parameter remains
        the same during subsequent calls of the function, the corresponding hidden
        state of the network will be kept intact (this is how one can predict an
        item sequence in a session). If it changes, the hidden state is reset.

        Parameters
        --------
        session_ids : 1D array
            Contains session IDs of events in the batch. Length must equal batch.
        input_item_ids : 1D array
            Contains input item IDs of events in the batch. Length must equal batch.
            Unknown items are ignored in hidden-state update and receive NaN where
            appropriate in the output frame.
        predict_for_item_ids : 1D array (optional)
            IDs of items for which the network should give prediction scores.
            Default None means predict over all items in the training set.
        batch : int
            Prediction batch size.

        Returns
        --------
        out : pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by item IDs.
        """
        if self.error_during_train:
            raise Exception
        session_ids = np.asarray(session_ids)
        input_item_ids = np.asarray(input_item_ids)
        self._ensure_predict_state(batch)

        changed = np.arange(batch)[session_ids != self.current_session]
        if len(changed) > 0:
            self._pred_hidden[changed] = 0.0
            self.current_session = session_ids.copy()

        logits = self._forward_predict(input_item_ids, batch)

        if predict_for_item_ids is not None:
            predict_for_item_ids = np.asarray(predict_for_item_ids)
            pred_positions = self.itemidmap.reindex(predict_for_item_ids)
            scores = np.full((len(predict_for_item_ids), batch), np.nan, dtype=np.float32)
            known = ~pred_positions.isna().to_numpy()
            if known.any():
                known_idx = pred_positions.to_numpy()[known].astype(np.int64)
                scores[known, :] = logits[:, known_idx].T
            return pd.DataFrame(data=scores, index=predict_for_item_ids)

        return pd.DataFrame(data=logits.T, index=self.itemidmap.index)

    def symbolic_predict(self, X, Y, M, items, batch_size):
        raise NotImplementedError('symbolic_predict is not available in GRU4RecTorch.')

    def predict_next(self, session_id, input_item_id, predict_for_item_ids=None, skip=False, mode_type='view', timestamp=0):
        """
        Gives prediction scores for a selected set of items for a single event.

        Parameters
        --------
        session_id : int
            Session ID of the event.
        input_item_id : int
            Input item ID of the event.
        predict_for_item_ids : 1D array (optional)
            IDs of items for which the network should give prediction scores.
            Default None means predict over all items in the training set.
        skip : bool
            Present for API compatibility. Currently not used.
        mode_type : str
            Present for API compatibility. Currently not used.
        timestamp : int
            Present for API compatibility. Currently not used.

        Returns
        --------
        out : pandas.Series or None
            Prediction scores indexed by item ID, or None when input item is unknown.
        """
        if self.itemidmap is None or input_item_id not in self.itemidmap.index:
            return None
        return self.predict_next_batch(
            np.array([session_id]),
            np.array([input_item_id]),
            predict_for_item_ids,
            batch=1,
        )[0]

    def clear(self):
        self.model = None
        self.optimizer = None
        self.predict = None
        self.error_during_train = False
        self.current_session = None
        self._pred_hidden = None
        self.predict_batch = None

    def support_users(self):
        """
        whether it is a session-based or session-aware algorithm
        (if returns True, method "predict_with_training_data" must be defined as well)

        Parameters
        --------

        Returns
        --------
        True : if it is session-aware
        False : if it is session-based
        """
        return False
