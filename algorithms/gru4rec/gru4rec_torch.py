import numpy as np
import pandas as pd

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


class _GRU4RecTorchModel(nn.Module):
    def __init__(self, n_items, embedding_dim, hidden_size, dropout_p_hidden=0.0, constrained_embedding=False):
        super().__init__()
        self.constrained_embedding = bool(constrained_embedding)
        if self.constrained_embedding:
            self.embedding = nn.Embedding(n_items, hidden_size)
            self.output_bias = nn.Parameter(torch.zeros(n_items))
            self.output = None
        else:
            self.embedding = nn.Embedding(n_items, embedding_dim)
            self.output_bias = None
            self.output = nn.Linear(hidden_size, n_items)
        self.gru_cell = nn.GRUCell(embedding_dim, hidden_size)
        self.dropout = nn.Dropout(dropout_p_hidden)

    def output_logits(self, hidden):
        if self.constrained_embedding:
            return F.linear(self.dropout(hidden), self.embedding.weight, self.output_bias)
        return self.output(self.dropout(hidden))

    def score_items(self, hidden, item_indices):
        hidden = self.dropout(hidden)
        if self.constrained_embedding:
            weight = self.embedding.weight[item_indices]
            bias = self.output_bias[item_indices]
            return F.linear(hidden, weight, bias)
        weight = self.output.weight[item_indices]
        bias = self.output.bias[item_indices]
        return F.linear(hidden, weight, bias)

    def forward_step(self, item_indices, hidden):
        emb = self.embedding(item_indices)
        hidden = self.gru_cell(emb, hidden)
        logits = self.output_logits(hidden)
        return hidden, logits


class GRU4RecTorch:
    """
    Minimal PyTorch GRU4Rec implementation for staged migration from Theano.

    The public API mirrors legacy GRU4Rec so existing evaluation code can be
    reused while model internals are modernized incrementally.
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

        if len(self.layers) != 1:
            raise NotImplementedError('GRU4RecTorch currently supports exactly one GRU layer.')

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model = None
        self.optimizer = None
        self.itemidmap = None
        self.n_items = 0
        self.predict = None
        self.current_session = None
        self._pred_hidden = None
        self.predict_batch = None

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

    def fit(self, data, test=None, sample_store=10000000):
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
        ).to(self.device)

        if self.adapt.lower() in ('adam', 'adagrad', 'rmsprop', 'adadelta'):
            if self.adapt.lower() == 'adagrad':
                # Disable foreach to avoid large temporary tensors on big item spaces.
                self.optimizer = torch.optim.Adagrad(
                    self.model.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.lmbd,
                    foreach=False,
                )
            elif self.adapt.lower() == 'rmsprop':
                self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.lmbd)
            elif self.adapt.lower() == 'adadelta':
                self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.learning_rate, weight_decay=self.lmbd)
            else:
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.lmbd)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.lmbd)

        n_sessions = len(offset_sessions) - 1
        if n_sessions == 0:
            raise ValueError('No sessions were found in training data.')
        active_batch = min(self.batch_size, n_sessions)

        for epoch in range(self.n_epochs):
            self.model.train()
            session_idx_arr = np.random.permutation(n_sessions) if self.train_random_order else base_order
            iters = np.arange(active_batch, dtype=np.int64)
            maxiter = iters.max() if len(iters) else -1
            start = offset_sessions[session_idx_arr[iters]]
            end = offset_sessions[session_idx_arr[iters] + 1]

            hidden = torch.zeros((active_batch, self.layers[0]), dtype=torch.float32, device=self.device)
            total_loss = 0.0
            total_count = 0
            finished = False
            while not finished:
                minlen = (end - start).min()
                out_idx = data_items[start]
                for i in range(minlen - 1):
                    in_idx = out_idx
                    out_idx = data_items[start + i + 1]
                    reset = (start + i + 1 == end - 1)

                    xb = torch.as_tensor(in_idx, dtype=torch.long, device=self.device)
                    yb = torch.as_tensor(out_idx, dtype=torch.long, device=self.device)

                    self.optimizer.zero_grad(set_to_none=True)
                    h_active = hidden[:len(iters)]
                    new_hidden, _ = self.model.forward_step(xb, h_active)
                    if reset.any():
                        new_hidden = new_hidden.clone()
                        new_hidden[torch.as_tensor(reset, dtype=torch.bool, device=self.device)] = 0.0
                    hidden[:len(iters)] = new_hidden.detach()
                    scores = self.model.score_items(new_hidden, yb)
                    yhat = self._apply_final_activation(scores)
                    loss = (len(iters) / float(self.batch_size)) * self._loss_fn(yhat, len(iters))
                    loss.backward()
                    if self.grad_cap > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_cap)
                    self.optimizer.step()

                    bs = len(iters)
                    total_loss += float(loss.detach().cpu().item()) * bs
                    total_count += bs

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

            print('Epoch{}\tloss: {:.6f}'.format(epoch, total_loss / max(total_count, 1)))

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
            new_hidden, _ = self.model.forward_step(known_idx, known_hidden)
            self._pred_hidden[known_pos] = new_hidden

        logits = self.model.output_logits(self._pred_hidden)
        return logits.detach().cpu().numpy()

    def predict_next_batch(self, session_ids, input_item_ids, predict_for_item_ids=None, batch=100):
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
        self.current_session = None
        self._pred_hidden = None
        self.predict_batch = None

    def support_users(self):
        return False
