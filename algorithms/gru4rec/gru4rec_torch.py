import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except Exception as exc:
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


class _GRU4RecTorchModel(nn.Module):
    def __init__(self, n_items, embedding_dim, hidden_size, dropout_p_hidden=0.0):
        super().__init__()
        self.embedding = nn.Embedding(n_items, embedding_dim)
        self.gru_cell = nn.GRUCell(embedding_dim, hidden_size)
        self.dropout = nn.Dropout(dropout_p_hidden)
        self.output = nn.Linear(hidden_size, n_items)

    def forward_step(self, item_indices, hidden):
        emb = self.embedding(item_indices)
        hidden = self.gru_cell(emb, hidden)
        logits = self.output(self.dropout(hidden))
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
        self.criterion = None
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

    def _build_training_pairs(self, data):
        data = data.sort_values([self.session_key, self.time_key]).copy()
        itemids = data[self.item_key].unique()
        self.n_items = len(itemids)
        self.itemidmap = pd.Series(data=np.arange(self.n_items, dtype=np.int64), index=itemids)
        item_idx = self.itemidmap[data[self.item_key]].to_numpy(dtype=np.int64)
        sessions = data[self.session_key].to_numpy()
        starts_new_session = np.ones(len(data), dtype=bool)
        starts_new_session[1:] = sessions[1:] != sessions[:-1]
        valid = ~starts_new_session
        x = item_idx[:-1][valid[1:]]
        y = item_idx[1:][valid[1:]]
        return x, y

    def fit(self, data, test=None, sample_store=10000000):
        if len(data) == 0:
            raise ValueError('Training data is empty.')

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        x_np, y_np = self._build_training_pairs(data)
        if len(x_np) == 0:
            raise ValueError('Not enough sequential events to build training pairs.')

        hidden_size = int(self.layers[0])
        embedding_dim = int(self.embedding) if self.embedding > 0 else hidden_size
        self.model = _GRU4RecTorchModel(
            n_items=self.n_items,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            dropout_p_hidden=self.dropout_p_hidden,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        if self.adapt.lower() in ('adam', 'adagrad', 'rmsprop', 'adadelta'):
            if self.adapt.lower() == 'adagrad':
                self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.lmbd)
            elif self.adapt.lower() == 'rmsprop':
                self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.lmbd)
            elif self.adapt.lower() == 'adadelta':
                self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.learning_rate, weight_decay=self.lmbd)
            else:
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.lmbd)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.lmbd)

        x_all = torch.as_tensor(x_np, dtype=torch.long, device=self.device)
        y_all = torch.as_tensor(y_np, dtype=torch.long, device=self.device)

        n_samples = x_all.shape[0]
        for epoch in range(self.n_epochs):
            self.model.train()
            perm = np.random.permutation(n_samples)
            total_loss = 0.0
            total_count = 0
            for start in range(0, n_samples, self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                xb = x_all[batch_idx]
                yb = y_all[batch_idx]
                h0 = torch.zeros((xb.shape[0], self.layers[0]), dtype=torch.float32, device=self.device)
                _, logits = self.model.forward_step(xb, h0)
                loss = self.criterion(logits, yb)

                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_cap > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_cap)
                self.optimizer.step()

                bs = xb.shape[0]
                total_loss += float(loss.detach().cpu().item()) * bs
                total_count += bs

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

        logits = self.model.output(self._pred_hidden)
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
        self.criterion = None
        self.predict = None
        self.current_session = None
        self._pred_hidden = None
        self.predict_batch = None

    def support_users(self):
        return False
