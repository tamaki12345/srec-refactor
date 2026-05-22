class GRU4RecTorch:
    """
    Migration target for a PyTorch-based GRU4Rec implementation.

    This class keeps the high-level API shape (`fit`, `predict_next_batch`,
    `predict_next`, `support_users`) so callers can gradually switch from the
    legacy Theano-backed `GRU4Rec`.
    """

    def __init__(self, *args, **kwargs):
        self._init_args = args
        self._init_kwargs = kwargs

    def fit(self, data, test=None, sample_store=10000000):
        raise NotImplementedError(
            'GRU4RecTorch.fit is not implemented yet. '
            'Please keep using algorithms.gru4rec.gru4rec.GRU4Rec for now.'
        )

    def predict_next_batch(self, session_ids, input_item_ids, predict_for_item_ids=None, batch=100):
        raise NotImplementedError('GRU4RecTorch.predict_next_batch is not implemented yet.')

    def predict_next(self, session_id, input_item_id, predict_for_item_ids=None, skip=False, mode_type='view', timestamp=0):
        raise NotImplementedError('GRU4RecTorch.predict_next is not implemented yet.')

    def support_users(self):
        return False
