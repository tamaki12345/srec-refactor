from algorithms.gru4rec.gru4rec import GRU4Rec

try:
    from algorithms.gru4rec.gru4rec_torch import GRU4RecTorch
except Exception:
    # Keep Theano-only environments importable even when PyTorch is unavailable.
    GRU4RecTorch = None

__all__ = ['GRU4Rec', 'GRU4RecTorch']
