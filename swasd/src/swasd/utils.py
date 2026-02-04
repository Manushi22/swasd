import numpy as np
from scipy import stats
from importlib import resources

def detrend_samples(samples):
    """
    Detrend each column of a 2D array by removing a linear trend over rows.
    samples: shape (iterations, parameters)
    """
    samples = np.asarray(samples, dtype=float)
    n_iter, n_params = samples.shape
    x = np.arange(n_iter, dtype=float)
    detrended_samples = np.empty_like(samples, dtype=float)

    X = np.column_stack([x, np.ones(n_iter)])     # (n, 2)
    beta, *_ = np.linalg.lstsq(X, samples, rcond=None)  # (2, d)
    trend = X @ beta

    return samples - trend


def _flatten_samples(samples, num_warmup=0):
    """
    Normalize samples to 2D (T, d).
    - If 3D (C, T, d): drop `num_warmup` per chain, then stack to (C*T_warm, d).
    - If 2D (T, d): drop the first `num_warmup` rows globally.
    """
    if samples.ndim == 3:
        C, T, d = samples.shape
        T_eff = max(T - int(num_warmup), 0)
        if T_eff <= 0:
            # Nothing left after warmup; return empty with correct dim
            return np.empty((0, d), dtype=samples.dtype)
        # Drop warmup per chain and stack chains vertically
        return samples[:, int(num_warmup):, :].reshape(C * T_eff, d)
    elif samples.ndim == 2:
        T, d = samples.shape
        start = int(num_warmup)
        start = min(max(start, 0), T)
        return samples[start:, :]
    else:
        raise ValueError("samples must be a 2D (T,d) or 3D (C,T,d) array")

def compute_scale_from_last_blocks(
    samples,
    num_blocks=6,
    num_warmup=0,
    detrend_fn=None,
):
    """
    Compute per-dimension scale (std. dev.) from the last two blocks of
    post-warmup samples.

    Parameters
    ----------
    samples : np.ndarray
        - 2D (draws, dims) for BBVI/SGD, OR
        - 3D (chains, draws, dims) for MCMC.
    num_blocks : int
        Number of total blocks to divide the post-warmup samples into.
    num_warmup : int
        Number of warmup samples to discard:
          - If 3D, discarded per chain before flattening.
          - If 2D, discarded globally from the start.
    detrend_fn : callable or None
        Optional detrending function applied to the 2D post-warmup array before SD.
        If None, defaults to `detrend_samples`.
    """
    # Normalize to 2D after applying warmup handling
    arr2d = _flatten_samples(samples, num_warmup=num_warmup)  # (T_eff, d)

    if arr2d.size == 0:
        # Degenerate case: return unit scales with best-effort dimensionality
        d = samples.shape[-1]
        return np.ones(d, dtype=float)

    # Detrend if requested (default: linear detrend per dim)
    if detrend_fn is None:
        arr2d = detrend_samples(arr2d)
    else:
        arr2d = detrend_fn(arr2d)

    blocked_samples = np.array_split(arr2d, int(num_blocks))

    if len(blocked_samples) < 2:
        sd_values = np.std(arr2d, axis=0, ddof=1) if arr2d.shape[0] > 1 else np.ones(arr2d.shape[1], dtype=float)
        sd_values[sd_values == 0] = 1.0
        return sd_values

    last_two = np.concatenate(blocked_samples[-2:], axis=0)

    if last_two.shape[0] <= 1:
        sd_values = np.std(arr2d, axis=0, ddof=1) if arr2d.shape[0] > 1 else np.ones(arr2d.shape[1], dtype=float)
    else:
        sd_values = np.std(last_two, axis=0, ddof=1)

    sd_values[np.isclose(sd_values, 0.0, atol=1e-12)] = 1.0
    return sd_values
    

def load_stan_text(rel_path: str):
    """
    Read a bundled Stan file from swasd/models and return its text.
    Example: load_stan_text("monotone_swd.stan")
    """
    with resources.files("swasd.models").joinpath(rel_path).open("r", encoding="utf-8") as f:
        return f.read()
