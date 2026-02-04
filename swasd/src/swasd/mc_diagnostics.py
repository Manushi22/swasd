import numpy as np
import arviz as az
import xarray as xr


def ensure_chains_draws_dim(samples: np.ndarray):
    """
    Return samples as shape (chains, draws, dim).

    Accepts:
      - (draws, dim)
      - (chains, draws, dim)
    """
    x = np.asarray(samples)
    if x.ndim == 2:
        return x[None, :, :]  # (1, draws, dim)
    if x.ndim == 3:
        return x
    raise ValueError("samples must have shape (draws, dim) or (chains, draws, dim)")


def split_chain(x: np.ndarray):
    """
    If x has shape (1, draws, dim), split draws into 2 chains:
      -> (2, draws//2, dim)

    If already has >=2 chains, return as-is.
    """
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError("expected (chains, draws, dim)")

    C, T, D = x.shape
    if C >= 2:
        return x

    half = T // 2
    if half < 2:
        raise ValueError("not enough draws to split into 2 chains")

    T2 = 2 * half
    x = x[:, :T2, :]

    first = x[0, :half, :]
    second = x[0, half:, :]
    return np.stack([first, second], axis=0)


def compute_rhat_max(samples: np.ndarray, method: str = "rank"):
    """
    Compute max R-hat across parameters.
    samples can be (draws, dim) or (chains, draws, dim).
    """
    x = ensure_chains_draws_dim(samples)

    # Force >=2 chains by splitting if needed
    if x.shape[0] < 2:
        x = split_chain(x)

    C, T, D = x.shape
    posterior = xr.Dataset(
        {f"param_{i}": (("chain", "draw"), x[:, :, i]) for i in range(D)}
    )
    idata = az.InferenceData(posterior=posterior)
    rhat = az.rhat(idata, method=method).to_array().values
    return float(np.nanmax(rhat))


def rhat_check(
    samples: np.ndarray,
    windows,
    rhat_threshold: float = 1.01,
    method: str = "rank",
):
    """
    Evaluate R-hat over multiple window sizes (last w draws).

    Returns:
      success (bool), best_window (int|None), best_rhat (float)
    """
    x = ensure_chains_draws_dim(samples)
    C, T, D = x.shape

    windows = np.asarray(windows, dtype=int)
    if windows.size == 0:
        return False, None, float("inf")

    rhat_vals = []
    for w in windows:
        xw = x[:, -w:, :]
        rhat_vals.append(compute_rhat_max(xw, method=method))

    rhat_vals = np.asarray(rhat_vals, dtype=float)
    best_idx = int(np.nanargmin(rhat_vals))
    best_w = int(windows[best_idx])
    best_rhat = float(rhat_vals[best_idx])
    return best_rhat <= rhat_threshold, best_w, best_rhat
