import numpy as np
from tqdm import tqdm
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
from .metrics import MetricComputer
from .regression import MonotoneSWDModel
from .mc_diagnostics import rhat_check

def _geometric_checkpoints(n_iters, start, rate):
    """
    Returns integer checkpoints: start, ceil(start*rate), ceil(start*rate^2), ...
    up to and including n_iters.
    """
    if rate <= 1.0:
        raise ValueError("check_rate must be > 1.0")
    ks = []
    k = max(1, int(start))
    seen = set()
    while k < n_iters:
        ki = int(np.ceil(k))
        if ki not in seen:
            ks.append(ki)
            seen.add(ki)
        k = int(np.ceil(k * rate))
    if not ks or ks[-1] != n_iters:
        ks.append(int(n_iters))
    return np.array(ks, dtype=int)
    
def _linear_checkpoints(n_iters, start, step, stop = None):
    if step < 1:
        raise ValueError("step must be >= 1")
    if stop is None:
        stop = n_iters
    stop = min(int(stop), int(n_iters))
    start = max(1, int(start))
    if start > stop:
        return np.array([], dtype=int)

    ks = list(range(start, stop + 1, int(step)))
    if ks[-1] != stop:
        ks.append(stop)
    return np.array(sorted(set(ks)), dtype=int)
    
def _slice_samples(samples, k):
    """Return samples truncated to first k iterations."""
    if samples.ndim == 3:
        return samples[:, :k, :]
    return samples[:k, :]

def _print_summary(history, k_conv, reason, rhat_threshold=None, swd_threshold=None):
    print("\n" + "=" * 60)
    print("CONVERGENCE DETECTED!")
    print("=" * 60)
    print(f"Convergence at iteration: {k_conv}")
    print(f"Detection method: {reason.upper()}")

    if reason == "rhat":
        rhat = history["rhat_best"][-1] if history["rhat_best"] else None
        if rhat is not None and rhat_threshold is not None:
            print(f"Final Rhat: {rhat:.4f} (threshold: {rhat_threshold:.4f})")

    if reason == "swd":
        swd = history["swd_to_stationary"][-1] if history["swd_to_stationary"] else None
        if swd is not None and swd_threshold is not None:
            print(f"Final SWD distance: {swd:.4f} (threshold: {swd_threshold:.4f})")

    if history["rhat_best"]:
        print(f"\nRhat checks performed: {len(history['rhat_best'])}")
        print(f"Best Rhat achieved: {min(history['rhat_best']):.4f}")

    if history["swd_to_stationary"]:
        print(f"\nSWD checks performed: {len(history['swd_to_stationary'])}")
        print(f"Initial SWD: {history['swd_to_stationary'][0]:.4f}")
        print(f"Final SWD: {history['swd_to_stationary'][-1]:.4f}")

    print("=" * 60)


def convergence_check(
    samples,
    *,
    do_rhat_check,
    do_swd_check,
    history,
    metric_comp,
    n_blocks=6,
    block_mode="all_pairs",
    wo_init_block=True,
    convg_threshold=1.0,
    true_swd=False,
    true_samples=None,
    rhat_threshold=1.01,
    rhat_wmin=200,
    rhat_method="rank",
    rhat_num_windows=5,
    verbose=2,
):
    """
    Algorithm 1 (function): Given samples and an iteration k, run whichever checks
    are scheduled at k (Rhat and/or SWD), update history, and return:

      (k_conv, convergence_reason) or (None, None)
    """
    k = samples.shape[0] if samples.ndim == 2 else samples.shape[1]
    # --------------------
    # Rhat check at k
    # --------------------
    if do_rhat_check:
        w_upper = int(0.95 * k)
        if w_upper > rhat_wmin:
            windows = np.linspace(rhat_wmin, w_upper, num=rhat_num_windows, dtype=int)
            success, best_w, best_rhat = rhat_check(
                samples,
                windows=windows,
                rhat_threshold=rhat_threshold,
                method=rhat_method,
            )

            history["rhat_check_iterate"].append(k)
            history["rhat_best_w"].append(best_w)
            history["rhat_best"].append(best_rhat)

            if verbose == 3:
                tqdm.write(f"[Rhat] k={k} | rhat={best_rhat:.3f} (best_w={best_w})")

            if success:
                history["k_conv"] = k
                history["convergence_reason"] = "rhat"
                return k, "rhat"

    # --------------------
    # SWD check at k
    # --------------------
    if do_swd_check:
        swd_results = metric_comp.compute_blockwise(
            samples,
            metric="swd",
            num_blocks=n_blocks,
            mode=block_mode,
            verbose=False,
        )
        history["pairwise_swd_results"].append(swd_results)

        if true_swd:
            swd_true_results = metric_comp.compute_blockwise(
                samples,
                metric="swd",
                num_blocks=n_blocks,
                mode="true",
                ref_samples=true_samples,
                verbose=False,
            )
            history["true_swd_results"].append(swd_true_results)

        block_pairs = np.array(swd_results["block_pair"], dtype=object)
        sw_values = np.array(swd_results.get("est_all", swd_results.get("mean_all")), dtype=float)

        if wo_init_block:
            mask = np.array([not (pair[0] == 1 or pair[1] == 1) for pair in block_pairs], dtype=bool)
            block_pairs = block_pairs[mask]
            sw_values = sw_values[mask]

        bp_int = []
        for p in block_pairs:
            i, j = p
            if isinstance(j, str):  # skip "true" pairs etc
                continue
            bp_int.append((int(i), int(j)))
        block_pairs_int = np.array(bp_int, dtype=int)

        model = MonotoneSWDModel()
        est_result = model.fit_and_estimate(
            n_blocks,
            block_pairs_int,
            sw_values,
            return_bands=True,
            cred_interval=(10, 90),
        )
        history["estimated_swd_results"].append(est_result)

        # regression diagnostics
        sw_log = np.log(sw_values)
        i_idx = block_pairs_int[:, 0]
        j_idx = block_pairs_int[:, 1]
        reg_result = model.predicted_vs_residual(sw_log, i_idx, j_idx)
        history["regression_results"].append(reg_result)

        swd_to_stationary = float(est_result["swd_est_mean"][-1])
        history["swd_to_stationary"].append(swd_to_stationary)
        history["convg_check_iterate"].append(k)

        if verbose == 3:
            tqdm.write(f"[SWD] k={k} | final block distance={swd_to_stationary:.4g}")

        if swd_to_stationary < convg_threshold:
            history["k_conv"] = k
            history["convergence_reason"] = "swd"
            return k, "swd"

    return None, None


def swasd(
    samples,
    n_iters=None,
    n_blocks=6,
    n_projections=250,
    n_bootstrap=10,
    convg_threshold=1.0,
    min_iters_per_block=250,
    check_rate=1.2,
    block_mode="all_pairs",
    wo_init_block=True,
    diagnostic=True,
    true_swd=False,
    true_samples=None,
    rhat_threshold=1.01,
    rhat_wmin=100,
    rhat_method="rank",
    rhat_num_windows=5,
    rhat_stop_factor=1.5,
    rhat_check_iter=100,
    verbose=2,
):
    """
    Sliced-Wasserstein Automated Stationarity Detection (SWASD).

    Iteratively partitions the first k iterations of `samples` into `n_blocks`,
    computes pairwise Sliced-Wasserstein distances, fits the *monotone SWD model*
    to estimate the blockwise distances d[1]...d[B], and stops when the last
    block's estimated distance less than `convg_threshold`.
    
    Parameters
    ----------
    samples : `numpy.ndarray`, shape (draws, num_param) or (chains, draws, num_param)
        Markov chain or SGD samples.
    n_iters : int
        Maximum number of iterations to consider from `samples`.
    n_blocks : int, optional
        Number of blocks to partition samples into. Default: 6.
    n_projections : int, optional
        Number of projections for sliced Wasserstein distance. Default: 250.
    n_bootstrap : int, optional
        Bootstrap replicates for SW estimates. Default: 10.
    convg_threshold : float, optional
        Convergence criterion on last block's estimated distance (d[B]). Default: 1.0.
    min_iters_per_block : int, optional
        Minimum iterations per block before the first check. Default: 250.
    check_rate : float, optional
        Geometric growth factor for subsequent checks. Default: 1.2.
    block_mode : {"all_pairs","adjacent"}, optional
        Which block pairs to compare for SWD. Default: "all_pairs".
    wo_init_block : bool, optional
        Exclude any pair involving block 1 from regression fit. Default: True.
    diagnostic : bool, optional
        Store regression fit and predicted-vs-residual diagnostics. Default: True.
    true_swd : bool, optional
        If True, compare each block to provided stationary samples (requires `true_samples`).
    true_samples : np.ndarray, optional
        Stationary samples if `true_swd=True`.
    verbose : bool or int, optional
        Verbosity level:
        - False or 0: Silent
        - True or 1: Summary only (default)
        - 2: Progress bar
        - 3: Detailed output (all checks)
    """

    if isinstance(verbose, bool):
        verbose = 1 if verbose else 0
    else:
        verbose = int(verbose)

    # ---- validations ----
    if n_blocks < 2:
        raise ValueError("n_blocks must be >= 2.")
    if min_iters_per_block < 1:
        raise ValueError("min_iters_per_block must be >= 1.")
    if check_rate <= 1.0:
        raise ValueError("check_rate should be > 1.0.")
    if block_mode not in {"all_pairs", "adjacent"}:
        raise ValueError("block_mode must be 'all_pairs' or 'adjacent'.")
    if true_swd and true_samples is None:
        raise ValueError("true_swd=True requires `true_samples`.")

    if n_iters is None:
        n_iters = samples.shape[0] if samples.ndim == 2 else samples.shape[1]
    else:
        max_avail = samples.shape[0] if samples.ndim == 2 else samples.shape[1]
        if n_iters > max_avail:
            raise ValueError("n_iters exceeds available iterations in `samples`.")

    history = defaultdict(list)

    k0 = min_iters_per_block * n_blocks
    if k0 > n_iters:
        if verbose >= 1:
            print(f"Not enough iterations for first check: need {k0}, have {n_iters}.")
        history["k_conv"] = None
        return dict(history)

    # ---- checkpoints ----
    swasd_checks = _geometric_checkpoints(n_iters, k0, check_rate)

    rhat_stop = min(n_iters, int(np.ceil(rhat_stop_factor * k0)))
    rhat_checks = _linear_checkpoints(
        n_iters=n_iters,
        start=rhat_wmin,
        step=rhat_check_iter,
        stop=rhat_stop,
    )

    rhat_check_set = set(rhat_checks.tolist())
    swasd_check_set = set(swasd_checks.tolist())
    checkpoints = np.unique(np.concatenate([rhat_checks, swasd_checks])).astype(int)

    metric_comp = MetricComputer(n_projections, n_bootstrap)

    best_rhat_overall = None
    best_rhat_k = None

    pbar = None
    pbar_state = {}
    if verbose >= 2:
        pbar = tqdm(total=len(checkpoints), desc="SWASD Convergence Detection", unit="check", leave=True)

    for k in checkpoints:
        do_rhat = (k in rhat_check_set) and (k <= rhat_stop)
        do_swd = (k in swasd_check_set) and (k >= k0)

        if verbose >= 2 and pbar is not None:
            pbar_state["k"] = k
            if history["rhat_best"]:
                pbar_state["rhat"] = f"{history['rhat_best'][-1]:.3f}"
            if history["swd_to_stationary"]:
                pbar_state["SWD"] = f"{history['swd_to_stationary'][-1]:.3f}"
            pbar.set_postfix(pbar_state)
            pbar.update(1)
            
        xk = _slice_samples(samples, k)
        
        k_conv, reason = convergence_check(
            xk,
            do_rhat_check=do_rhat,
            do_swd_check=do_swd,
            history=history,
            metric_comp=metric_comp,
            n_blocks=n_blocks,
            block_mode=block_mode,
            wo_init_block=wo_init_block,
            convg_threshold=convg_threshold,
            true_swd=true_swd,
            true_samples=true_samples,
            rhat_threshold=rhat_threshold,
            rhat_wmin=rhat_wmin,
            rhat_method=rhat_method,
            rhat_num_windows=rhat_num_windows,
            verbose=verbose,
        )

        # track best rhat (for final reporting)
        if history["rhat_best"]:
            cur = history["rhat_best"][-1]
            if (best_rhat_overall is None) or (cur < best_rhat_overall):
                best_rhat_overall = cur
                best_rhat_k = k

        if k_conv is not None:
            if verbose >= 2 and pbar is not None:
                pbar.close()
            if verbose >= 1:
                _print_summary(
                    history,
                    k_conv,
                    reason,
                    rhat_threshold=rhat_threshold,
                    swd_threshold=convg_threshold,
                )
            return dict(history)

    if verbose >= 2 and pbar is not None:
        pbar.close()

    history["k_conv"] = None

    if verbose >= 1:
        print("\n" + "=" * 60)
        print("CONVERGENCE NOT DETECTED")
        print("=" * 60)
        if history["swd_to_stationary"]:
            print(f"Final SWD distance: {history['swd_to_stationary'][-1]:.4f}")
            print(f"Threshold: {convg_threshold:.4f}")
        if best_rhat_overall is not None:
            print(f"Best Rhat: {best_rhat_overall:.4f} at k={best_rhat_k}")
        print(f"Total checkpoints evaluated: {len(checkpoints)}")
        print("=" * 60)

    return dict(history)