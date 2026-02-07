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

class swasd:
    """
    Sliced-Wasserstein Automated Stationarity Detection (SWASD).

    Iteratively partitions the first k iterations of `samples` into `n_blocks`,
    computes pairwise Sliced-Wasserstein distances, fits the *monotone SWD model*
    to estimate the blockwise distances d[1]...d[B], and stops when the last
    block's estimated distance less than `convg_threshold`.
    """

    def __init__(self, samples, n_iters=None, n_blocks=6, n_projections=250, n_bootstrap=10,
                 convg_threshold=1.0, min_iters_per_block=250, check_rate=1.2,
                 block_mode="all_pairs", wo_init_block=True, diagnostic=True,
                 true_swd=False, true_samples=None, rhat_threshold=1.01,
                 rhat_wmin=100, rhat_method="rank", rhat_num_windows=5,
                 rhat_stop_factor=1.5, rhat_check_iter=100, verbose=True):
        """
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
        self.samples = samples
        self.n_iters = n_iters
        self.n_blocks = n_blocks
        self.n_projections = n_projections
        self.n_bootstrap = n_bootstrap
        self.convg_threshold = convg_threshold
        self.min_iters_per_block = min_iters_per_block
        self.check_rate = check_rate
        self.block_mode = block_mode
        self.wo_init_block = wo_init_block
        self.diagnostic = diagnostic
        self.true_swd = true_swd
        self.true_samples = true_samples
        self.rhat_threshold = rhat_threshold
        self.rhat_wmin = rhat_wmin
        self.rhat_method = rhat_method
        self.rhat_num_windows = rhat_num_windows
        self.rhat_stop_factor = rhat_stop_factor
        self.rhat_check_iter = rhat_check_iter
        if isinstance(verbose, bool):
            self.verbose = 1 if verbose else 0
        else:
            self.verbose = int(verbose)
        self.history = defaultdict(list)

    def run(self):
        """Run the iterative SWASD convergence detection using the monotone model."""
        # validations
        if self.n_blocks < 2:
            raise ValueError("n_blocks must be >= 2.")
        if self.min_iters_per_block < 1:
            raise ValueError("min_iters_per_block must be >= 1.")
        if self.check_rate <= 1.0:
            raise ValueError("check_rate should be > 1.0 to space checks out.")
        if self.block_mode not in {"all_pairs", "adjacent"}:
            raise ValueError("block_mode must be 'all_pairs' or 'adjacent'.")
        if self.true_swd and self.true_samples is None:
            raise ValueError("true_swd=True requires `true_samples`.")

        if self.n_iters is None:
            self.n_iters = self.samples.shape[0] if self.samples.ndim == 2 else self.samples.shape[1]
        elif (self.samples.ndim == 2 and self.n_iters > self.samples.shape[0]) or \
             (self.samples.ndim == 3 and self.n_iters > self.samples.shape[1]):
            raise ValueError("n_iters exceeds available iterations in `samples`.")
            
        k0 = self.min_iters_per_block * self.n_blocks
        if k0 > self.n_iters:
            if self.verbose >= 1:
                print(f"Not enough iterations for first check: need {k0}, have {self.n_iters}.")
            self.history["k_conv"] = None
            return dict(self.history)

        swasd_checks = _geometric_checkpoints(self.n_iters, k0, self.check_rate)
        rhat_stop = min(self.n_iters, int(np.ceil(self.rhat_stop_factor * k0)))
        rhat_checks = _linear_checkpoints(n_iters=self.n_iters,
                                            start=self.rhat_wmin,
                                            step=self.rhat_check_iter,
                                            stop=rhat_stop,
                                            )
        rhat_check_set = set(rhat_checks.tolist())
        checkpoints = np.unique(np.concatenate([rhat_checks, swasd_checks])).astype(int)
        
        k_conv = None

        metric_comp = MetricComputer(self.n_projections, self.n_bootstrap)
        
        if self.verbose >= 2:
            pbar = tqdm(total=len(checkpoints), desc="SWASD Convergence Detection", 
                       unit="check", leave=True)
            pbar_state = {}
        
        best_rhat_overall = None
        best_rhat_k = None
        
        for checkpoint_idx, k in enumerate(checkpoints):
            if (k in rhat_check_set) and (k <= rhat_stop):
                xk = self.samples[:, :k, :] if self.samples.ndim == 3 else self.samples[:k, :]
                w_upper = int(0.95 * k)
        
                if w_upper > self.rhat_wmin:
                    windows = np.linspace(self.rhat_wmin, w_upper, num=self.rhat_num_windows, dtype=int)
                    success, best_w, best_rhat = rhat_check(
                        xk,
                        windows=windows,
                        rhat_threshold=self.rhat_threshold,
                        method=self.rhat_method,
                    )
        
                    self.history["rhat_check_iterate"].append(k)
                    self.history["rhat_best_w"].append(best_w)
                    self.history["rhat_best"].append(best_rhat)
        
                    if best_rhat_overall is None or best_rhat < best_rhat_overall:
                        best_rhat_overall = best_rhat
                        best_rhat_k = k
        
                    if self.verbose == 3:
                        tqdm.write(f"[Rhat] k={k} | rhat={best_rhat:.3f} (best_w={best_w})")
                    elif self.verbose == 2:
                        pbar_state['rhat'] = f'{best_rhat:.3f}'
                        pbar_state['k'] = k
                        pbar.set_postfix(pbar_state)
                        
                    if success:
                        k_conv = k
                        self.history["k_conv"] = k_conv
                        self.history["convergence_reason"] = "rhat"
                    
                        if self.verbose >= 2:
                            pbar.close()
                        if self.verbose >= 1:
                            self._print_summary(k_conv, "rhat", best_rhat=best_rhat)
                            
                        return dict(self.history)
                    
            if self.verbose >= 2:
                pbar_state['k'] = k
                pbar.set_postfix(pbar_state)
                pbar.update(1)
                
            if k < k0:
                continue

            if self.samples.ndim == 3:
                new_samples = self.samples[:, :k, :]
            else:
                new_samples = self.samples[:k, :]

            swd_results = metric_comp.compute_blockwise(
                new_samples,
                metric="swd",
                num_blocks=self.n_blocks,
                mode=self.block_mode,
                verbose=False,
            )
            self.history["pairwise_swd_results"].append(swd_results)
            
            if self.true_swd:
                swd_true_results = metric_comp.compute_blockwise(
                    new_samples,
                    metric="swd",
                    num_blocks=self.n_blocks,
                    mode="true",
                    ref_samples=self.true_samples if self.true_swd else None,
                    verbose=False,
                )
                self.history["true_swd_results"].append(swd_true_results)

            block_pairs = np.array(swd_results["block_pair"], dtype=object)
            sw_values = np.array(swd_results.get("est_all", swd_results.get("mean_all")), dtype=float)

            if self.wo_init_block:
                # Filter out any pair involving block 1
                mask = np.array([not (pair[0] == 1 or pair[1] == 1) for pair in block_pairs], dtype=bool)
                block_pairs = block_pairs[mask]
                sw_values = sw_values[mask]

            bp_int = []
            for p in block_pairs:
                i, j = p
                if isinstance(j, str):
                    continue
                bp_int.append((int(i), int(j)))
            block_pairs_int = np.array(bp_int, dtype=int)

            model = MonotoneSWDModel()
            est_result = model.fit_and_estimate(
                self.n_blocks,
                block_pairs_int,
                sw_values,
                return_bands=True,
                cred_interval=(10, 90),
            )
            self.history["estimated_swd_results"].append(est_result)

            # Regression diagnostics on log scale
            sw_log = np.log(sw_values)
            i_idx = block_pairs_int[:, 0]
            j_idx = block_pairs_int[:, 1]
            reg_result = model.predicted_vs_residual(
                sw_log, i_idx, j_idx
            )
            self.history["regression_results"].append(reg_result)

            swd_to_stationary = float(est_result["swd_est_mean"][-1])
            self.history["swd_to_stationary"].append(swd_to_stationary)
            self.history["convg_check_iterate"].append(k)
            
            if self.verbose == 3:
                msg = f" k = {k} iterations| final block distance: {swd_to_stationary:.4g}"
                tqdm.write(msg)
            elif self.verbose == 2:
                pbar_state['SWD'] = f'{swd_to_stationary:.3f}'
                pbar_state['k'] = k
                pbar.set_postfix(pbar_state)
            
            if swd_to_stationary < self.convg_threshold:
                k_conv = k
                self.history["convergence_reason"] = "swd"
                if self.verbose >= 2:
                    pbar.close()
                if self.verbose >= 1:
                    self._print_summary(k_conv, "swd", swd=swd_to_stationary)
                    
                break
        
        if self.verbose >= 2:
            pbar.close()
        
        self.history["k_conv"] = k_conv
        
        if k_conv is None and self.verbose >= 1:
            print("\n" + "="*60)
            print("CONVERGENCE NOT DETECTED")
            print("="*60)
            if self.history["swd_to_stationary"]:
                print(f"Final SWD distance: {self.history['swd_to_stationary'][-1]:.4f}")
                print(f"Threshold: {self.convg_threshold:.4f}")
            if best_rhat_overall is not None:
                print(f"Best Rhat: {best_rhat_overall:.4f} at k={best_rhat_k}")
            print(f"Total checkpoints evaluated: {len(checkpoints)}")
            print("="*60)
            
        return dict(self.history)
    
    def _print_summary(self, k_conv, reason, **kwargs):
        """Print a clean convergence summary."""
        print("\n" + "="*60)
        print("CONVERGENCE DETECTED!")
        print("="*60)
        print(f"Convergence at iteration: {k_conv}")
        print(f"Detection method: {reason.upper()}")
        
        if reason == "rhat":
            rhat = kwargs.get('best_rhat')
            if rhat is not None:
                print(f"Final Rhat: {rhat:.4f} (threshold: {self.rhat_threshold:.4f})")
        elif reason == "swd":
            swd = kwargs.get('swd')
            if swd is not None:
                print(f"Final SWD distance: {swd:.4f} (threshold: {self.convg_threshold:.4f})")
        
        # Summary statistics
        if self.history["rhat_best"]:
            print(f"\nRhat checks performed: {len(self.history['rhat_best'])}")
            print(f"Best Rhat achieved: {min(self.history['rhat_best']):.4f}")
        
        if self.history["swd_to_stationary"]:
            print(f"\nSWD checks performed: {len(self.history['swd_to_stationary'])}")
            print(f"Initial SWD: {self.history['swd_to_stationary'][0]:.4f}")
            print(f"Final SWD: {self.history['swd_to_stationary'][-1]:.4f}")
        
        print("="*60)