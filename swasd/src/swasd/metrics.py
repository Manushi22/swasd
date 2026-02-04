import numpy as np
import ot
from tqdm import tqdm
import itertools
from .utils import compute_scale_from_last_blocks

class MetricComputer:
    """
    Compute a distance or divergence value between two sample sets.
    """
    def __init__(self, n_projections=100, n_bootstrap=1):
        """
        Parameters
        ----------
        n_projections : `int, optional
            Number of random projections used for sliced Wasserstein distances. The default is 100.
        n_bootstrap : `int`
            Number of bootstrap replicates. The default is 1.
        """
        self.n_projections = n_projections
        self.n_bootstrap = n_bootstrap
        self._metrics = {
            "swd": ot.sliced_wasserstein_distance,
            "max_swd": ot.sliced.max_sliced_wasserstein_distance
        }

    def compute_value(self, metric, X, Y, seed=1):
        """
        Parameters
        ----------
        metric : `str`
            Name of the metric to compute (must be a key in `self_metrics
        X : `numpy.ndarray`, shape (draws, num_param)
            First set of samples
        Y : `numpy.ndarray`, shape (draws, num_param)
                Second set of samples
        """
        if metric not in self._metrics:
            raise ValueError(f"Unsupported metric: {metric}")
        return self._metrics[metric](X, Y, n_projections=self.n_projections, seed=seed)

    def bootstrap(self, metric, X, Y, seed=1):
        """
        Parameters
        ----------
        metric : `str`
            Name of the metric to compute (must be a key in `self_metrics`)
        X : `numpy.ndarray`, shape (draws, num_param)
            First set of samples
        Y : `numpy.ndarray`, shape (draws, num_param)
                Second set of samples
        """
        rng = np.random.default_rng(seed)
        
        # Point estimate on original samples
        seed0 = int(rng.integers(0, 2**31 - 1))
        est = self.compute_value(metric, X, Y, seed=seed0)

        # If no bootstrap, keep your original return format
        if self.n_bootstrap <= 1:
            return dict(est=est, mean=est, median=est, std=0.0, ci=(est, est))

        # IID bootstrap of rows
        nX, nY = X.shape[0], Y.shape[0]
        estimates = np.empty(self.n_bootstrap, dtype=float)
        for b in range(self.n_bootstrap):
            idxX = rng.integers(0, nX, size=nX)
            idxY = rng.integers(0, nY, size=nY)
            seed_b = int(rng.integers(0, 2**31 - 1))
            estimates[b] = self.compute_value(metric, X[idxX], Y[idxY], seed=seed_b)

        return dict(
            est=float(est),
            mean=float(np.mean(estimates)),
            std=float(np.std(estimates, ddof=1)),
            ci=(float(np.percentile(estimates, 10)), float(np.percentile(estimates, 90))),
        )
        

    def compute_blockwise(self, samples, metric="swd", num_blocks=6, 
                          mode="adjacent", ref_samples=None, verbose=True):
        """
        Parameters
        ----------
        samples : `numpy.ndarray`, shape(draws, num_param)
            Markov chain samples
        metric : `str`
            Name of the metric to compute (must be a key in `self_metrics`)
        num_blocks : `int`
            Number of blocks to partition the samples into. The default is 6.
        mode : {'adjacent', 'true', 'all_pairs'}, default='adjacent'
            Defines which pairs of blocks to compare:
              - **'adjacent'** : compare each block with its immediate successor (i, i+1)
              - **'true'**     : compare each block to stationary samples (`stnry_samples`)
              - **'all_pairs'**: compare all unique pairs of blocks (i < j)
        ref_samples : `numpy.ndarray`, shape(draws, num_param), optional
            Reference stationary samples. Required if `mode='true'`.
        """
        results = {"est_all": [], "mean_all": [], "block_pair": [], 
                   "ci_upper": [], "ci_lower": []}
        
        if samples.ndim == 2:
            blocks = np.array_split(samples, num_blocks)
        else:
            raise ValueError("samples must be 2D")
        scale = compute_scale_from_last_blocks(samples, num_blocks)
        rescaled = [b / scale for b in blocks]

        if mode == "true":
            if ref_samples is None:
                raise ValueError("Reference stationary samples required for mode='true'")
            ref = ref_samples / scale
            pairs = [(i, "S") for i in range(1, num_blocks + 1)]
        elif mode == "all_pairs":
            pairs = list(itertools.combinations(range(1, num_blocks + 1), 2))
        else:
            pairs = [(i, i + 1) for i in range(1, num_blocks)]
                
        iterator = pairs 
        # if not verbose else tqdm(pairs, disable=not verbose)
        for pair in iterator:
            a = rescaled[pair[0] - 1]
            b = ref if pair[1] == "S" else rescaled[pair[1] - 1]
            stats = self.bootstrap(metric, a, b)
            results["est_all"].append(stats["est"])
            results["mean_all"].append(stats["mean"])
            results["block_pair"].append(pair)
            if self.n_bootstrap > 1:
                results["ci_lower"].append(stats["ci"][0])
                results["ci_upper"].append(stats["ci"][1])
        return results


