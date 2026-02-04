import numpy as np
import stan
import arviz as az
import logging
import io
from contextlib import redirect_stdout, redirect_stderr
from .utils import load_stan_text

# Suppress HTTPStan and Stan logging
logging.getLogger('httpstan').setLevel(logging.ERROR)
logging.getLogger('stan').setLevel(logging.ERROR)


class MonotoneSWDModel:
    """
    Monotone regression for blockwise SWD-to-stationary curve d[1..B] using PyStan.

    Model (log scale):
      log_d[1] ~ Normal(d_scale, 1)
      delta[k] ~ Normal(mu_delta, sigma_delta), k=1..B-1
      log_d[k] = log_d[1] - sum_{m=1}^{k-1} delta[m]
      d = exp(log_d)  (monotone decreasing, positive)

    Observations:
      y_n = log(SWD_hat(i_n, j_n)) ~ Normal( log( |d[i_n] - d[j_n]| ), sigma )
    """
    def __init__(self, num_samples=2000, num_warmup=2000, num_chains=4, seed=1234, verbose=False):
        """
        Parameters
        ----------
        num_samples : int
            Number of MCMC samples per chain
        num_warmup : int
            Number of warmup iterations per chain
        num_chains : int
            Number of MCMC chains
        seed : int
            Random seed for Stan sampling
        verbose : bool
            If True, show Stan sampling output. If False, suppress all output.
        """
        self.num_samples = int(num_samples)
        self.num_warmup = int(num_warmup)
        self.num_chains = int(num_chains)
        self.seed = int(seed)
        self.verbose = verbose
        
        self.fit_ = None
        self.idata = None
        self._B = None  # number of blocks cached after fit
        
    def fit(self, y, i, j, *, random_seed=1234, adapt_delta=0.95, max_depth=10):
        """
        Parameters
        ----------
        y : `numpy.ndarray`, shape(N,)
            Observed pairwise log(SWD)
        i : `numpy.ndarray`, shape(N,)
            First block index (1-based)
        j : `numpy.ndarray`, shape(N,)
            Second block index, with j > i
        random_seed : int
            Random seed for this fit
        adapt_delta : float
            Stan adaptation parameter
        max_treedepth : int
            Stan max tree depth parameter
        """
        y = np.asarray(y, dtype=float)
        i = np.asarray(i, dtype=int)
        j = np.asarray(j, dtype=int)

        if i.min() < 1:
            i = i + 1
        if j.min() < 1:
            j = j + 1

        B = int(max(i.max(), j.max()))
        self._B = B
        N = int(y.shape[0])
        
        stan_code = load_stan_text("monotone_swd.stan")
        data = dict(N=N, B=B, y=y, i=i, j=j)
        
        posterior = stan.build(stan_code, data=data, random_seed=random_seed)
        
        # Suppress Stan output unless verbose mode is on
        if not self.verbose:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                fit = posterior.sample(
                    num_chains=self.num_chains,
                    num_warmup=self.num_warmup,
                    num_samples=self.num_samples,
                    delta=adapt_delta,
                    max_depth=max_depth,
                )
        else:
            fit = posterior.sample(
                num_chains=self.num_chains,
                num_warmup=self.num_warmup,
                num_samples=self.num_samples,
               delta=adapt_delta,
                max_epth=max_depth,
            )
        
        self.fit_ = fit
        
        try:
            self.idata = az.from_pystan(
                posterior=fit,
                posterior_predictive="y_rep",
                observed_data={"y": y},
                log_likelihood="log_lik",
            )
        except Exception:
            self.idata = None
        return fit
    
    def _check_fitted(self):
        if self.fit_ is None:
            raise RuntimeError("Call fit() first.")
            
    def compute_swd_est_ci(self, block_array, cred_interval=(10, 90)):
        """
        Credible bands for the monotone SWD-to-stationary curve d[b].

        Parameters
        ----------
        block_array : (B,) arraylike of block indices (1..B)
        cred_interval : (low, high) percentiles
        """
        self._check_fitted()
        d_samps = self.fit_["d"]               # shape: (chains, draws, B)
        d_flat = d_samps.reshape(-1, d_samps.shape[-1]).T  # (S, B)
        
        mean = d_flat.mean(axis=0)
        median = np.median(d_flat, axis=0)
        lower = np.percentile(d_flat, cred_interval[0], axis=0)
        upper = np.percentile(d_flat, cred_interval[1], axis=0)
        return mean, median, lower, upper


    def fit_and_estimate(self, num_blocks, block_pairs, sw_values,
                         return_bands=True, cred_interval=(10, 90),
                         random_seed=None, adapt_delta=0.9, max_depth=10):
        """
        Fit monotone model and return the estimated d[b] curve.
        """
        y_log = np.log(np.asarray(sw_values, dtype=float))
        block_pairs = np.asarray(block_pairs, dtype=int)
        i = block_pairs[:, 0]
        j = block_pairs[:, 1]

        # Fit
        self.fit(y_log, i, j,
                 random_seed=self.seed if random_seed is None else random_seed,
                 adapt_delta=adapt_delta,
                 max_depth=max_depth)

        # Extract SWD_i summaries
        blocks = np.arange(1, num_blocks + 1, dtype=float)
        mean, median, lower, upper = self.compute_swd_est_ci(blocks, cred_interval)

        out = dict(
            swd_est_mean=mean,
            swd_est_median=median,
            swd_lci=lower,
            swd_uci=upper,
            idata=self.idata,
        )
        return out

    # ---------------------------
    # Fitted vs residuals (log scale)
    # ---------------------------
    def predicted_vs_residual(self, y_log, i, j, compute_fit_diagnostic=True):
        """
        Posterior fitted mean and residuals for y=log(SWD).
        
        Parameters
        ----------
        y_log : np.ndarray
            Observed log(SWD) values
        i : np.ndarray
            Block indices i
        j : np.ndarray
            Block indices j
        compute_fit_diagnostic : bool
            Whether to compute fit diagnostic metric
        """
        self._check_fitted()  # Make sure model is fitted
        
        y_fit = self.fit_["y_fit"].T
    
        y_pred_mean = y_fit.mean(axis=0)
        lower_CI = np.percentile(y_fit, 10, axis=0)
        upper_CI = np.percentile(y_fit, 90, axis=0)
        residuals = y_log - y_pred_mean  # FIXED: was y - y_pred_mean
    
        out = {
            "y_predicted": y_pred_mean,
            "residuals": residuals,
            "lower_CI": lower_CI,
            "upper_CI": upper_CI,
        }

        if compute_fit_diagnostic:
            N = len(y_log)
            y_std = np.std(y_log, ddof=1) if N > 1 else 1.0
            predictive_std = np.std(y_fit, axis=0)
            norm_uncertainty = np.linalg.norm(predictive_std, ord=2)
            rms_uncertainty = norm_uncertainty / np.sqrt(max(N, 1))
            relative_rms = rms_uncertainty / y_std if y_std > 0 else np.nan
            out["fit_diagnostic"] = relative_rms

        return out
