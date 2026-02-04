# SWASD: Sliced-Wasserstein Automated Stationarity Detection

**SWASD** (Sliced-Wasserstein Automated Stationarity Detection) is a Python package that provides an automated framework for detecting convergence of Markov chains to their stationary distributions. SWASD is applicable to both Markov Chain Monte Carlo (MCMC) and fixed-learning-rate stochastic optimization (FLSO) algorithms.

## Overview

Traditional convergence diagnostics like $\hat{R}$ (Gelman-Rubin statistic) assess whether multiple chains have forgotten their initial conditions, but they don't directly measure the distance to the stationary distribution. SWASD addresses this by:

- **Quantifying distributional distance**: Uses the sliced Wasserstein distance to measure how close samples are to the stationary distribution
- **Automated detection**: Continuously monitors convergence and provides early stopping criteria
- **Handling high dimensions**: The sliced Wasserstein distance is computationally efficient and has sample complexity independent of dimensionality
- **Interpretable threshold**: Provides a natural, interpretable convergence threshold based on relative mean and scale errors

## Key Features

- **Quantitative convergence measure**: Estimates the actual distance between current samples and the stationary distribution  
- **Early convergence detection**: Often declares convergence earlier than traditional diagnostics  
- **Broadly applicable**: Works with MCMC and fixed learning rate stochastic  optimization
- **Efficient computation**: Uses sliced Wasserstein distance for scalability  
- **Robust estimation**: Employs block-based pairwise comparisons and Bayesian regression  

## Installation

### From PyPI (once published)

```bash
pip install swasd
```

### From source

```bash
git clone https://github.com/Manushi22/swasd.git
cd swasd
pip install -e .
```

### Dependencies

SWASD requires Python 3.10+ and the following packages:

- `numpy >= 2.0`
- `scipy >= 1.10`
- `matplotlib >= 3.7`
- `tqdm >= 4.66`
- `arviz >= 0.16`
- `pystan >= 3.7`
- `httpstan >= 4.10`
- `POT >= 0.9.3` (Python Optimal Transport)
- `xarray >= 2023.7.0`

## Quick Start

### Basic Usage

```python
import numpy as np
from swasd import swasd

# Your MCMC or optimization samples (T iterations × d parameters)
samples = np.random.randn(1000, 5)

# Create detector
detector = swasd(
    samples=samples,
    n_blocks=6,              # Number of blocks to partition samples
    n_projections=250,       # Number of random projections for SW distance
    convg_threshold=1.0,     # Convergence threshold
    min_iters_per_block=250  # Minimum iterations per block
)

# Run convergence detection
result = detector.run()

# Check if converged
if result["k_conv"] is not None:
    print(f"Converged at iteration {result['k_conv']}")
else:
    print("Did not converge within the iterations")

# View convergence history
print("SW distances to stationarity:", result["swd_to_stationary"])
```

### The Sliced Wasserstein Distance

The sliced Wasserstein (SW) distance is defined as:

$$\text{SW}_{p,\Sigma}(\eta, \zeta) = \left(\int_{S^{d-1}} W_p(\alpha_*\sharp\eta, \alpha_*\sharp\zeta)^p \, d\lambda(\alpha)\right)^{1/p}$$

where:
- $W_p$ is the Wasserstein distance
- $\alpha_*\sharp\eta$ is the pushforward of distribution $\eta$ by projection $\alpha$
- $\Sigma$ is a scaling matrix (estimated from the data)

## Output

The `run()` method returns a dictionary containing:

- **`k_conv`**: Iteration at which convergence was detected (or `None`)
- **`swd_to_stationary`**: List of estimated SW distances at each checkpoint
- **`convg_check_iterate`**: List of iteration numbers where checks occurred
- **`pairwise_swd_results`**: Pairwise block distance computations
- **`estimated_swd_results`**: Regression model estimates with credible intervals
- **`regression_results`**: Diagnostic information from the regression fits

## Examples

See the `examples/` directory for detailed notebooks demonstrating:

- MCMC convergence detection
- Black-box variational inference (BBVI) with stochastic optimization
- Comparison with $\hat{R}$ diagnostic
- Visualization of convergence diagnostics

## Citation









```
