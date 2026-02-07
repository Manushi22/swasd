# SWASD: Sliced-Wasserstein Automated Stationarity Detection

**SWASD** (Sliced-Wasserstein Automated Stationarity Detection) is a Python package that provides an automated framework for detecting convergence of Markov chains to their stationary distributions. SWASD is applicable to both Markov Chain Monte Carlo (MCMC) and fixed-learning-rate stochastic optimization (FLSO) algorithms.

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
- `POT >= 0.9.3` (Python Optimal Transport)
- `xarray >= 2023.7.0`

## Examples

See the `notebook/` directory for detailed example demonstrating usage of SWASD using FLSO updates.
