import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MaxNLocator
from typing import Optional, Dict, Any, Tuple, List, Union
import warnings
import seaborn as sns

# Use your old seaborn palette
_SNS_COLORS = sns.color_palette("colorblind")

DEFAULT_COLORS = {
    "primary":   _SNS_COLORS[0],
    "secondary": _SNS_COLORS[1],
    "accent":    _SNS_COLORS[2],
    "success":   _SNS_COLORS[2],  # or pick a different index you like
    "warning":   _SNS_COLORS[3],
    "true":      _SNS_COLORS[0],
    "estimated": _SNS_COLORS[3],
    "ci":        _SNS_COLORS[3],
    "threshold": _SNS_COLORS[7] if len(_SNS_COLORS) > 7 else _SNS_COLORS[-1],
}

def swd_comparison_plot(
    swd_estimated: np.ndarray,
    swd_true: np.ndarray,
    ax: Optional[plt.Axes] = None,
    xlog: bool = False,
    ylog: bool = False,
    add_diagonal: bool = True,
    title: Optional[str] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot comparison between estimated and true SWD values.
    
    Parameters
    ----------
    swd_estimated : np.ndarray or dict
        Estimated SWD values. Can be array or dict with 'swd_est_mean' key.
    swd_true : np.ndarray or dict
        True SWD values. Can be array or dict with 'mean_all' key.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new figure.
    xlog : bool, default=False
        Use log scale for x-axis.
    ylog : bool, default=False
        Use log scale for y-axis.
    add_diagonal : bool, default=True
        Add y=x diagonal reference line.
    title : str, optional
        Plot title.
    figsize : tuple, default=(8, 6)
        Figure size if creating new figure.
    **kwargs
        Additional keyword arguments passed to scatter plot.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
        
    Examples
    --------
    >>> fig, ax = swd_comparison_plot(estimated, true, xlog=True, ylog=True)
    >>> ax.set_title("My Custom Title")
    >>> plt.savefig("comparison.png", dpi=300, bbox_inches='tight')
    """
    # Handle dict inputs
    if isinstance(swd_estimated, dict):
        swd_est_vals = swd_estimated.get('swd_est_mean', swd_estimated.get('mean', None))
        swd_est_vals = np.asarray(swd_est_vals)
        if swd_est_vals is None:
            raise ValueError("Dict must contain 'swd_est_mean' or 'mean' key")
    else:
        swd_est_vals = np.asarray(swd_estimated)
    
    if isinstance(swd_true, dict):
        swd_true_vals = swd_true.get('mean_all', swd_true.get('mean', None))
        swd_true_vals = np.asarray(swd_true_vals)
        if swd_true_vals is None:
            raise ValueError("Dict must contain 'mean_all' or 'mean' key")
    else:
        swd_true_vals = np.asarray(swd_true)
    
    # Validate shapes
    if swd_est_vals.shape != swd_true_vals.shape:
        warnings.warn(f"Shape mismatch: estimated {swd_est_vals.shape} vs true {swd_true_vals.shape}")
        min_len = min(len(swd_est_vals), len(swd_true_vals))
        swd_est_vals = swd_est_vals[:min_len]
        swd_true_vals = swd_true_vals[:min_len]
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    
    # Plot scatter
    scatter_kwargs = {
        'alpha': 1,
        's': 80,
        'color': DEFAULT_COLORS['primary'],
        'edgecolors': 'white',
        'linewidths': 0.5,
    }
    scatter_kwargs.update(kwargs)
    
    ax.scatter(swd_true_vals, swd_est_vals, **scatter_kwargs)
    
    # Add diagonal reference line
    if add_diagonal:
        x = np.asarray(swd_true_vals, dtype=float)
        y = np.asarray(swd_est_vals, dtype=float)

        if xlog or ylog:
            # For log plots, ignore nonpositive values
            mask = (x > 0) & (y > 0)
            x = x[mask]
            y = y[mask]

        lo = np.nanmin(np.concatenate([x, y]))
        hi = np.nanmax(np.concatenate([x, y]))

        ax.plot([lo, hi], [lo, hi],
                '--', color="red", alpha=1, linewidth=2,
                label='y=x', zorder=0)
        ax.legend()
    
    # Set scales
    if xlog:
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    if ylog:
        ax.set_yscale('log')
    
    # Labels
    ax.set_xlabel(r'$\mathrm{SWD}(\pi_{(i)}, \pi)$')
    ax.set_ylabel(r'$\widehat{\mathrm{SWD}}(\pi_{(i)}, \pi)$')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Improve tick labels
    ax.tick_params(axis='x', rotation=45)
    plt.setp(ax.get_xticklabels(), ha='right')
    
    # ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    return fig, ax


def regression_diagnostic_plot(
    pairwise_results: Dict[str, Any],
    regression_results: Dict[str, Any],
    num_blocks: int = 6,
    ax: Optional[plt.Axes] = None,
    skip_first_block: bool = True,
    show_ci: bool = True,
    xlog: bool = False,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot regression fit vs observed pairwise SWD values.
    
    Parameters
    ----------
    pairwise_results : dict
        Results from pairwise SWD computation with keys:
        - 'mean_all': observed SWD values
        - 'block_pair': list of (i, j) tuples
    regression_results : dict
        Results from regression with keys:
        - 'y_predicted': predicted log-SWD values
        - 'upper_CI', 'lower_CI': confidence intervals (optional)
    num_blocks : int, default=6
        Total number of blocks.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on.
    skip_first_block : bool, default=True
        Skip pairs involving block 1.
    show_ci : bool, default=True
        Show confidence intervals.
    xlog : bool, default=False
        Use log scale for x-axis (pair indices).
    title : str, optional
        Plot title.
    figsize : tuple, default=(10, 6)
        Figure size if creating new figure.
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    
    # Extract data
    index = num_blocks - 1 if skip_first_block else 0
    y_true = np.asarray(pairwise_results["mean_all"][index:])
    y_pred = np.exp(regression_results["y_predicted"])
    block_pairs = np.array(pairwise_results["block_pair"][index:])
    
    # Create x positions and labels
    x = np.arange(len(y_true))
    labels = [f"{i}-{j}" for i, j in block_pairs]
    
    # Plot observed values
    ax.scatter(x, y_true, s=80, color=DEFAULT_COLORS['true'], 
              label='Observed', marker='o', edgecolors='white')
    
    # Plot fitted values
    ax.scatter(x, y_pred, alpha=0.9, s=80, color=DEFAULT_COLORS['estimated'],
              label='Fitted', marker='x', linewidths=2)
    
    # Add confidence intervals
    if show_ci and 'upper_CI' in regression_results and 'lower_CI' in regression_results:
        y_pred_uci = np.exp(regression_results["upper_CI"])
        y_pred_lci = np.exp(regression_results["lower_CI"])
        
        ax.vlines(x, y_pred_lci, y_pred_uci, 
                 color=DEFAULT_COLORS['ci'], linestyle='--', alpha=0.6,
                 linewidth=1.5, label='80% CI')
    
    # Formatting
    ax.set_xlabel('Block Pairs')
    ax.set_ylabel(r'$\mathrm{SWD}(\hat{\pi}_{(i)}, \hat{\pi}_{(j)})$')
    ax.set_yscale('log')
    
    if xlog:
        ax.set_xscale('log')
    
    # Set ticks
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.legend(loc='upper center', 
                   bbox_to_anchor=(0.5, 1.2), ncol=3)
    # ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    
    return fig, ax


def convergence_plot(
    history: Dict[str, List],
    threshold: float = 1.0,
    ax: Optional[plt.Axes] = None,
    show_true_swd: bool = True,
    show_ci: bool = True,
    xlog: bool = False,
    ylog: bool = False,
    mark_convergence: bool = True,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot SWD convergence over iterations.
    
    Improved features:
    - Handles both estimated and true SWD
    - Optional convergence marker
    - Better legend positioning
    - Flexible data extraction
    
    Parameters
    ----------
    history : dict
        SWASD results dictionary with keys:
        - 'convg_check_iterate': iteration numbers
        - 'estimated_swd_results': list of estimation results
        - 'true_swd_results': list of true SWD results (optional)
        - 'k_conv': convergence iteration (optional)
    threshold : float, default=1.0
        Convergence threshold to display.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on.
    show_true_swd : bool, default=True
        Plot true SWD if available.
    show_ci : bool, default=True
        Show credible intervals for estimates.
    xlog : bool, default=False
        Use log scale for x-axis.
    ylog : bool, default=False
        Use log scale for y-axis.
    mark_convergence : bool, default=True
        Mark convergence point if detected.
    title : str, optional
        Plot title.
    figsize : tuple, default=(10, 6)
        Figure size.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    
    # Extract iteration numbers
    iterations = np.array(history.get('convg_check_iterate', []))
    
    if len(iterations) == 0:
        warnings.warn("No convergence check iterations found in history")
        return fig, ax
    
    # Extract estimated SWD
    if 'estimated_swd_results' in history:
        # Extract from detailed results
        est_results = history['estimated_swd_results']
        swd_est_mean = np.array([r["swd_est_mean"][-1] for r in est_results])
        if show_ci and 'swd_lci' in est_results[0]:
            swd_est_lci = np.array([r["swd_lci"][-1] for r in est_results])
            swd_est_uci = np.array([r["swd_uci"][-1] for r in est_results])
        else:
            swd_est_lci = swd_est_uci = None
    else:
        raise ValueError("history must contain 'estimated_swd_results'")
    
    # Plot estimated SWD
    ax.plot(iterations, swd_est_mean, color=_SNS_COLORS[9],
           linestyle='--', linewidth=2, label=r'$\widehat{SWD}(\pi_{(B)},\pi)$')
    
    # Add confidence intervals
    if show_ci and swd_est_lci is not None and swd_est_uci is not None:
        ax.fill_between(iterations, swd_est_lci, swd_est_uci,
                       color=_SNS_COLORS[9], alpha=0.2,
                       label='80% CI')
    
    # Plot true SWD if available
    if show_true_swd and 'true_swd_results' in history:
        true_results = history['true_swd_results']
        if len(true_results) > 0:
            swd_true = np.array([r["mean_all"][-1] for r in true_results])
            ax.plot(iterations, swd_true, color=DEFAULT_COLORS['true'],
                   linestyle='-', linewidth=2, label=r'$SWD(\pi_{(B)},\pi)$')
    
    # Add threshold line
    ax.axhline(threshold, linestyle=':', color=DEFAULT_COLORS['threshold'],
              alpha=0.8, linewidth=2,
              label=f'$\\varepsilon={threshold:.2f}$')
    
    # Mark convergence
    # if mark_convergence and 'k_conv' in history and history['k_conv'] is not None:
    #     k_conv = history['k_conv']
    #     ax.axvline(k_conv, linestyle='--', color=DEFAULT_COLORS['success'],
    #               alpha=0.7, linewidth=2, label=f'Convergence (k={k_conv})')
    
    # Formatting
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Diagnostic Value')
    
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
     
    # Legend
    ax.legend(loc='upper center', 
                   bbox_to_anchor=(0.5, 1.3), ncol=2)
    
    # ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # fig.tight_layout()
    
    return fig, ax


def trace_plot(
    samples: np.ndarray,
    dims: Optional[List[int]] = None,
    k_conv: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    xlog: bool = False,
    max_dims: int = 5,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot parameter traces over iterations.
    
    Improved features:
    - Automatic dimension selection
    - Better color cycling
    - Convergence marker
    - Handles 2D and 3D arrays
    
    Parameters
    ----------
    samples : np.ndarray
        Samples array of shape (T, d) or (C, T, d).
    dims : list of int, optional
        Dimensions to plot. If None, plots first `max_dims` dimensions.
    k_conv : int, optional
        Convergence iteration to mark.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on.
    xlog : bool, default=False
        Use log scale for x-axis.
    max_dims : int, default=5
        Maximum dimensions to plot if dims not specified.
    title : str, optional
        Plot title.
    figsize : tuple, default=(12, 6)
        Figure size.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    
    # Handle 3D samples (take first chain)
    if samples.ndim == 3:
        warnings.warn("3D samples detected, using first chain for trace plot")
        X = samples[0, :, :]
    else:
        X = samples
    
    # Determine dimensions to plot
    n_params = X.shape[1]
    if dims is None:
        dims = list(range(min(max_dims, n_params)))
    
    # Create color map
    colors = plt.cm.tab10(np.linspace(0, 1, len(dims)))
    
    # Plot traces
    for idx, dim in enumerate(dims):
        ax.plot(X[:, dim], linewidth=1.5, alpha=0.8,
               color=colors[idx], label=f'Parameter {dim}')
    
    # Mark convergence
    if k_conv is not None:
        ax.axvline(k_conv, linestyle='--', linewidth=2.5,
                  color=DEFAULT_COLORS['success'], alpha=0.8,
                  label=f'Convergence (k={k_conv})', zorder=10)
    
    # Formatting
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Parameter Value')
    
    if xlog:
        ax.set_xscale('log')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    
    return fig, ax

# set_plot_style()









# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import LogLocator
# import seaborn as sns

# colors = sns.color_palette("colorblind")

# def swd_est_true_comparison_plot(swd_estimated_convg_results,
#                                  swd_true_convg_results,
#                                  xlog=False, ylog=False):
#     swd_est_mean = swd_estimated_convg_results["swd_est_mean"]
#     swd_true = swd_true_convg_results["mean_all"]
    
#     plt.scatter(swd_true, swd_est_mean, color=colors[0])
#     plt.plot(swd_true, swd_true, linestyle="--", color="red")
#     if xlog:
#         plt.xscale("log")
#     if ylog:
#         plt.yscale("log")
#     plt.xlabel(r"$SWD(\pi_{(i)},\pi)$")
#     plt.ylabel(r"$\widehat{SWD}(\pi_{(i)},\pi)$")
#     subs = np.arange(1, 10)
#     plt.gca().xaxis.set_major_locator(LogLocator(base=10.0, subs=subs, numticks=12))
#     plt.xticks(rotation=45, ha="right")
#     plt.show()
#     plt.close()

# def regression_true_vs_fitted_plot(swd_pairwise_convg_results,
#                                    swd_regression_convg_results,
#                                    num_blocks = 6,
#                                    xlog=True, xlabel=None, label=None):
#     index = num_blocks - 1
#     y_true = swd_pairwise_convg_results["mean_all"][index:]
#     y_pred = np.exp(swd_regression_convg_results["y_predicted"])
#     y_pred_uci = np.exp(swd_regression_convg_results["upper_CI"])
#     y_pred_lci = np.exp(swd_regression_convg_results["lower_CI"])
#     block_pairs = np.array(swd_pairwise_convg_results["block_pair"])[index:]
#     pairs = np.asarray(block_pairs)
    
#     labels = [f"{i}-{j}" for i, j in pairs]
#     x = np.arange(len(labels))
    
#     plt.scatter(x, y_true, alpha=1, color=colors[0])
#     plt.scatter(x, y_pred, alpha=1, color=colors[3], marker="x")#, label=label)
#     plt.vlines(x, y_pred_lci, y_pred_uci, color=colors[3], 
#                 linestyle="--")
#     if xlabel is not None:
#         plt.xlabel(xlabel)
#     else:   
#         plt.xlabel("Block Pairs")
#     plt.ylabel(r"$SWD(\hat{\pi}_{(i)}, \hat{\pi}_{(j)})$")
#     if xlog:
#         plt.xscale("log")
#     plt.yscale("log")
#     plt.xticks(x, labels, rotation=45, ha="right")
#     plt.title(label=label)
#     plt.show()
#     plt.close()

# def swd_est_plot(results, iterations,
#                   conv_threshld=1.0, xlog=False, ylog=False):
    
#     swd_est_mean = [results['estimated_swd_results'][i]["swd_est_mean"][-1] for i in range(len(iterations))]
#     swd_est_lci = [results['estimated_swd_results'][i]["swd_lci"][-1] for i in range(len(iterations))]
#     swd_est_uci = [results['estimated_swd_results'][i]["swd_uci"][-1] for i in range(len(iterations))]
#     swd_true = [results['true_swd_results'][i]["mean_all"][-1] for i in range(len(iterations))]
    
#     plt.plot(iterations, swd_true, color=colors[0], label=r"$SWD(\pi_{(B)},\pi)$", linewidth=2)
#     plt.plot(iterations, swd_est_mean, color=colors[9], linestyle="--", label=r"$\widehat{SWD}(\pi_{(B)},\pi)$")
#     plt.fill_between(iterations, swd_est_lci, swd_est_uci, color=colors[9], alpha=0.25)

#     plt.axhline(conv_threshld, linestyle="dashed", color="black", alpha=0.6, 
#                 label=r"$\varepsilon={:.2f}$".format(conv_threshld))
#     plt.xticks(rotation=45, ha="right")
#     if xlog:
#         plt.xscale("log")
#     if ylog:
#         plt.yscale("log")
#     plt.xlabel("Iteration")
#     plt.ylabel("Diagnostic Value")
#     plt.legend(loc='upper center', 
#                bbox_to_anchor=(0.5, 1.3), ncol=2, 
#                frameon=False)
#     plt.show()
#     plt.close()

# def plot_iterate_trace(X, dims=(0, 1, 2, 3, 4), k_conv=None, xlog=True):
#     plt.figure()
#     for j in dims:
#         plt.plot(X[:, j], linewidth=1.3, label=f"dim {j}")
#     if k_conv is not None:
#         plt.axvline(k_conv, linestyle="--", linewidth=2, label=f"k_conv={k_conv}")
#     plt.xlabel("iteration")
#     plt.ylabel("value")
#     if xlog:
#         plt.xscale('log')
#     plt.show()
