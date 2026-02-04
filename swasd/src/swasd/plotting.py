import numpy as np
import matplotlib.pyplot as plt

def regression_true_vs_fitted_plot(
    y_true, y_pred, y_pred_lci, y_pred_uci, block_pairs,
    xlog=False, ylog=True, xlabel="Block Pairs", title=None, save_path=None
):
    """
    Plot observed SWD vs fitted SWD (with CI) by block pair label "i-j".
    """
    y_true  = np.asarray(y_true, dtype=float)
    y_pred  = np.asarray(y_pred, dtype=float)
    y_lci   = np.asarray(y_pred_lci, dtype=float)
    y_uci   = np.asarray(y_pred_uci, dtype=float)
    pairs   = np.asarray(block_pairs)

    labels = [f"{int(i)}-{int(j)}" for i, j in pairs]
    x = np.arange(len(labels))

    fig, ax = plt.subplots()
    ax.scatter(x, y_true,  alpha=0.9, s=28, label="Observed SWD")
    ax.scatter(x, y_pred,  alpha=0.9, s=28, marker="x", label="Fitted SWD")
    ax.vlines(x, y_lci, y_uci, linestyles="--", alpha=0.7, label="80% CI")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\mathrm{SWD}(\hat{\pi}_{(i)}, \hat{\pi}_{(j)})$")
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False)

    fig.tight_layout()
    plt.show()


def residuals_vs_fitted_plot(
    fitted_values, residuals, xlabel="Fitted Values",
    ylabel="Residuals", title=None, save_path=None
):
    fitted_values = np.asarray(fitted_values, dtype=float)
    residuals     = np.asarray(residuals, dtype=float)

    fig, ax = plt.subplots()
    ax.scatter(fitted_values, residuals, alpha=0.7, s=24)
    ax.axhline(0.0, linestyle="--", linewidth=1, color="black", alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    fig.tight_layout()
    plt.show()
