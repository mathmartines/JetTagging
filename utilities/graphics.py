from typing import Dict
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.lines import Line2D
import os
import numpy as np

os.environ['PATH'] = f"/Library/TeX/texbin:{os.environ['PATH']}"
rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Times']})
plt.rcParams.update({'font.size': 15})


def plot_roc_curve(
        signal_eff: Dict[str, np.ndarray], background_rej: Dict[str, np.ndarray], labels: Dict[str, str],
        colors: Dict[str, str], linestyle, title, file_path=None
):
    """Plots the roc curve for a list of classifiers"""

    for classifier in signal_eff:
        plt.plot(signal_eff[classifier], background_rej[classifier], label=labels[classifier],
                 color=colors[classifier], linestyle=linestyle[classifier])
    plt.plot([0, 1], [1, 0], 'k--')
    plt.ylabel(r"Background Rejection ($1 - \varepsilon_b$)")
    plt.xlabel(r"Signal Efficiency ($\varepsilon_s$)")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    # plt.yscale('log')
    plt.title(title)
    plt.tick_params(axis="both", which="minor", top=True, right=True, length=2, direction="in")
    plt.tick_params(axis="both", which="major", top=True, right=True, length=5, direction="in")
    legend = plt.legend(loc="best", frameon=True, framealpha=1, fontsize="11", fancybox=False)
    legend.get_frame().set_edgecolor('none')
    plt.minorticks_on()
    if file_path is not None:
        plt.savefig(file_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()


def plot_metric_per_epoch(metrics: Dict[str, np.ndarray], labels: Dict[str, str], colors: Dict[str, str], metric_name,
                          xlim, title,
                          file_path=None):
    """Display the metrics per epoch"""

    for metric in metrics:
        line_type = "dashed" if "val" in metric else "solid"
        plt.plot(metrics[metric], label=labels[metric], color=colors[metric], linestyle=line_type)

    legend = plt.legend(loc="best", frameon=True, framealpha=1, fontsize="11", fancybox=False, ncols=2)
    legend.get_frame().set_edgecolor('none')
    plt.xlim(xlim)
    plt.tick_params(axis="both", which="minor", top=True, right=True, length=2, direction="in")
    plt.tick_params(axis="both", which="major", top=True, right=True, length=5, direction="in")
    plt.minorticks_on()
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(title)
    if file_path is not None:
        plt.savefig(file_path, format="png", bbox_inches="tight", dpi=300)
    plt.show()


def plot_distributions(
        bin_egdes, values_hist, colors, labels, title, x_label, y_label, file_path=None):
    """Plot the histogram for different distributions."""
    values = np.array([
        np.mean([lower_edge, upper_edge]) for lower_edge, upper_edge in zip(bin_egdes[:-1], bin_egdes[1:])
    ])
    legend_list = []
    for dist_name, counts in values_hist.items():
        weights, _ = np.histogram(counts, bins=bin_egdes)
        print(weights)
        plt.hist(x=values, bins=bin_egdes, weights=weights, color=colors[dist_name], label=labels[dist_name],
                 histtype="step")
        legend_list.append(
            Line2D([0], [0], color=colors[dist_name], lw=2, label=labels[dist_name],
                   linestyle="solid")
        )
    plt.tick_params(axis="both", which="minor", top=True, right=True, length=2, direction="in")
    plt.tick_params(axis="both", which="major", top=True, right=True, length=5, direction="in")
    plt.minorticks_on()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(handles=legend_list, loc="best", frameon=False, framealpha=1, fontsize=15, fancybox=False, ncols=1)
    if file_path is not None:
        plt.savefig(file_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
