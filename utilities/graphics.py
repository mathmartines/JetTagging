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
        signal_eff: Dict[str, np.ndarray], background_eff: Dict[str, np.ndarray], labels: Dict[str, str],
        colors: Dict[str, str], file_path=None
):
    """Plots the roc curve for a list of classifiers"""

    for classifier in signal_eff:
        plt.plot(signal_eff[classifier], background_eff[classifier], label=labels[classifier],
                 color=colors[classifier])
    plt.plot([0, 1], [1, 0], 'k--')
    plt.ylabel(r"Background Rejection ($1 - \varepsilon_b$)")
    plt.xlabel(r"Signal Efficiency ($\varepsilon_s$)")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.tick_params(axis="both", which="minor", top=True, right=True, length=2, direction="in")
    plt.tick_params(axis="both", which="major", top=True, right=True, length=5, direction="in")
    plt.legend(loc="best", frameon=False, framealpha=1, fontsize="12", fancybox=False)
    plt.minorticks_on()
    if file_path is not None:
        plt.savefig(file_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()


def plot_metric_per_epoch(metrics: Dict[str, np.ndarray], labels: Dict[str, str], colors: Dict[str, str], metric_name,
                          file_path=None):
    """Display the metrics per epoch"""

    for metric in metrics:
        line_type = "dashed" if "val" in metric else "solid"
        plt.plot(metrics[metric], label=labels[metric], color=colors[metric], linestyle=line_type)

    # plt.legend(loc="best", frameon=False, framealpha=1, fontsize="11", fancybox=False, ncols=2)
    plt.xlim((0, 100))
    plt.tick_params(axis="both", which="minor", top=True, right=True, length=2, direction="in")
    plt.tick_params(axis="both", which="major", top=True, right=True, length=5, direction="in")
    plt.minorticks_on()
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    if file_path is not None:
        plt.savefig(file_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()


def invariant_mass_hist(events: Dict[str, np.ndarray], colors: Dict[str, str], labels: Dict[str, str],
                        linestyle, title, nbins=20, file_path=None):
    """Plot invariant mass histograms for each of the events"""
    legend_list = []
    for hist_name in events:
        hist_bins, bin_edges = np.histogram(events[hist_name], bins=nbins, density=False)
        plt.hist(bin_edges[:-1], bins=bin_edges, weights=hist_bins / sum(hist_bins), color=colors[hist_name],
                 histtype="step", linestyle=linestyle[hist_name])
        legend_list.append(
            Line2D([0], [0], color=colors[hist_name], lw=2, label=labels[hist_name],
                   linestyle=linestyle[hist_name])
        )
    plt.legend(handles=legend_list, loc="best", frameon=False, framealpha=1, fontsize="10", fancybox=False, ncols=1)
    plt.tick_params(axis="both", which="minor", top=True, right=True, length=2, direction="in")
    plt.tick_params(axis="both", which="major", top=True, right=True, length=5, direction="in")
    plt.minorticks_on()
    plt.title(title)
    plt.xlim(left=0, right=3)
    plt.ylabel("Fraction of events")
    plt.xlabel(r"$m_{WWbb}$ after scaling")
    if file_path is not None:
        plt.savefig(file_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()