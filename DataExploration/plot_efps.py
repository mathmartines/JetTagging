from utilities.graphics import plot_distributions
import numpy as np


if __name__ == '__main__':
    # loading the data
    data_quark = np.load("../Data/Trainning/q_jets_efps_d5_primed.npy")
    data_gluon = np.load("../Data/Trainning/g_jets_efps_d5_primed.npy")
    data_top = np.load("../Data/Trainning/t_jets_efps_d5_primed.npy")

    # selecting the EFP to plot
    index_pol = 2
    efp_quark = data_quark[:, index_pol]
    efp_gluon = data_gluon[:, index_pol]
    efp_top = data_top[:, index_pol]

    bin_edges = np.arange(0., 0.08, 0.005)

    distributions = {"top": efp_top, "quark": efp_quark, "gluon": efp_gluon}
    labels = {
        "top": r"Top jets",
        "quark": r"Quark jets",
        "gluon": r"Gluon jets",
    }
    colors = {
        "top": "green",
        "quark": "blue",
        "gluon": "red"
    }

    plot_distributions(
        bin_egdes=bin_edges,
        values_hist=distributions,
        labels=labels,
        colors=colors,
        x_label=r"EFP$_2$",
        y_label=r"Number of jets",
        title="",
        file_path="../Plots/Dist/efp_2.pdf"
    )
