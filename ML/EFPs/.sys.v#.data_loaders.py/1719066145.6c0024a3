"""File that contains the routines to load the datasets for the Top-tagging and Quark-Gluon tagging using the EFPs."""

import numpy as np
from src.Preprocessing.LabelsGeneration import create_labels_single_column
from sklearn.utils import shuffle
from typing import Tuple


def load_data_qg_tagging(quark_data_path: str, gluon_data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads the data for the Quark-Gluon tagging."""
    # loading the datasets
    quark_efps = np.load(quark_data_path)
    gluon_efps = np.load(gluon_data_path)

    all_efps = np.vstack((quark_efps, gluon_efps))
    jet_labels = create_labels_single_column([(0, len(quark_efps) - 1), (len(quark_efps), len(all_efps) - 1)])

    # shuffling the data
    X, y = shuffle(all_efps, jet_labels, random_state=42)

    return X, y


def load_data_top_tagging(top_data_path: str, quark_data_path: str,
                          gluon_data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads the data for the Top-tagging"""
    # loading the EFPs for each jet
    top_efps = np.load(top_data_path)
    quark_efps = np.load(quark_data_path)
    gluon_efps = np.load(gluon_data_path)
    non_top_efps = shuffle(np.vstack((quark_efps, gluon_efps)), random_state=42)

    # we only take half of the quark and gluons dataset
    half_size_non_top_efps = int(non_top_efps.shape[0] / 2) + 1

    all_efps = np.vstack((top_efps, non_top_efps[:half_size_non_top_efps]))
    jet_labels = create_labels_single_column([(0, len(top_efps) - 1), (len(top_efps), len(all_efps) - 1)])

    X, y = shuffle(all_efps, jet_labels)

    return X, y
