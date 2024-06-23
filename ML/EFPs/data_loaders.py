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


def load_data_qg_tagging_wR(quark_data_path: str, gluon_data_path: str,
                            mean_deltaRij_quark_path: str, mean_deltaRij_gluon_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads the data for Quark-Gluon tagging with mean DeltaRij."""
    # Load the datasets
    quark_efps = np.load(quark_data_path)
    gluon_efps = np.load(gluon_data_path)

    # Combine quark and gluon EFPs
    all_efps = np.vstack((quark_efps, gluon_efps))

    # Load the mean DeltaRij values
    mean_deltaRij_quark = np.load(mean_deltaRij_quark_path)
    mean_deltaRij_gluon = np.load(mean_deltaRij_gluon_path)
    all_mean_deltaRij = np.hstack((mean_deltaRij_quark, mean_deltaRij_gluon))

    # Concatenate the mean DeltaRij to the EFP features
    all_efps = np.hstack((all_efps, all_mean_deltaRij.reshape(-1, 1)))

    jet_labels = create_labels_single_column([(0, len(quark_efps) - 1), (len(quark_efps), len(all_efps) - 1)])

    # Shuffle the data and labels
    X, y = shuffle(all_efps, jet_labels, random_state=42)

    return X, y


def load_data_top_tagging_wR(quark_data_path: str, gluon_data_path: str, top_data_path: str, 
                             mean_deltaRij_quark_path: str, mean_deltaRij_gluon_path: str, 
                             mean_deltaRij_top_path: str) -> Tuple[np.ndarray, np.ndarray]:
    
    """Loads the data for the Top-tagging"""
    # Loading the jet EFPs for each category
    top_efps = np.load(top_data_path)
    quark_efps = np.load(quark_data_path)
    gluon_efps = np.load(gluon_data_path)
    
    # Loading the mean DeltaRij values
    mean_deltaRij_top = np.load(mean_deltaRij_top_path)
    mean_deltaRij_quark = np.load(mean_deltaRij_quark_path)
    mean_deltaRij_gluon = np.load(mean_deltaRij_gluon_path)
    
    # Ensure mean_deltaRij has the same number of entries as the jets
    if len(top_efps) != len(mean_deltaRij_top) or len(quark_efps) != len(mean_deltaRij_quark) or len(gluon_efps) != len(mean_deltaRij_gluon):
        raise ValueError("Mean DeltaRij arrays must have the same length as their respective EFP arrays.")

    # Take half of the quark and gluon datasets
    half_size_quark_efps = int(quark_efps.shape[0] / 2) + 1
    half_size_gluon_efps = int(gluon_efps.shape[0] / 2) + 1
    
    # Select half of the datasets
    non_top_quark_efps = quark_efps[:half_size_quark_efps]
    non_top_gluon_efps = gluon_efps[:half_size_gluon_efps]
    
    mean_deltaRij_non_top_quark = mean_deltaRij_quark[:half_size_quark_efps]
    mean_deltaRij_non_top_gluon = mean_deltaRij_gluon[:half_size_gluon_efps]
    
    # Combine non-top quarks and gluons
    non_top_efps = np.vstack((non_top_quark_efps, non_top_gluon_efps))
    mean_deltaRij_non_top = np.hstack((mean_deltaRij_non_top_quark, mean_deltaRij_non_top_gluon))
    
    # Combine top jets with non-top jets
    all_efps = np.vstack((top_efps, non_top_efps))
    all_mean_deltaRij = np.concatenate((mean_deltaRij_top, mean_deltaRij_non_top))

    # Concatenating the mean DeltaRij values to the EFP features
    all_efps = np.hstack((all_efps, all_mean_deltaRij.reshape(-1, 1)))

    jet_labels = create_labels_single_column([(0, len(top_efps) - 1), (len(top_efps), len(all_efps) - 1)])

    X, y = shuffle(all_efps, jet_labels)
    
    return X, y
