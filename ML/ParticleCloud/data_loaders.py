"""Loads the data for the ParticleCloud NN."""

import numpy as np
import pandas as pd
from src.Preprocessing.LabelsGeneration import create_jet_labels_one_column_per_category
from typing import Tuple
from src.Preprocessing.JetPreprocessing import JetProcessingParticleCloud
from sklearn.utils import shuffle


def load_data_qg_tagging(quark_data_path: str, gluon_data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads the data for the QG-tagging NN"""
    # loading the data files for the quark and gluon jets
    data_quark = pd.read_csv(quark_data_path, header=None)
    data_gluon = pd.read_csv(gluon_data_path, header=None)
    all_jets = pd.concat([data_quark, data_gluon], axis=0)
    all_jets.reset_index(drop=True, inplace=True)  # reset the indices

    # preprocessing class
    jet_preprocessing = JetProcessingParticleCloud()
    X = jet_preprocessing.transform(all_jets.to_numpy())
    # labels
    y = create_jet_labels_one_column_per_category(
        [(0, data_quark.shape[0] - 1), (data_quark.shape[0], all_jets.shape[0] - 1)]
    )

    # shuffling the data
    return shuffle(X, y, random_state=0)


def load_data_top_tagging(top_quark_path: str, quark_data_path: str,
                          gluon_data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads the data for the Top-tagging NN"""
    # loading the data files for the top, quark, and gluon jets
    data_top = pd.read_csv(top_quark_path, header=None).to_numpy()
    data_quark = pd.read_csv(quark_data_path, header=None).to_numpy()
    data_gluon = pd.read_csv(gluon_data_path, header=None).to_numpy()
    # selecting only half of quarks and gluons
    all_jets = np.vstack([data_top, data_quark[:int(len(data_quark)/2)], data_gluon[:int(len(data_gluon)/2)]])

    # preprocessing class
    jet_preprocessing = JetProcessingParticleCloud()
    X = jet_preprocessing.transform(all_jets)
    # labels
    y = create_jet_labels_one_column_per_category(
        [(0, data_top.shape[0] - 1), (data_top.shape[0], all_jets.shape[0] - 1)]
    )

    # shuffling the data
    return shuffle(X, y, random_state=0)
