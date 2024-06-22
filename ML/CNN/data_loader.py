"""Functions to load the data for trainning the data for CNN with Images"""


from src.Preprocessing.JetPreprocessing import PreprocessingJetImages
from src.Preprocessing.LabelsGeneration import create_jet_labels_one_column_per_category
from sklearn.utils import shuffle
import numpy as np
import pandas as pd


def load_data_qg_tagging(quark_data_path, gluon_data_path, phi_range, eta_range, n_phi_bins, n_eta_bins):
    """
    Loads the data and preprocesses the images for the QG tagging.

    :param quark_data_path: path to the quarks dataset
    :param gluon_data_path: path to the gluon dataset
    :param phi_range: limits on the Phi range
    :param eta_range: limits on the Eta range
    :param n_phi_bins: number of bins for Phi
    :param n_eta_bins: number of bins for Eta
    :return: images and labels for the JetTagging NN.
    """
    # reading the data files
    data_quark = pd.read_csv(quark_data_path, header=None)
    data_gluon = pd.read_csv(gluon_data_path, header=None)

    # Setting up the images using the Preprocessing class
    jet_image_preprocessing = PreprocessingJetImages(phi_range=phi_range, eta_range=eta_range, n_bins_phi=n_phi_bins,
                                                     n_bins_eta=n_eta_bins)

    # getting the jet images
    quark_images = jet_image_preprocessing.transform(X=data_quark.to_numpy())
    gluon_images = jet_image_preprocessing.transform(X=data_gluon.to_numpy())

    # creating the data with all the images
    all_jet_images = np.vstack((quark_images, gluon_images))
    # creating the labels, the first tuple telss the initial and final index of the first jet type, and so on.
    jet_labels = create_jet_labels_one_column_per_category(
        [(0, len(quark_images) - 1), (len(quark_images), len(all_jet_images) - 1)]
    )

    # shuffling the data
    return shuffle(all_jet_images, jet_labels, random_state=0)
