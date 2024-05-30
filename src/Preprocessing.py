from typing import Tuple, Dict
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import shuffle
from src.JetImage import JetImage, IntensityPtCalculator, JetImageCalculatorPandas
from src.Particle import ParticleType
import numpy as np


def create_jet_labels(jet_inputs: Dict[ParticleType, Tuple[int, int]]) -> np.ndarray:
    """
    Creates the Jet labels.
    The jet_inputs is a dictionary where the key stores the type of jet (Gluon, Top, or LightQuarks), while
    the value stores a tuple with the ranges of each jet in the data X.
    For example, if X has first Top jets, and then Gluon jets, the dictionary would look like:

    {
        ParticleType.Top: (index of the first top jet, index of the last top jet),
        ParticleType.Gluon: (index of the first gluon jet, index of the last gluon jet)
    }

    This class assumes that the data X is ordered by the type of the jet. So, X is like the
    following

    X = [
        Top jet, Top jet, Top jet, ..., Last Top jet, Gluon jet, ..., Last Gluon jet
    ]

    The first column of the labels represent the first jet type, the second column represents the
    second jet type, and so on.

    :param jet_inputs: dictionary with the information needed to create the jet labels.
    :return: the labels in the same order as X.
    """
    # counting the total number of jets
    number_of_jets = sum([final_index - initial_index + 1 for initial_index, final_index in jet_inputs.values()])
    # creating the array of the same lenght as the data and with the number of columns
    # equal to the number of jet types
    jet_labels = np.zeros(shape=(number_of_jets, len(jet_inputs.keys())))
    # filling the jet labels
    for jet_index, indices in zip(range(len(jet_inputs)), jet_inputs.values()):
        jet_labels[indices[0]: indices[1] + 1, jet_index] = 1

    return jet_labels


class PreprocessingJetImages(BaseEstimator, TransformerMixin):
    """
    Class reponsible for processing the data to set up the data
    for the ML image algorithms.

    It delegates the work to the JetImages class.
    """

    def __init__(self,
                 phi_range: Tuple[float, float],
                 eta_range: Tuple[float, float],
                 n_bins_phi: int,
                 n_bins_eta: int,
                 jet_image_strategy: IntensityPtCalculator = JetImageCalculatorPandas()
                 ):
        """
        All the parameters that are required to initialize the image.

        :param phi_range: (phi min, phi max) - minimum and maximum values of the phi range.
        :param eta_range: (eta min, eta max) - minimum and maximum values of the eta range.
        :param n_bins_phi: number of bins for phi axis
        :param n_bins_eta: number of bins for the eta axis
        :param jet_image_strategy: strategy on how to calculate the intesity of each bin.
                                   The default strategy is the one using pandas.Series
        """
        # delegating the job of creating the Images to the JetImages class
        self._jet_image = JetImage(
            phi_range=phi_range,
            eta_range=eta_range,
            n_bins_phi=n_bins_phi,
            n_bins_eta=n_bins_eta,
            pt_intensity_calculator=jet_image_strategy
        )
        self._labels = None

    def fit(self, X, y=None):
        # default (does nothing)
        return self

    def transform(self, X, y: Dict[ParticleType, Tuple[int, int]]):
        """
        Creates the data for the ML image algorithms.
        It sets up jet image using the JetImage class and the transform
        each entry into a one dimensional numpy array.
        It also sets up the labels for each entry of our data.

        :param X: numpy array taken from the pandas dataframe
        :param y: order of the data as specified by create_jet_labels function
        :return: the data processed for the ML image algorithms
        """
        jet_images = np.array([self._jet_image.create_jet_image(jet_features).reshape(-1) for jet_features in X])
        self._labels = create_jet_labels(y)
        # shuffling the data
        jet_images, self._labels = shuffle(jet_images, self._labels)

        return jet_images

    @property
    def jet_labels(self):
        return self._labels

