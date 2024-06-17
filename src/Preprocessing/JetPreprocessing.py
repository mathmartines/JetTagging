from typing import Tuple
from sklearn.base import TransformerMixin, BaseEstimator
from src.JetImage import JetImage, IntensityPtCalculator, JetImageCalculatorPandas
from abc import ABC, abstractmethod
import energyflow as ef
import numpy as np


class JetProcessing(ABC, BaseEstimator, TransformerMixin):
    """
    Abstract class to process the jet input data for the ML algorithms.
    Here we use the Template Design Pattern.

    Each subclass should implement the method prepare_data which specifies how the data is being
    prepared for the ML algorithm.
    """

    def __init__(self):
        # stores the data being prepared
        self._X = None

    def fit_transform(self, X, y=None, **fit_params):
        # does nothing
        return self

    def transform(self, X):
        """
        Creates the data for the ML image algorithms.
        It uses the prepare_data method declared by the subclases to prepare the data for the ML image algorithms.

        :param X: list of jet per row, just as it's in the initial database
        :return: the data processed for the ML algorithms
        """
        # delegating the job to the subclasses
        self.prepare_data(X)
        # copying to ensure that any modification on the class attribute X is not propagated to
        # other places
        return self._X.copy()

    @abstractmethod
    def prepare_data(self, X):
        """How to preprocess the data."""
        raise NotImplementedError("Please use one of the subclasses")


class PreprocessingJetImages(JetProcessing):
    """
    Class reponsible for processing the data for the ML image algorithms.

    It delegates the work to the JetImages class.
    """

    def __init__(self, phi_range: Tuple[float, float], eta_range: Tuple[float, float], n_bins_phi: int,
                 n_bins_eta: int, jet_image_strategy: IntensityPtCalculator = JetImageCalculatorPandas()):
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
        super().__init__()

    def prepare_data(self, X):
        """
        Creates the data for the ML image algorithms.
        It sets up jet image using the JetImage class and then transforms each entry into a one dimensional numpy array.

        :param X: numpy array taken from the pandas dataframe
        :return: the data processed for the ML image algorithms
        """
        self._X = np.array([self._jet_image.create_jet_image(jet_features).reshape(-1) for jet_features in X])


class PreprocessingEFPs(JetProcessing):
    """
    Process the data for the ML algorithms that uses the Energy Flow Polynomials as
    input features.
    To process the data the degree d of the polynomial must be given.
    """

    def __init__(self, d: int, *args):
        """
        It will transform the jet substructure to a set of polynomial with degree up to d.

        :param d: degree of the polynomial.
        """
        super().__init__()
        self._efps_set = ef.EFPSet(("d<=", d), *args, measure='hadr', beta=1, normed=False, verbose=True)


    @property
    def efps_set(self):
        return self._efps_set

    def prepare_data(self, X):
        """
        Tranforms the input jets into a set of polynomials up to degree d.

        :param X: list of jet per row
        :return: the X data for the ML algorithms that uses the Energy Flow Polynomials as input features
        """
        self._X = np.array([self._efps_set.compute(self._get_jet_constituents(jet)) for jet in X])

    @staticmethod
    def _get_jet_constituents(jet) -> np.ndarray:
        """
        Takes the list containing all the jets constituents and breaks it them to a list
        of the jet constituents 3-momentum in the order [pt, eta, phi]

        :param jet: array with all the jet constituents
        :return: a list of all the jet constituents momentum, where the momentum are given in the order
                [pt, eta, phi].
        """
        return np.array([
            # only non Zero Padded jets
            [jet[const_index + 2], jet[const_index], jet[const_index + 1]]
            for const_index in range(0, len(jet), 4) if jet[const_index + 3] == 1
        ])


class JetProcessingParticleCloud(JetProcessing):

    def prepare_data(self, X):
        # each entry represent a jet
        self._X = np.array([self.get_jet_constituents(jet) for jet in X])

    @staticmethod
    def get_jet_constituents(jet):
        jets_constituents = np.array([
             jet[jet_index: jet_index + 4] for jet_index in range(0, len(jet), 4)
        ])
        # the only normalization will take is the log of the pT
        # since we have zero-padded particles, we must only take the look of the one where the mask == 1
        real_particles = jets_constituents[:, 3] == 1
        jets_constituents[real_particles, 2] = np.log(jets_constituents[real_particles, 2])

        return jets_constituents
