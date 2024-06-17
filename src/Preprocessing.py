from typing import Tuple, Dict, Callable
from sklearn.base import TransformerMixin, BaseEstimator
from src.JetImage import JetImage, IntensityPtCalculator, JetImageCalculatorPandas
from src.Particle import ParticleType
from abc import ABC, abstractmethod
import energyflow as ef
import numpy as np


def create_labels_single_column(jet_inputs: Dict[ParticleType, Tuple[int, int]]):
    """
    Creates the Jet labels using a single column and diferent integer values for different categories.

    The jet_inputs is a dictionary where the key stores the type of jet (Gluon, Top, or LightQuarks), while
    the value stores a tuple with the ranges of each jet in the data X. For example, if X has first Top jets,
    and then Gluon jets, the dictionary would look like:

        {
            ParticleType.Top: (index of the first top jet, index of the last top jet),
            ParticleType.Gluon: (index of the first gluon jet, index of the last gluon jet)
        }

    This class assumes that the data X is ordered by the type of the jet. So, X must look like the following

        X = [
            Top jet, Top jet, Top jet, ..., Last Top jet, Gluon jet, ..., Last Gluon jet
        ]

    The first jet in the dictionary is assigned to the category 0, the second to category 1, and so on.

    :param jet_inputs: dictionary with the information needed to create the jet labels.
    :return: the labels in the same order as X.
    """
    # counting the total number of jets
    number_of_jets = sum([final_index - initial_index + 1 for initial_index, final_index in jet_inputs.values()])
    # creating the array of the same lenght as the data and with the number of columns
    # equal to the number of jet types
    jet_labels = np.zeros(number_of_jets)
    limit_categories = list(jet_inputs.values())
    # filling the jet labels (starting from the category 1 since the category 0 is already done)
    for jet_index, indices in zip(range(1, len(jet_inputs)), limit_categories[1:]):
        jet_labels[indices[0]: indices[1] + 1] = jet_index
    return jet_labels


def create_jet_labels_one_column_per_category(jet_inputs):
    """
    Creates the Jet labels creating one column per category.

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
    # filling the columns represeting each category to 1 for the jets that belong to the
    # same category
    for jet_index, indices in zip(range(len(jet_inputs)), jet_inputs.values()):
        jet_labels[indices[0]: indices[1] + 1, jet_index] = 1
    return jet_labels


class JetProcessing(ABC, BaseEstimator, TransformerMixin):
    """Abstract class to process the jet input data for the ML algorithms."""

    def __init__(self):
        self._X = None

    def transform(self, X):
        """
        Creates the data for the ML image algorithms.
        It uses the prepare_X methods declared by the subclases to prepare the data for the ML image algorithms.

        :param X: list of jet per row, just as it's in the initial database
        :return: the data processed for the ML image algorithms
        """
        self.prepare_data(X)
        # copying to ensure that any modification on the class attribute X is not propagated to
        # other places
        return self._X.copy()

    @abstractmethod
    def prepare_data(self, X):
        """How to preprocess the data for the algorithm"""
        raise NotImplementedError("Please use one of the subclasses")


class PreprocessingJetImages(JetProcessing):
    """
    Class reponsible for processing the data to set up the data
    for the ML image algorithms.

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
        It sets up jet image using the JetImage class and the transform
        each entry into a one dimensional numpy array.
        It also sets up the labels for each entry of our data.

        :param X: numpy array taken from the pandas dataframe
        :param y: order of the data as specified by create_jet_labels function
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
        :param y: order of the data as specified by create_jet_labels function
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
