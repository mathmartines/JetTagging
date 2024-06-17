from typing import List, Tuple, Dict
from src.Particle import ParticleType
import numpy as np


def create_jet_labels_one_column_per_category(jet_type_indices: List[Tuple[int, int]]):
    """
    Creates the Jet labels creating one column per category.

    The jet_type_indices is a list where item stores a tuple with the ranges of each jet type in the data X.
    For example, if X has first Top jets, and then Gluon jets, the dictionary would look like:

        [
            (index of the first top jet, index of the last top jet),
            (index of the first gluon jet, index of the last gluon jet)
        ]

    This class assumes that the data X is ordered by the type of the jet. So, X is like the
    following

        X = [
            Top jet, Top jet, Top jet, ..., Last Top jet, Gluon jet, ..., Last Gluon jet
        ]

    The first column of the labels represent the first jet type, the second column represents the
    second jet type, and so on.

    :param jet_type_indices: dictionary with the information needed to create the jet labels.
    :return: the labels in the same order as X.
    """
    # counting the total number of jets
    number_of_jets = sum([final_index - initial_index + 1 for initial_index, final_index in jet_type_indices])
    # creating the array of the same lenght as the data and with the number of columns
    # equal to the number of jet types
    jet_labels = np.zeros(shape=(number_of_jets, len(jet_type_indices)))
    # filling the columns represeting each category to 1 for the jets that belong to the same category
    for jet_index, indices in enumerate(jet_type_indices):
        jet_labels[indices[0]: indices[1] + 1, jet_index] = 1
    return jet_labels


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



