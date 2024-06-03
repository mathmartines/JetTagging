import numpy as np
from typing import List, Tuple


class EFPDataGenerator:
    """Creates the data with the features and the respective labels"""

    def __init__(self, **classes):
        """
        :param classes: Dictionary where the names correspond to the type of Jet and the values are
                        the path to the corresponding data storing all the EFPs for the respective classes.
        """
        self.files_paths = classes
        self._data = {key: np.load(file_path) for key, file_path in self.files_paths.items()}

    def create_data(self, class_definitions: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates the dataset X joinning all the datasets and creates the respective labels for each of the
        possible jets.

        :param class_definitions:
            Definition of the classes. The names in the first tuple represents the class 0, the second
            represents the class 1, and so on.
            The names in each tuple must correspond to the types of Jets passed to the constructor of the
            class.

        :return:
            The dataset X and the respective labels y
        """
        # creating the X dataset by stacking the jets of each class
        # here we assume that if the class is made more than one jet type we just take 1/N fraction of jets
        # from each type, where N is the total number of jet types in the definition of the class
        datasets = tuple(
            self._data[jet_type][:int(self._data[jet_type].shape[0] / len(class_def))]
            for class_def in class_definitions for jet_type in class_def
        )
        X = np.vstack(datasets)

        # creating the labels for data
        y, cls_label, index = np.empty(len(X)), 0, 0
        for class_def in class_definitions:
            size_class = sum(int(self._data[jet_type].shape[0] / len(class_def)) for jet_type in class_def)
            y[index: index + size_class] = cls_label
            cls_label += 1
            index += size_class

        return X, y


if __name__ == "__main__":
    gen = EFPDataGenerator(
        top="../Data/t_jets_efp_d5.npy",
        gluon="../Data/g_jets_efp_d5.npy",
        quark="../Data/q_jets_efp_d5.npy"
    )
    X, y = gen.create_data(class_definitions=[('top',), ('gluon', 'quark')])
    top = np.load("../Data/t_jets_efp_d5.npy")
    gluon = np.load("../Data/g_jets_efp_d5.npy")
    print(y[0])
