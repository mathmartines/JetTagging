"""Applies the BackwardElimination algorithm to find the best features for the Quark-Gluon Tagging tree"""

from sklearn.tree import DecisionTreeClassifier
from src.BackwardElimination import BackwardElimination
from ML.EFPs.data_loaders import load_data_qg_tagging

if __name__ == "__main__":
    # loading the data for the Decision Tree
    root_data_folder = "../../../Data"
    X_train, y_train = load_data_qg_tagging(
        quark_data_path=f"{root_data_folder}/Trainning/q_jets_efps_d5_primed.npy",
        gluon_data_path=f"{root_data_folder}/Trainning/g_jets_efps_d5_primed.npy"
    )
    X_val, y_val = load_data_qg_tagging(
        quark_data_path=f"{root_data_folder}/Validation/q_jets_efps_d5_primed.npy",
        gluon_data_path=f"{root_data_folder}/Validation/g_jets_efps_d5_primed.npy"
    )

    # declaring the tree
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=9, min_samples_split=250)

    # declaring the algorithm and running
    backward_elimination = BackwardElimination(
        tree=tree, x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val, file_name="QG_Tagging_BestFeatures_Set"
    )
    backward_elimination.run(accuracy_fraction_limit=0.99)
