"""File that contains the routines to split the datasets for the Top-tagging and Quark-Gluon tagging using the EFPs."""

from ML.EFPs.data_loaders import load_data_top_tagging, load_data_qg_tagging


def split(tagging):
    """Splits the data according to the tagging."""
    
    if tagging == "top":
        X_train, y_train = load_data_top_tagging(
            top_data_path="../../../Data/DataJetTagging/Trainning/t_jets_efps_d5_primed.npy",
            quark_data_path="../../../Data/DataJetTagging/Trainning/q_jets_efps_d5_primed.npy",
            gluon_data_path="../../../Data/DataJetTagging/Trainning/g_jets_efps_d5_primed.npy"
        )
        X_val, y_val = load_data_top_tagging(
            top_data_path="../../../Data/DataJetTagging/Validation/t_jets_efps_d5_primed.npy",
            quark_data_path="../../../Data/DataJetTagging/Validation/q_jets_efps_d5_primed.npy",
            gluon_data_path="../../../Data/DataJetTagging/Validation/g_jets_efps_d5_primed.npy"
        )
        X_test, y_test = load_data_top_tagging(
            top_data_path="../../../Data/DataJetTagging/Test/t_jets_efps_d5_primed.npy",
            quark_data_path="../../../Data/DataJetTagging/Test/q_jets_efps_d5_primed.npy",
            gluon_data_path="../../../Data/DataJetTagging/Test/g_jets_efps_d5_primed.npy"
        )
    else:
        X_train, y_train = load_data_qg_tagging(
            quark_data_path="../../../Data/DataJetTagging/Trainning/q_jets_efps_d5_primed.npy",
            gluon_data_path="../../../Data/DataJetTagging/Trainning/g_jets_efps_d5_primed.npy"
        )
        X_val, y_val = load_data_qg_tagging(
            quark_data_path="../../../Data/DataJetTagging/Validation/q_jets_efps_d5_primed.npy",
            gluon_data_path="../../../Data/DataJetTagging/Validation/g_jets_efps_d5_primed.npy"
        )
        X_test, y_test = load_data_qg_tagging(
            quark_data_path="../../../Data/DataJetTagging/Test/q_jets_efps_d5_primed.npy",
            gluon_data_path="../../../Data/DataJetTagging/Test/g_jets_efps_d5_primed.npy"
        )

    return X_train, X_val, X_test, y_train, y_val, y_test
