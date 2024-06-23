"""Evaluates the ROC curve and the AUC score for all the trainned models"""

from sklearn.metrics import roc_curve, recall_score, precision_score, confusion_matrix
import keras
import numpy as np
from ML.CNN.data_loader import load_data_qg_tagging as load_data_cnn
from src.Preprocessing.norm import norm
from ML.EFPs.data_loaders import load_data_qg_tagging as load_data_efps
from ML.ParticleCloud.data_loaders import load_data_qg_tagging as load_data_pc
import joblib


def find_threshold(y_true, scores, signal_eff_threshold):
    """
    Evaluates the model on the X_test data and returns the signal efficiency (recall) and
    the background rejection (FPR).
    """
    # calculating the roc curve
    background_eff, signal_eff, thresholds = roc_curve(y_true, scores)
    # finds the threshold value for the given signal efficiency
    threshold_index = np.where(signal_eff >= signal_eff_threshold)[0][0]
    return thresholds[threshold_index], background_eff[threshold_index]


if __name__ == "__main__":
    # signal efficiency
    signal_eff = 0.5

    # For Keras models
    model = keras.models.load_model("../ParticleCloud/PointNet/QuarkGluon_Tagging_PointNet.keras")
    # For sklearn models
    # model = joblib.load("../EFPs/DiscriminantAnalysis/LinearDiscriminant_qg_n3.joblib")

    # reading the Jet image
    # x, y_true = load_data_cnn(quark_data_path="../../Data/Test/q_jets.csv",
    #                           gluon_data_path="../../Data/Test/g_jets.csv",
    #                           eta_range=(-0.4, 0.4), phi_range=(-0.4, 0.4), n_eta_bins=16, n_phi_bins=16)
    # y_pred = model.predict(x)[:, 0]

    # reading the data for the particle cloud
    x, y_true = load_data_pc(quark_data_path="../../Data/Test/q_jets.csv", gluon_data_path="../../Data/Test/g_jets.csv")
    y_pred = model.predict(x)[:, 0]
    y_true = y_true[:, 0]

    # reading EFPs
    # best_features_set = [0, 1, 5, 6, 12, 17, 33, 42, 43, 48]
    # x, y_true = load_data_efps(
    #     quark_data_path="../../Data/Test/q_jets_efps_d5_primed.npy",
    #     gluon_data_path="../../Data/Test/g_jets_efps_d5_primed.npy"
    # )
    # y_pred = model.predict_proba(norm(3, x))[:, 1]


    threshold, background_eff = find_threshold(y_true, y_pred, signal_eff)

    print(f"threshold value: {threshold}")
    print(f"Performance metrics for a signal efficiency of {signal_eff}:")
    print(f"Precision: {precision_score(y_true, y_pred >= threshold):.3f}")
    print(f"Background Efficiency:{background_eff:.3f}")
    print(f"Background Rejection (1/epsilon_b): {1./background_eff:.3f}")
    print(f"Background Rejection (1 - epsilon_b): {1. - background_eff:.3f}")
    print("Confusion Matrix:")
    conf_mat = confusion_matrix(y_true, y_pred >= threshold, labels=[0, 1])
    print(conf_mat)
