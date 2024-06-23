"""Evaluates the ROC curve and the AUC score for all the trainned models"""

from sklearn.metrics import roc_curve, recall_score, precision_score, confusion_matrix
import keras
import numpy as np
from ML.CNN.data_loader import load_data_top_tagging as load_data_cnn
from src.Preprocessing.norm import norm
from ML.EFPs.data_loaders import load_data_qg_tagging as load_data_efps
from ML.ParticleCloud.data_loaders import load_data_top_tagging as load_data_pc
from ML.EFPs.data_loaders import load_data_qg_tagging_wR, load_data_top_tagging_wR
import joblib
from sklearn.preprocessing import StandardScaler


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

    # reading the Jet image
    # x, y_true = load_data_cnn(top_data_path="../../Data/Test/t_jets.csv", quark_data_path="../../Data/Test/q_jets.csv",
    #                           gluon_data_path="../../Data/Test/g_jets.csv",
    #                           eta_range=(-0.4, 0.4), phi_range=(-0.4, 0.4), n_eta_bins=16, n_phi_bins=16)

    # reading the data for the particle cloud
    # x, y_true = load_data_pc(top_quark_path="../../Data/Test/t_jets.csv",
    #                          quark_data_path="../../Data/Test/q_jets.csv", gluon_data_path="../../Data/Test/g_jets.csv")

    # reading EFPs
    best_features_set = [0, 1, 3, 9, 11, 12, 15, 16, 18, 22, 28, 29, 34, 38, 39, 44, 45, 48]
    x, y_true = load_data_efps(
        quark_data_path="../../Data/Test/q_jets_efps_d5_primed.npy",
        gluon_data_path="../../Data/Test/g_jets_efps_d5_primed.npy"
    )


    # CNN EFPs
    # cnn_efps = keras.saving.load_model("../EFPs/NN/NN_EFPs_Top_Tagging.keras")
    # X_train, y_true = load_data_top_tagging_wR(
    #     top_data_path=f"../../Data/Test/t_jets_efps_d5_primed.npy",
    #     quark_data_path=f"../../Data/Test/q_jets_efps_d5_primed.npy",
    #     gluon_data_path=f"../../Data/Test/g_jets_efps_d5_primed.npy",
    #     mean_deltaRij_top_path=f"../../Data/Test/mean_deltaRij_t.npy",
    #     mean_deltaRij_quark_path=f"../../Data/Test/mean_deltaRij_q.npy",
    #     mean_deltaRij_gluon_path=f"../../Data/Test/mean_deltaRij_g.npy",
    # )
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # y_pred = cnn_efps.predict(X_train)

    # Particle Cloud
    # X_pc, y_true = load_data_pc(
    #     quark_data_path="../../Data/Test/q_jets.csv", gluon_data_path="../../Data/Test/g_jets.csv"
    # )
    # Particle Cloud
    # particle_cloud = keras.models.load_model(
    #     "../ParticleCloud/PointNet/QuarkGluon_Tagging_PointNet.keras")
    # y_pred = particle_cloud.predict(X_pc)

    # For Keras models
    # model = keras.models.load_model("../ParticleCloud/PointNet/Top_Tagging_PointNet.keras")

    # For sklearn models
    model = joblib.load("../EFPs/DecisionTree/QuarkGluon_Tagging_DT.joblib")

    y_pred = model.predict_proba(x)[:, 1]
    # y_pred = model.predict(x)[:, 1]

    threshold, background_eff = find_threshold(y_true, y_pred, signal_eff)
    print(f"Background Rejection (1 - epsilon_b): {1. - background_eff:.4f}")

