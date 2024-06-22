"""Evaluates the ROC curve and the AUC score for all the trainned models"""

from sklearn.metrics import roc_curve, auc
from utilities.graphics import plot_roc_curve
import keras
from ML.CNN.data_loader import load_data_qg_tagging as load_data_cnn
from ML.ParticleCloud.data_loaders import load_data_qg_tagging as load_data_pc
from ML.EFPs.data_loaders import load_data_qg_tagging as load_data_efps
import joblib


def evaluate_model(y_true, y_score):
    """
    Evaluates the model on the X_test data and returns the signal efficiency (recall) and
    the background rejection (FPR).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return tpr, fpr


def load_joblib_object(joblib_file):
    with open(joblib_file, 'rb') as fo:
        model = joblib.load(fo)
    return model


if __name__ == "__main__":
    # CNN
    cnn = keras.models.load_model("../CNN/QuarkGluonTagger/Quark_Gluon_Tagging_CNN.keras")
    X_test_cnn, y_test_cnn = load_data_cnn(
        quark_data_path="../../Data/Trainning/q_jets.csv",
        gluon_data_path="../../Data/Trainning/g_jets.csv",
        eta_range=(-0.4, 0.4),
        phi_range=(-0.4, 0.4),
        n_eta_bins=16,
        n_phi_bins=16
    )
    predict_cnn = cnn.predict(X_test_cnn)
    signal_eff_cnn, back_miss_id_rate_cnn = evaluate_model(y_test_cnn[:, 0], predict_cnn[:, 0])

    # PointNet
    point_net = keras.models.load_model("../ParticleCloud/PointNet/QuarkGluon_Tagging_PointNet.keras")
    X_pc, y_pc = load_data_pc(
        quark_data_path="../../Data/Test/q_jets.csv", gluon_data_path="../../Data/Test/g_jets.csv"
    )
    predict_pt = point_net.predict(X_pc)
    signal_eff_pn, back_miss_id_rate_pn = evaluate_model(y_pc[:, 0], predict_pt[:, 0])

    # Particle Cloud
    particle_cloud = keras.models.load_model("../ParticleCloud/ParticleNet/QuarkGluon_Tagging_ParticleCloud.keras")
    predict_pc = particle_cloud.predict(X_pc)
    signal_eff_pc, back_miss_id_rate_pc = evaluate_model(y_pc[:, 0], predict_pc[:, 0])

    # Decision Trees with EFPs
    decision_tree = load_joblib_object("../EFPs/DecisionTree/QuarkGluon_Tagging_DT.joblib")
    X_efps, y_efps = load_data_efps(
        quark_data_path="../../Data/Test/q_jets_efps_d5_primed.npy",
        gluon_data_path="../../Data/Test/g_jets_efps_d5_primed.npy"
    )
    predict_dt = decision_tree.predict_proba(X_efps)
    signal_eff_dt, back_miss_id_rate_dt = evaluate_model(y_efps, predict_dt[:, 1])

    signal_efficiencies = {
        "imagem": signal_eff_cnn,
        "pointnet": signal_eff_pn,
        "particlecloud": signal_eff_pc,
        "decision_tree": signal_eff_dt,
        # "Combined": signal_eff_combined
    }
    background_rej = {
        "imagem": 1 - back_miss_id_rate_cnn,
        "pointnet": 1 - back_miss_id_rate_pn,
        "particlecloud": 1 - back_miss_id_rate_pc,
        "decision_tree": 1 - back_miss_id_rate_dt
        # "Combined": 1 - fpr_combined
    }
    labels = {
        "imagem": f"CNN JetImages (AUC = {auc(back_miss_id_rate_cnn, signal_eff_cnn):.3f})",
        "pointnet": f"Point-Net (AUC = {auc(back_miss_id_rate_pn, signal_eff_pn):.3f})",
        "particlecloud": f"Particle Cloud (AUC = {auc(back_miss_id_rate_pc, signal_eff_pc):.3f})",
        "decision_tree": f"Decision-Tree (AUC = {auc(back_miss_id_rate_dt, signal_eff_dt):.3f})"
        # "PointNet": f"Point-Net (AUC = {auc(fpr_pn, signal_eff_pn):.2f})",
        # "Combined": f"Point-Net + Particle Cloud (AUC = {auc(fpr_combined, signal_eff_combined):.2f})",
    }
    colors = {
        "imagem": "#01204E",
        "pointnet": "#E9C874",
        "particlecloud": "#A34343",
        "decision_tree": "darkgray"
    }

    plot_roc_curve(
        signal_eff=signal_efficiencies,
        background_rej=background_rej,
        labels=labels,
        colors=colors,
        title="Quark-Gluon Tagger"
        # file_path="../../Plots/ROC.pdf"
    )
