"""Evaluates the ROC curve and the AUC score for all the trainned models"""

from sklearn.metrics import roc_curve, auc
from utilities.graphics import plot_roc_curve
import keras
from ML.CNN.data_loader import load_data_qg_tagging as load_data_cnn
from ML.ParticleCloud.data_loaders import load_data_qg_tagging as load_data_pc
from ML.EFPs.data_loaders import load_data_qg_tagging as load_data_efps
from src.Preprocessing.norm import norm
import numpy as np
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


def find_best_index(signal_eff):
    return np.where(signal_eff > 0.01)[0][0]


if __name__ == "__main__":
    # CNN
    cnn = keras.models.load_model("../CNN/QuarkGluonTagger/Quark_Gluon_Tagging_CNN.keras")
    X_test_cnn, y_test_cnn = load_data_cnn(
        quark_data_path="../../Data/Test/q_jets.csv",
        gluon_data_path="../../Data/Test/g_jets.csv",
        eta_range=(-0.4, 0.4),
        phi_range=(-0.4, 0.4),
        n_eta_bins=16,
        n_phi_bins=16
    )
    predict_cnn = cnn.predict(X_test_cnn)
    signal_eff_cnn, back_miss_id_rate_cnn = evaluate_model(y_test_cnn[:, 0], predict_cnn[:, 0])
    cnn_index = find_best_index(signal_eff=signal_eff_cnn)

    # PointNet
    point_net = keras.models.load_model("../ParticleCloud/PointNet/QuarkGluon_Tagging_PointNet.keras")
    X_pc, y_pc = load_data_pc(
        quark_data_path="../../Data/Test/q_jets.csv", gluon_data_path="../../Data/Test/g_jets.csv"
    )
    predict_pt = point_net.predict(X_pc)
    signal_eff_pn, back_miss_id_rate_pn = evaluate_model(y_pc[:, 0], predict_pt[:, 0])
    pn_index = find_best_index(signal_eff=signal_eff_pn)

    # Particle Cloud
    particle_cloud = keras.models.load_model("../ParticleCloud/ParticleNet/QuarkGluon_Tagging_ParticleCloud.keras")
    predict_pc = particle_cloud.predict(X_pc)
    signal_eff_pc, back_miss_id_rate_pc = evaluate_model(y_pc[:, 0], predict_pc[:, 0])
    pc_index = find_best_index(signal_eff=signal_eff_pc)

    # Decision Trees with EFPs
    decision_tree = load_joblib_object("../EFPs/DecisionTree/QuarkGluon_Tagging_DT.joblib")
    X_efps, y_efps = load_data_efps(
        quark_data_path="../../Data/Test/q_jets_efps_d5_primed.npy",
        gluon_data_path="../../Data/Test/g_jets_efps_d5_primed.npy"
    )
    predict_dt = decision_tree.predict_proba(X_efps)
    signal_eff_dt, back_miss_id_rate_dt = evaluate_model(y_efps, predict_dt[:, 1])
    dt_index = find_best_index(signal_eff=signal_eff_dt)

    # BDTs
    best_features_set = [0, 1, 5, 6, 12, 17, 33, 42, 43, 48]

    # Boosted Decision Trees
    bdt = load_joblib_object("../EFPs/BoostedDecisionTrees/QuarkGluon_Tagging_BoostedDT.joblib")
    predict_bdt = bdt.predict_proba(X_efps[:, best_features_set])
    signal_eff_bdt, back_miss_id_rate_bdt = evaluate_model(y_efps, predict_bdt[:, 1])
    bdt_index = find_best_index(signal_eff=signal_eff_bdt)

    # Random Forests
    random_forests = load_joblib_object("../EFPs/RandomForests/QuarkGluon_Tagging_RandomForest.joblib")
    predict_random_forests = random_forests.predict_proba(X_efps[:, best_features_set])
    signal_eff_random_forests, back_miss_id_rate_random_forests = evaluate_model(y_efps, predict_random_forests[:, 1])
    rf_index = find_best_index(signal_eff=signal_eff_random_forests)

    # Linear discriminat analysis
    discriminant = load_joblib_object("../EFPs/DiscriminantAnalysis/LinearDiscriminant_qg_n3.joblib")
    predict_discriminant = discriminant.predict_proba(norm(3, X_efps))
    signal_eff_disc, back_miss_id_rate_disc = evaluate_model(y_efps, predict_discriminant[:, 1])
    disc_index = find_best_index(signal_eff=signal_eff_disc)

    signal_efficiencies = {
        "particlecloud": signal_eff_pc[pc_index:],
        "pointnet": signal_eff_pn[pn_index:],
        "bdt": signal_eff_bdt[bdt_index:],
        "random_forest": signal_eff_random_forests[rf_index:],
        "decision_tree": signal_eff_dt[dt_index:],
        "imagem": signal_eff_cnn[cnn_index:],
        "discriminant": signal_eff_disc[disc_index:],
    }
    background_rej = {
        "imagem": 1. / back_miss_id_rate_cnn[cnn_index:],
        "pointnet": 1. / back_miss_id_rate_pn[pn_index:],
        "particlecloud": 1. / back_miss_id_rate_pc[pc_index:],
        "decision_tree": 1. / back_miss_id_rate_dt[dt_index:],
        "bdt": 1. / back_miss_id_rate_bdt[bdt_index:],
        "random_forest": 1. / back_miss_id_rate_random_forests[rf_index:],
        "discriminant": 1 / back_miss_id_rate_disc[disc_index:]

    }
    labels = {
        "imagem": f"CNN JetImages (AUC = {auc(back_miss_id_rate_cnn, signal_eff_cnn):.3f})",
        "pointnet": f"Point-Net (AUC = {auc(back_miss_id_rate_pn, signal_eff_pn):.3f})",
        "particlecloud": f"Particle Cloud (AUC = {auc(back_miss_id_rate_pc, signal_eff_pc):.3f})",
        "decision_tree": f"Decision-Tree EFPs (AUC = {auc(back_miss_id_rate_dt, signal_eff_dt):.3f})",
        "bdt": f"Boosted DTs EFPs (AUC = {auc(back_miss_id_rate_bdt, signal_eff_bdt):.3f})",
        "random_forest": f"Random Forests EFPs (AUC = {auc(back_miss_id_rate_random_forests, signal_eff_random_forests):.3f})",
        "discriminant": f"Discriminant Analysis EFPs (AUC = {auc(back_miss_id_rate_disc, signal_eff_disc):.3f})",
    }
    colors = {
        "nn": "#AF47D2",
        "imagem": "#01204E",
        "pointnet": "#E9C874",
        "particlecloud": "#A34343",
        "decision_tree": "#588157",
        "bdt": "#FF7F3E",
        "random_forest": "#948979",
        "discriminant": "#C65BCF"
    }

    linestyle = {
        "nn": "dashdot",
        "particlecloud": "dashed",
        "imagem": "dashdot",
        "pointnet": "dotted",
        "bdt": "dashdot",
        "random_forest": "dashed",
        "decision_tree": "solid",
        "discriminant": "solid"
    }

    plot_roc_curve(
        signal_eff=signal_efficiencies,
        background_rej=background_rej,
        labels=labels,
        colors=colors,
        linestyle=linestyle,
        title="Quark-Gluon Tagger",
        file_path="../../Plots/ROC_QG.pdf"
    )
