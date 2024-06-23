"""Evaluates the ROC curve and the AUC score for all the trainned models"""

from sklearn.metrics import roc_curve, auc
from utilities.graphics import plot_roc_curve
import keras
from ML.CNN.data_loader import load_data_qg_tagging as load_data_cnn
from ML.ParticleCloud.data_loaders import load_data_qg_tagging as load_data_pc
from ML.EFPs.data_loaders import load_data_qg_tagging as load_data_efps
from src.Preprocessing.norm import norm
from sklearn.preprocessing import StandardScaler
from ML.EFPs.data_loaders import load_data_qg_tagging_wR
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
        quark_data_path="../../Data/Test/q_jets.csv",
        gluon_data_path="../../Data/Test/g_jets.csv",
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
    particle_cloud = keras.models.load_model("../ParticleCloud/ParticleNet/QuarkGluon_Tagging_ParticleCloud_Final.keras")
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

    # BDTs
    best_features_set = [0, 1, 5, 6, 12, 17, 33, 42, 43, 48]

    # Boosted Decision Trees
    bdt = load_joblib_object("../EFPs/BoostedDecisionTrees/QuarkGluon_Tagging_BoostedDT.joblib")
    predict_bdt = bdt.predict_proba(X_efps[:, best_features_set])
    signal_eff_bdt, back_miss_id_rate_bdt = evaluate_model(y_efps, predict_bdt[:, 1])

    # Random Forests
    random_forests = load_joblib_object("../EFPs/RandomForests/QuarkGluon_Tagging_RandomForest.joblib")
    predict_random_forests = random_forests.predict_proba(X_efps[:, best_features_set])
    signal_eff_random_forests, back_miss_id_rate_random_forests = evaluate_model(y_efps, predict_random_forests[:, 1])

    # Linear discriminat analysis
    discriminant = load_joblib_object("../EFPs/DiscriminantAnalysis/LinearDiscriminant_qg_n3.joblib")
    predict_discriminant = discriminant.predict_proba(norm(3, X_efps))
    signal_eff_disc, back_miss_id_rate_disc = evaluate_model(y_efps, predict_discriminant[:, 1])

    # regressão logística
    reg_log = load_joblib_object("../EFPs/LogisticRegression/LogisticRegression_qg_n3.joblib")
    predict_reg_log = reg_log.predict_proba(norm(3, X_efps))
    signal_eff_reg_log, back_miss_id_rate_reg_log = evaluate_model(y_efps, predict_reg_log[:, 1])

    # CNN EFPs
    cnn_efps = keras.saving.load_model("../EFPs/NN/NN_EFPs_QG_Tagging.keras")
    X_train, y_train = load_data_qg_tagging_wR(
        quark_data_path=f"../../Data/Test/q_jets_efps_d5_primed.npy",
        gluon_data_path=f"../../Data/Test/g_jets_efps_d5_primed.npy",
        mean_deltaRij_quark_path=f"../../Data/Test/mean_deltaRij_q.npy",
        mean_deltaRij_gluon_path=f"../../Data/Test/mean_deltaRij_g.npy",
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    predict_cnn_efps = cnn_efps.predict(X_train)
    signal_eff_cnn_efp, back_miss_id_rate_cnn_efp = evaluate_model(y_train, predict_cnn_efps)


    signal_efficiencies = {
        "reg_log": signal_eff_reg_log,
        "discriminant": signal_eff_disc,
        "imagem": signal_eff_cnn,
        "decision_tree": signal_eff_dt,
        "bdt": signal_eff_bdt,
        "random_forest": signal_eff_random_forests,
        "pointnet": signal_eff_pn,
        "particlecloud": signal_eff_pc,
        "cnn_efps": signal_eff_cnn_efp
    }
    background_rej = {
        "imagem": 1. - back_miss_id_rate_cnn,
        "decision_tree": 1. - back_miss_id_rate_dt,
        "bdt": 1. - back_miss_id_rate_bdt,
        "random_forest": 1. - back_miss_id_rate_random_forests,
        "discriminant": 1 - back_miss_id_rate_disc,
        "reg_log": 1 - back_miss_id_rate_reg_log,
        "pointnet": 1. - back_miss_id_rate_pn,
        "particlecloud": 1. - back_miss_id_rate_pc,
        "cnn_efps": 1 - back_miss_id_rate_cnn_efp
    }
    labels = {
        "imagem": f"CNN JetImages (AUC = {auc(back_miss_id_rate_cnn, signal_eff_cnn):.3f})",
        "pointnet": f"Point-Net (AUC = {auc(back_miss_id_rate_pn, signal_eff_pn):.3f})",
        "particlecloud": f"Particle Cloud (AUC = {auc(back_miss_id_rate_pc, signal_eff_pc):.3f})",
        "decision_tree": f"Decision-Tree EFPs (AUC = {auc(back_miss_id_rate_dt, signal_eff_dt):.3f})",
        "bdt": f"Boosted DTs EFPs (AUC = {auc(back_miss_id_rate_bdt, signal_eff_bdt):.3f})",
        "random_forest": f"Random Forests EFPs (AUC = {auc(back_miss_id_rate_random_forests, signal_eff_random_forests):.3f})",
        "discriminant": f"Discriminant Analysis EFPs (AUC = {auc(back_miss_id_rate_disc, signal_eff_disc):.3f})",
        "reg_log": f"Logistic Regression EFPs (AUC = {auc(back_miss_id_rate_reg_log, signal_eff_reg_log):.3f})",
        "cnn_efps": rf"CNN EFPs + mean $\Delta R_{{ij}}$ (AUC = {auc(back_miss_id_rate_cnn_efp, signal_eff_cnn_efp):.3f})"
    }
    colors = {
        "nn": "#AF47D2",
        "imagem": "#01204E",
        "pointnet": "#E9C874",
        "particlecloud": "#A34343",
        "decision_tree": "#588157",
        "bdt": "#FF7F3E",
        "random_forest": "#948979",
        "discriminant": "#C65BCF",
        "reg_log": "#03AED2",
        "cnn_efps": "#891652"
    }

    linestyle = {
        "nn": "dashdot",
        "particlecloud": "dashed",
        "imagem": "dashdot",
        "pointnet": "dotted",
        "bdt": "dashdot",
        "random_forest": "dashed",
        "decision_tree": "solid",
        "discriminant": "solid",
        "reg_log": "solid",
        "cnn_efps": "solid"
    }

    plot_roc_curve(
        signal_eff=signal_efficiencies,
        background_rej=background_rej,
        labels=labels,
        colors=colors,
        linestyle=linestyle,
        title="Quark-Gluon Tagger",
        file_path="../../Plots/ROC_QG/ROC_PC.png"
    )
