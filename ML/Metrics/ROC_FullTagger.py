"""Evaluates the ROC curve and the AUC score for all the trainned models"""

from sklearn.metrics import roc_curve, auc
from utilities.graphics import plot_roc_curve
import keras
import joblib
from ML.ParticleCloud.data_loaders import load_data_full_tagger_tagging
from src.FullTagger import KerasBinaryTaggerSoftMax, FullTagger


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
    # loading the tagger
    qg_pc_model = keras.models.load_model("../ParticleCloud/ParticleNet/QuarkGluon_Tagging_ParticleCloud_Final.keras")
    top_pc_model = keras.models.load_model("../ParticleCloud/ParticleNet/Top_Tagging_ParticleCloud_Final.keras")

    # creating the classes
    top_tagger = KerasBinaryTaggerSoftMax(classifier=top_pc_model)
    qg_tager = KerasBinaryTaggerSoftMax(classifier=qg_pc_model)
    full_tagger = FullTagger(top_tagger=top_tagger, quark_gluon_tagger=qg_tager)

    X, y_true = load_data_full_tagger_tagging(
        top_quark_path="../../Data/Test/t_jets.csv", quark_data_path="../../Data/Test/q_jets.csv",
        gluon_data_path="../../Data/Test/g_jets.csv"
    )

    # making the predictions
    y_proba = full_tagger.predict_proba(X)

    # signal and background efficiencies
    epsilon_s_top, epsilon_b_top = evaluate_model(y_true[:, 0], y_proba[:, 0])
    epsilon_s_quark, epsilon_b_quark = evaluate_model(y_true[:, 1], y_proba[:, 1])
    epsilon_s_gluon, epsilon_b_gluon = evaluate_model(y_true[:, 2], y_proba[:, 2])

    signal_efficiencies = {
        "top": epsilon_s_top,
        "quark": epsilon_s_quark,
        "gluon": epsilon_s_gluon
    }

    background_rej = {
        "top": 1 - epsilon_b_top,
        "quark": 1 - epsilon_b_quark,
        "gluon": 1 - epsilon_b_gluon
    }

    labels = {
        "top": f"Top tagging (AUC = {auc(epsilon_b_top, epsilon_s_top):.2f})",
        "quark": f"Light quark tagging (AUC = {auc(epsilon_b_quark, epsilon_s_quark):.2f})",
        "gluon": f"Gluon tagging (AUC = {auc(epsilon_b_gluon, epsilon_s_gluon):.2f})"
    }
    colors = {
        "top": "#A34343",
        "quark": "#01204E",
        "gluon": "#E9C874",
    #     "particlecloud": "#A34343",
    #     "decision_tree": "#588157",
    #     "bdt": "#FF7F3E",
    #     "random_forest": "#948979",
    #     "discriminant": "#C65BCF",
    #     "reg_log": "#03AED2",
    #     "cnn_efps": "#891652"
    }
    #
    linestyle = {
        "top": "solid",
        "quark": "solid",
        "gluon": "solid",
    }

    plot_roc_curve(
        signal_eff=signal_efficiencies,
        background_rej=background_rej,
        labels=labels,
        colors=colors,
        linestyle=linestyle,
        title="Full Tagger",
        file_path="../../Plots/ROC_FullTagger.pdf"
    )
