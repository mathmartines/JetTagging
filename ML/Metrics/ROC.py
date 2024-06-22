"""Evaluates the ROC curve and the AUC score for all the trainned models"""

from sklearn.metrics import roc_curve, auc
from utilities.graphics import plot_roc_curve
import keras
from ML.CNN.data_loader import load_data_qg_tagging as load_data_cnn


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the X_test data and returns the signal efficiency (recall) and
    the background rejection (FPR).
    """
    y_pred = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test[:, 0], y_pred[:, 1])
    return tpr, fpr


if __name__ == "__main__":
    # CNN
    cnn_model = keras.models.load_model("../CNN/Testing Algorithms/QG_Tag_Model_3.keras")
    X_test_cnn, y_test_cnn = load_data_cnn(
        quark_data_path="../../Data/Trainning/q_jets.csv",
        gluon_data_path="../../Data/Trainning/g_jets.csv",
        eta_range=(-0.4, 0.4),
        phi_range=(-0.4, 0.4),
        n_eta_bins=16,
        n_phi_bins=16
    )
    signal_eff_cnn, back_miss_id_rate_cnn = evaluate_model(cnn_model, X_test_cnn, y_test_cnn)

    # PointNet
    # custom_objects = {'PointNetLayer': PointNetLayer, 'PointNet': PointNet, 'MLP': MLP}
    # point_net_model = keras.models.load_model("../ModelFiles/PointNet.keras", custom_objects=custom_objects)
    # X_test, y_test = load_data_as_set_of_particles("../../Data/HiggsTest.csv")
    # signal_eff_pn, fpr_pn = evaluate_model(point_net_model, X_test, y_test)

    # Particle Cloud
    # custom_objects_pc = {'ParticleCloud': ParticleCloud, 'EdgeConvLayer': EdgeConvLayer, 'MLP': MLP}
    # particle_cloud = keras.models.load_model("../ModelFiles/ParticleCloud.keras", custom_objects=custom_objects_pc)
    # signal_eff_pc, fpr_pc = evaluate_model(particle_cloud, X_test, y_test)

    # Combined
    # combined = keras.models.load_model("../ModelFiles/CombinedModel.keras",
    #                                    custom_objects={**custom_objects, **custom_objects})
    # signal_eff_combined, fpr_combined = evaluate_model(combined, X_test, y_test)

    signal_efficiencies = {
        "imagem": signal_eff_cnn,
        # "MLP": signal_eff_mlp,
        # "ParticleCloud": signal_eff_pc,
        # "Combined": signal_eff_combined
    }
    background_rej = {
        "imagem": 1 - back_miss_id_rate_cnn,
        # "PointNet": 1 - fpr_pn,
        # "ParticleCloud": 1 - fpr_pc,
        # "Combined": 1 - fpr_combined
    }
    labels = {
        "imagem": f"CNN JetImages (AUC = {auc(back_miss_id_rate_cnn, signal_eff_cnn):.3f})",
        # "PointNet": f"Point-Net (AUC = {auc(fpr_pn, signal_eff_pn):.2f})",
        # "ParticleCloud": f"Particle Cloud (AUC = {auc(fpr_pc, signal_eff_pc):.2f})",
        # "Combined": f"Point-Net + Particle Cloud (AUC = {auc(fpr_combined, signal_eff_combined):.2f})",
    }
    colors = {
        "imagem": "#01204E",
        # "PointNet": "#E9C874",
        # "ParticleCloud": "#A34343",
        # "Combined": "darkgray"
    }

    plot_roc_curve(
        signal_eff=signal_efficiencies,
        background_rej=background_rej,
        labels=labels,
        colors=colors,
        title="Quark-Gluon Tagger"
        # file_path="../../Plots/ROC.pdf"
    )
