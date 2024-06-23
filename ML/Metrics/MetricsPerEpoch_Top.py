"""Plots the metrics per epoch"""

import json
from utilities.graphics import plot_metric_per_epoch


def load_json(json_file_path):
    """Loads the json file and returns it as a dictionary"""
    file_ = open(json_file_path, "r")
    return json.load(file_)


if __name__ == "__main__":
    # loading models
    model_files = {
        "imagem": "../CNN/Top tagger/top-tagger-final.json",
        # "point_net": "../ParticleCloud/PointNet/Top_Tagging_PointNet.json",
        # "particle_cloud": "../ParticleCloud/ParticleNet/Top_Tagging_ParticleCloud.json",
    }
    # metrics for each of the models
    all_metrics = {model: load_json(model_file) for model, model_file in model_files.items()}

    # desired metric
    metric = "accuracy"
    dict_metrics = {}
    for model_name in all_metrics:
        dict_metrics[model_name] = all_metrics[model_name][metric]
        dict_metrics[f"val_{model_name}"] = all_metrics[model_name][f"val_{metric}"]

    # color dict
    color_dict = {
        "val_imagem": "#01204E",
        "imagem": "#01204E",
        "point_net": "darkgray",
        "val_point_net": "darkgray",
        "particle_cloud": "#A34343",
        "val_particle_cloud": "#A34343",
    }

    labels_dict = {
        "val_imagem": f"CNN - JetImages Validation",
        "imagem": f"CNN - JetImages Trainning",
        "point_net": f"Point-Net Trainning",
        "val_point_net": f"Point-Net Validation",
        "particle_cloud": f"Particle Cloud Trainning",
        "val_particle_cloud": f"Particle Cloud Validation",
        # "combined": f"Point-Net + Particle Cloud Trainning",
        # "val_combined": f"Point-Net + Particle Cloud Validation"
    }

    plot_metric_per_epoch(
        metrics=dict_metrics,
        labels=labels_dict,
        colors=color_dict,
        xlim=(0, 50),
        metric_name="Accuracy",
        title="Top Tagger",
        file_path="../../Plots/PerEpoch/Acc_Top_Tagger_Imagem.png"
    )