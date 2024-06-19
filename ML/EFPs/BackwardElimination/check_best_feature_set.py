"""Plot the accuracy of the DT for a given number of features."""

import json
import numpy as np


if __name__ == "__main__":
    file_name = "TopTagging_BestFeatures_Set.json"

    # loading the JSON file
    file_ = open(file_name, "r")
    back_elimination_history = json.load(file_)
    file_.close()

    # accuracies
    accuracies = back_elimination_history['best_accuracy']
    set_of_features = back_elimination_history['best_features_set']

    # fiding the set with the highest accuracy
    indices_best_accuracies = np.argsort(accuracies)
    # For Top-Tagging the best accuracy is achived with index -2 (18 polynomials)
    # For Quark-Gluon the best accuracy is achived with index -1 (10 polynomials)
    index_best_acc = indices_best_accuracies[-2]
    print(f"best accuracy is at index {index_best_acc}, with value {accuracies[index_best_acc]:.5f}")
    print(f"best set of features has {len(set_of_features[index_best_acc])} polynomials")
    print(set_of_features[index_best_acc])
