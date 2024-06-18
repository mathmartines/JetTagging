from sklearn.metrics import recall_score, precision_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import json


def display_metrics(y_true, y_pred):
    """Display performance metrics"""
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))


def display_roc_curve(y_true, y_score):
    """Display ROC curve and the AUC metric"""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR (Recall)')
    plt.show()
    print(f"AUC: {auc(fpr, tpr):.4f}")


def plot_hist_trainning(history):
    """Plot the trainning history."""
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca()


def save_model(model, history, model_name):
    """Saves the model and the history"""
    # saving the model
    model.save(f"{model_name}.keras")
    # saving the history
    with open(f"{model_name}.json", "w") as json_file:
        json.dump(history.history, json_file, indent=4)

