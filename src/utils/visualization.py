import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_reconstruction_error(errors, threshold=None):
    plt.figure(figsize=(10, 5))
    sns.histplot(errors, bins=50, kde=True, color='skyblue')
    if threshold is not None:
        plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
    plt.title('Reconstruction Error Distribution')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def plot_roc_curve(fpr, tpr, auc_score):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show() 