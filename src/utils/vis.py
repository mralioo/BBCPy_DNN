import numpy as np
import pyrootutils
from matplotlib import pyplot as plt


def compute_percentages_cm(cm):
    group_percentages = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    return group_percentages


def calculate_cm_stats(cm_list, num_classes):
    # Initialize empty matrices for average and standard deviation
    avg_cm = np.zeros((num_classes, num_classes))
    std_cm = np.zeros((num_classes, num_classes))
    k = len(cm_list)

    # Iterate through each fold's confusion matrix and add to the average matrix
    for cm in cm_list:
        avg_cm += cm

    # Divide by the total number of folds to get the average matrix
    avg_cm /= k

    # Calculate the standard deviation matrix
    for cm in cm_list:
        std_cm += (cm - avg_cm) ** 2

    std_cm = np.sqrt(std_cm / k)

    return avg_cm, std_cm


def plot_roc_curve(tpr, fpr, scatter=False, ax=None):
    '''
    Plots the ROC Curve by using the list of coordinates (tpr and fpr).

    Args:
        tpr: The list of TPRs representing each coordinate.
        fpr: The list of FPRs representing each coordinate.
        scatter: When True, the points used on the calculation will be plotted with the line (default = True).
        ax: Matplotlib axis object. If None, a new figure and axis will be created.
    '''
    if ax == None:
        _, ax = plt.subplots(figsize=(5, 5))

    if scatter:
        ax.scatter(fpr, tpr)

    ax.plot(fpr, tpr, label="ROC curve")
    ax.plot([0, 1], [0, 1], color='green', linestyle='--', label="Random classifier")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
