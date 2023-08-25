import numpy as np
import pyrootutils

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
