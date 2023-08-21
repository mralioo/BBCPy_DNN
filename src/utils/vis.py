import itertools
import os

import mlflow
import numpy as np
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

    # Calculate the mean matrix
    # mean_cm = np.mean(avg_cm)
    #
    # # Print the average, standard deviation, and mean matrices
    # print("Average Confusion Matrix:")
    # print(avg_cm)
    #
    # print("Standard Deviation Matrix:")
    # print(std_cm)
    #
    # print("Mean Matrix:")
    # print(mean_cm)

    return avg_cm, std_cm


def confusion_matrix_to_png(conf_mat, class_names, title, figure_file_name=None, type='standard'):
    if type == 'standard':
        plt.rcParams["font.family"] = 'DejaVu Sans'
        figure = plt.figure(figsize=(9, 9))

        plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        # set title
        plt.title(title)

        # render the confusion matrix with percentage and ratio.
        group_counts = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                group_counts.append("{}/{}".format(conf_mat[i, j], conf_mat.sum(axis=1)[i]))
        group_percentages = np.around(conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis], decimals=2)
        labels = [f"{v1} \n {v2 * 100: .4}%" for v1, v2 in zip(group_counts, group_percentages.flatten())]
        labels = np.asarray(labels).reshape(len(class_names), len(class_names))

        # set the font size of the text in the confusion matrix
        for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
            color = "red"
            plt.text(j, i, labels[i, j], horizontalalignment="center", color=color, fontsize=15)

        plt.tight_layout()
        plt.ylabel('True label', fontsize=10)
        plt.xlabel('Predicted label', fontsize=10)

        if figure_file_name is None:
            fig_file_path = f'{title}.png'
        else:

            fig_file_path = f'{figure_file_name}.png'
        mlflow_artifact_path = mlflow.get_artifact_uri().replace("file:///", "")
        plt_cm_file_path = os.path.join(mlflow_artifact_path, fig_file_path)
        plt.savefig(plt_cm_file_path)
        mlflow.log_artifact(plt_cm_file_path)
        plt.close(figure)

    elif type == 'mean':

        # Fixme here I pass list of cm (better way)
        plt.rcParams["font.family"] = 'DejaVu Sans'
        figure = plt.figure(figsize=(9, 9))

        # Add values to the plot
        mean_cm_val, std_cm_val = calculate_cm_stats(cm_list=conf_mat, num_classes=len(class_names))

        plt.imshow(mean_cm_val, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        # set title
        plt.title(title)

        labels = [f"{v1 * 100:.4}%  ±{v2 * 100: .4}%" for v1, v2 in zip(mean_cm_val.flatten(), std_cm_val.flatten())]
        labels = np.asarray(labels).reshape(len(class_names), len(class_names))

        # thresh = mean_cm_val.max() / 2.0
        for i, j in itertools.product(range(mean_cm_val.shape[0]), range(mean_cm_val.shape[1])):
            color = "red"
            plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label', fontsize=10)
        plt.xlabel('Predicted label', fontsize=10)

        if figure_file_name is None:
            fig_file_path = f'{title}.png'
        else:
            fig_file_path = f'{figure_file_name}.png'

        mlflow_artifact_path = mlflow.get_artifact_uri().replace("file:///", "")
        plt_cm_mean_file_path = os.path.join(mlflow_artifact_path, fig_file_path)

        plt.savefig(plt_cm_mean_file_path)

        mlflow.log_artifact(plt_cm_mean_file_path)
        plt.close(figure)
