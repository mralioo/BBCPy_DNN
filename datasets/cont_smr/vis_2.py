import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from bbcpy.utils.data import normalize
from bbcpy.utils.file_management import get_dir_by_indicator
from bbcpy.utils.subject import load_bbci_data
from matplotlib import rc
from matplotlib.collections import LineCollection

sns.set_theme(style="darkgrid")
font = {'family': 'monospace',
        'weight': 'bold',
        'size': 10}
rc('font', **font)


def plot_eeg(raw_data, num_trial, ch_names, timepoints, fs=1000., norm_type="std", axis=1):
    EEG_norm, x_norm_params = normalize(raw_data, norm_type="std", axis=axis)

    EEG = EEG_norm[num_trial]

    n_samples, n_rows = EEG.shape[-1], len(ch_names)
    # t = timepoints / fs
    t = np.arange(n_samples) / fs

    dmin = EEG.min()
    dmax = EEG.max()
    dr = (dmax - dmin) * 0.5  # Crowd them a bit.
    y0 = dmin
    y1 = (n_rows - 1) * dr + dmax

    ticklocs = []
    segs = []
    for i in range(n_rows):
        segs.append(np.column_stack((t, EEG[i, :])))
        ticklocs.append(i * dr)

    offsets = np.zeros((n_rows, 2), dtype=float)
    offsets[:, 1] = ticklocs

    fig, ax = plt.subplots(figsize=(15, 10))

    lines = LineCollection(segs, offsets=offsets, transOffset=None)
    ax.add_collection(lines)
    ax.set_ylim(y0, y1)

    ax.set_yticks(ticklocs)
    ax.set_yticklabels(ch_names, rotation=30, fontsize=8)
    ax.xaxis.set_ticklabels(timepoints, fontsize=6, color='black')
    ax.set_ylabel('Channels')
    ax.set_xlabel('Time (s)')
    ax.set_title("S{:}s{:}\ntrial_{:}\n norm:{:}_axis{:}".format(1, 1, num_trial, norm_type, axis))

    # for sig, ch_name in zip(EEG, ch_names):
    #     ax.plot(time, ch_name)

    fig.tight_layout()
    plt.show()


def plot_cm(engine, class_type, figsize=(15, 6), fontsize=16):
    cm = engine.state.metrics["cm"].numpy().astype(int)
    if class_type == "LR":
        class_names = ["L", "R"]
        num_classes = len(class_names)
    else:
        raise NotImplemented

    group_counts = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            group_counts.append("{}/{}".format(cm[i, j], cm.sum(axis=1)[j]))
    group_percentages = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    group_percentages = np.nan_to_num(group_percentages)
    labels = [f"{v1} \n {v2 * 100: .4}%" for v1, v2 in zip(group_counts, group_percentages.flatten())]
    labels = np.asarray(labels).reshape(num_classes, num_classes)
    plt.ioff()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=labels, ax=ax, fmt="", cmap="Blues", cbar=True)
    # sns.heatmap(cm, annot=True, ax=ax, fmt="", cmap="Blues", cbar=True)

    # labels, title and ticks
    ax.set_xlabel('Predicted labels', fontsize=fontsize, color='black')
    ax.set_ylabel('True labels', fontsize=fontsize, color='black')
    ax.set_title('Confusion Matrix on Validation set')
    # True_class_name = [str(i) for i in range(len(class_names))]
    ax.xaxis.set_ticklabels(class_names, fontsize=20, color='black')
    ax.yaxis.set_ticklabels(class_names, fontsize=20, color='black')

    return fig


if __name__ == "__main__":
    root_dir = get_dir_by_indicator(indicator="ROOT")
    DATA_PATH = Path(root_dir).parent / "data"
    file_name = os.path.join(DATA_PATH, "S1_Session_1.mat")

    raw_data = load_bbci_data(data_path=DATA_PATH, subjects_list=[1], sessions_list=[[1]], task_type="LR",
                              time_interval=[-6001, -1], merge_sessions=True, reshape_type="left_pad")

    # raw_data = load_bbci_data(data_path=DATA_PATH, subjects_list=[1], sessions_list=[[1]], task_type="LR",
    #                           time_interval=[2000, 8000], merge_sessions=True, reshape_type="slice")

    X = np.concatenate([se.bci_recording["data"] for se in raw_data], axis=0)
    Y = np.expand_dims(np.concatenate([se.bci_recording["label"] for se in raw_data], axis=0), -1)
    # x_norm, x_norm_params = normalize(X, norm_type="std", axis=1)
    # times = np.linspace(0, 1, 100, endpoint=False)
    # sine = np.sin(20 * np.pi * times)
    # cosine = np.cos(10 * np.pi * times)
    # # data = np.array([sine, cosine])
    #
    # tdata = np.array([[0.2 * sine, 1.0 * cosine],
    #                  [0.4 * sine, 0.8 * cosine],
    #                  [0.6 * sine, 0.6 * cosine],
    #                  [0.8 * sine, 0.4 * cosine],
    #                  [1.0 * sine, 0.2 * cosine]])

    # mdata = loadmat(file_name, mat_dtype=True)["BCI"]
    # info = {x: str(y[0]) for x, y in mdata.dtype.fields.items()}
    # mdata, fs, clab, mnt, mrk_class, mrk_className, task_type, task_typeName, timepoints, trial_artifact = load_matlab_data_fast(
    #     subjectno=1, sessionno=1, data_path=DATA_PATH)

    # info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")
    #
    # simulated_raw = mne.io.RawArray(X[0], info)
    # simulated_raw.plot(duration=2. , n_channels=1, scalings=None)
    # simulated_raw.plot(scalings=dict(eeg=100e-6), duration=1, start=14)

    # data_raw = mne.EpochsArray(X, info)
    # data_raw.plot(picks='eeg', show_scrollbars=False)

    ch_names = list(raw_data[0].clab[0:10])
    # ch_names = list(raw_data[0].clab)
    timepoints = raw_data[0].bci_recording["timepoints"]
    fs = raw_data[0].fs[0]
    #
    num_trial = 5
    plot_eeg(raw_data=X, num_trial=num_trial, timepoints=timepoints, ch_names=ch_names, fs=fs, norm_type=None,
             axis=None)

    plot_eeg(raw_data=X, num_trial=num_trial, timepoints=timepoints, ch_names=ch_names, fs=fs, norm_type="std", axis=0)
    plot_eeg(raw_data=X, num_trial=num_trial, timepoints=timepoints, ch_names=ch_names, fs=fs, norm_type="std", axis=1)
    plot_eeg(raw_data=X, num_trial=num_trial, timepoints=timepoints, ch_names=ch_names, fs=fs, norm_type="std", axis=2)

    plot_eeg(raw_data=X, num_trial=num_trial, timepoints=timepoints, ch_names=ch_names, fs=fs, norm_type="minmax",
             axis=0)
    plot_eeg(raw_data=X, num_trial=num_trial, timepoints=timepoints, ch_names=ch_names, fs=fs, norm_type="minmax",
             axis=1)
    plot_eeg(raw_data=X, num_trial=num_trial, timepoints=timepoints, ch_names=ch_names, fs=fs, norm_type="minmax",
             axis=2)
