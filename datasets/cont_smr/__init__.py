"""
Source from project ss21 BCI
"""
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from lightning_torch.utils.file_mgmt import get_dir_by_indicator


def load_matlab_data_fast(subjectno, sessionno, data_path, outfile=False):
    if data_path is None:
        root_dir = get_dir_by_indicator(indicator="ROOT")
        DATA_PATH = Path(root_dir).parent / "data" / "raw"  # the data folder has to be in the parent folder!
    else:
        DATA_PATH = Path(data_path)

    mat_files = []
    pattern = "*.mat"
    for file in DATA_PATH.glob(pattern):
        mat_files.append(file)

    group_files = {}
    for f in mat_files:
        res = f.stem.split("_", 1)
        if res[0] in group_files:
            group_files[res[0]][res[1]] = f
        else:
            group_files[res[0]] = {}
            group_files[res[0]][res[1]] = f

    # Select the subject person and the session number to load the data
    subject = "S" + str(subjectno)
    session = "Session_" + str(sessionno)
    mdata = loadmat(os.path.join(DATA_PATH, group_files[subject][session]), mat_dtype=True)["BCI"]

    # data for all 450 trials:
    data = mdata[0][0][0][:][0]

    # time:
    timepoints = mdata[0][0][1][0]

    # srate
    fs = mdata[0][0][4][0]

    # Extracting channel names and mnt info, excluding the reference electrode
    chan_inf = np.array(mdata[0][0][7][0][0][2][0])
    clab = []
    mnt = []
    for element in chan_inf[:-1]:
        clab.append(str(element[0][0]))
        mnt.append([element[1][0][0], element[2][0][0], element[3][0][0]])
    mnt = np.array(mnt)
    clab = np.array(clab)

    # Extracting marker classes, original range 1-4, modified range 0-3 to match the indices of mrk_className
    # also extracting task type, original range 1-3, modified range 0-2 to match the indices of task_typeName
    # trial artifact from the original data set, if bandpass filtered data from any electrode crosses threshold of +-100 μV = True, else False
    mrk_className = ['right', 'left', 'up', 'down']
    task_typeName = ['LR', 'UD', '2D']
    meta = mdata[0][0][5][:][0]
    task_type = []
    mrk_class = []
    trial_artifact = []
    for i in range(0, len(meta)):
        task_type.append(meta[i][0][0][0] - 1)
        mrk_class.append(meta[i][3][0][0] - 1)
        trial_artifact.append(meta[i][9][0][0])
    mrk_class = np.array(mrk_class)
    task_type = np.array(task_type)
    trial_artifact = np.array(trial_artifact)

    if (outfile):
        np.savez(outfile, data=data, fs=fs, clab=clab, mnt=mnt, mrk_class=mrk_class, mrk_className=mrk_className,
                 task_type=task_type, task_typeName=task_typeName, timepoints=timepoints)
        return

    # trial artifact from the original data set, if bandpass filtered data from any electrode crosses threshold of +-100 μV = True, else False
    mrk_className = ['right', 'left', 'up', 'down']
    task_typeName = ['LR', 'UD', '2D']
    meta = mdata[0][0][5][:][0]
    task_type = []
    mrk_class = []
    trial_artifact = []
    for i in range(0, len(meta)):
        task_type.append(meta[i][0][0][0] - 1)
        mrk_class.append(meta[i][3][0][0] - 1)
        trial_artifact.append(meta[i][9][0][0])
    mrk_class = np.array(mrk_class)
    task_type = np.array(task_type)
    trial_artifact = np.array(trial_artifact)
    if (outfile):
        np.savez(outfile, data=data, fs=fs, clab=clab, mnt=mnt, mrk_class=mrk_class, mrk_className=mrk_className,
                 task_type=task_type, task_typeName=task_typeName, timepoints=timepoints, trial_artifact=trial_artifact)
        return

    else:
        return data, fs, clab, mnt, mrk_class, mrk_className, task_type, task_typeName, timepoints, trial_artifact
