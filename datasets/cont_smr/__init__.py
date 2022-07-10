import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from lightning_torch.utils.file_mgmt import get_dir_by_indicator

def load_subjects_by_filter(data_path, filter_key, outdir=None):
    # DATA_PATH = Path(data_path)
    # mat_files = []
    # pattern = "*.mat"
    # for file in DATA_PATH.glob(pattern):
    #     mat_files.append(file)
    #
    # group_files = {}
    # for f in mat_files:
    #     res = f.stem.split("_", 1)
    #     if res[0] in group_files:
    #         group_files[res[0]][res[1]] = f
    #     else:
    #         group_files[res[0]] = {}
    #         group_files[res[0]][res[1]] = f
    # if filter_key == "gender":
    pass


def load_all_mat_sessions(data_path, outdir=None):
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

    res = {}
    for subject, sessions in group_files.items():
        if subject not in res:
            res[subject] = {}
            for session_id, session_path in sessions.items():
                if session_id not in res[subject]:
                    res[subject][session_id] = {}
                    data, timepoints, fs, clab, mnt, trial_info, metadata = load_data_from_mat(session_path)
                    res[subject][session_id]["data"] = data
                    res[subject][session_id]["timepoints"] = timepoints
                    res[subject][session_id]["fs"] = fs
                    res[subject][session_id]["clab"] = clab
                    res[subject][session_id]["mnt"] = mnt
                    res[subject][session_id]["trial_info"] = trial_info
                    res[subject][session_id]["metadata"] = metadata
            if outdir is not None:
                outfile_name = os.path.join(outdir, f"{subject}_{session_id}")
                np.savez(outfile_name, X=data, timepoints=timepoints, fs=fs, clab=clab, mnt=mnt, trial_info=trial_info,
                         metadata=metadata)

    return res


def load_data_from_mat(mat_file):
    mdata = loadmat(mat_file, mat_dtype=True)["BCI"]

    # data for all 450 trials:
    data = mdata["data"][0][0][0]

    # time:
    timepoints = mdata["time"][0][0][0]

    # srate
    fs = mdata["SRATE"][0][0][0][0]

    # chaninfo = defaultdict(list)
    # chaninfo_keys = mdata["chaninfo"][0][0][0].dtype.names
    # Extracting channel names and mnt info, excluding the reference electrode (REF)
    chan_inf = mdata["chaninfo"][0][0][0]["electrodes"][0][0]
    clab = []
    mnt = []
    for element in chan_inf[:-1]:
        clab.append(str(element[0][0]))
        mnt.append([element[1][0][0], element[2][0][0], element[3][0][0]])
    mnt = np.array(mnt)
    clab = np.array(clab)

    trial_info = defaultdict(list)

    trial_info_keys = mdata["TrialData"][0][0][0].dtype.names
    for trial in mdata["TrialData"][0][0][0]:
        for i, key in enumerate(trial_info_keys):
            trial_info[key].append(trial[i][0][0])

    metadata = dict()

    metadata_keys = mdata["metadata"][0][0][0].dtype.names

    for s in mdata["metadata"][0][0][0]:
        for i, key in enumerate(metadata_keys):
            metadata[key] = s[i][0][0]
    # trial artifact from the original data set, if bandpass filtered data from any electrode crosses threshold of +-100 μV = True, else False
    return data, timepoints, fs, clab, mnt, trial_info, metadata


def load_single_mat_session(subjectno, sessionno, data_path, outdir=None):
    if data_path is None:
        root_dir = get_dir_by_indicator(indicator="ROOT")
        DATA_PATH = Path(root_dir).parent / "data" / "raw"  # the data folder has to be in the parent folder!
    else:
        DATA_PATH = Path(data_path)
    group_files = {}

    mat_files = []
    pattern = "*.mat"
    for file in DATA_PATH.glob(pattern):
        mat_files.append(file)
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
    file_path = os.path.join(DATA_PATH, group_files[subject][session])
    if os.path.exists(file_path):
        mdata = loadmat(file_path, mat_dtype=True)["BCI"]
    else:
        raise FileNotFoundError(f"{file_path} file does not exist !")

    data, timepoints, fs, clab, mnt, trial_info, metadata = load_data_from_mat(file_path)

    if outdir is not None:
        outfile_name = os.path.join(outdir, f"{subject}_{session}")
        np.savez(outfile_name, X=data, timepoints=timepoints, fs=fs, clab=clab, mnt=mnt, trial_info=trial_info,
                 metadata=metadata)
    else:
        return data, timepoints, fs, clab, mnt, trial_info, metadata


def trial_ids_key(data, keys):
    pass


if __name__ == "__main__":
    root_dir = get_dir_by_indicator(indicator="ROOT")
    RAW_DATA_PATH = os.path.join(root_dir, "data/SMR/raw")
    PREPROCESS_DATA_PATH = os.path.join(root_dir, "data/SMR/processed")
    # from bbcpy.load import eeg
    # from bbcpy.datatypes.eeg import Data,Chans,Epo
    # from bbcpy.functions.statistics import cov
    # res = eeg.data(fname=os.path.join(PREPROCESS_DATA_PATH,"S1_Session_1.npz"))
    res_dict = load_all_mat_sessions(RAW_DATA_PATH)
    print(res_dict.keys())
    # obj = Data(X, fs, chans=Chans(clab,mnt))

    # ep = Epo(data=X, time=)
