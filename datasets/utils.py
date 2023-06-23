import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io
from scipy.io import loadmat

from lightning_torch.utils.file_mgmt import get_dir_by_indicator


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def list_all_files(data_path, pattern="*.mat"):
    DATA_PATH = Path(data_path)
    mat_files = []

    for file in DATA_PATH.glob(pattern):
        mat_files.append(file)

    group_files = {}
    for f in mat_files:
        res = f.stem.split("_", 1)
        if len(res) > 1:
            if res[0] in group_files:
                group_files[res[0]][res[1]] = f
            else:
                group_files[res[0]] = {}
                group_files[res[0]][res[1]] = f
        else:
            group_files[res[0]] = f

    return group_files


def load_data_from_mat(mat_file, key=None):
    mdata = loadmat(mat_file, mat_dtype=True)["BCI"]

    # data for all 450 trials:
    if key == "data":
        data = mdata["data"][0][0][0]
        timepoints = mdata["time"][0][0][0]  # time
        fs = mdata["SRATE"][0][0][0][0]  # srate
        return data, timepoints, fs

    # chaninfo = defaultdict(list)
    # chaninfo_keys = mdata["chaninfo"][0][0][0].dtype.names
    # Extracting channel names and mnt info, excluding the reference electrode (REF)
    elif key == "clab":
        chan_inf = mdata["chaninfo"][0][0][0]["electrodes"][0][0]
        clab = []
        mnt = []
        for element in chan_inf[:-1]:
            clab.append(str(element[0][0]))
            mnt.append([element[1][0][0], element[2][0][0], element[3][0][0]])
        mnt = np.array(mnt)
        clab = np.array(clab)
        return mnt, clab

    elif key == "trial_info":
        trial_info = defaultdict(list)
        trial_info_keys = mdata["TrialData"][0][0][0].dtype.names
        for trial in mdata["TrialData"][0][0][0]:
            for i, key in enumerate(trial_info_keys):
                trial_info[key].append(trial[i][0][0])
        return trial_info

    elif key == "metadata":
        # trial artifact from the original data set, if bandpass filtered data from any electrode crosses threshold of +-100 μV = True, else False
        metadata = dict()
        metadata_keys = mdata["metadata"][0][0][0].dtype.names

        for s in mdata["metadata"][0][0][0]:
            for i, key in enumerate(metadata_keys):
                metadata[key] = s[i][0][0]
        return metadata
    else:
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


def load_all_eeg_data(data_path, outdir=None):
    group_files = list_all_files(data_path)
    res = {}
    for subject, sessions in group_files.items():
        if subject not in res:
            res[subject] = {}
            print(f"Subject {subject} : \n")
            for i, (session_id, session_path) in enumerate(sessions.items()):
                if session_id not in res[subject]:
                    res[subject][session_id] = {}
                    try:
                        data, timepoints, fs = load_data_from_mat(session_path, key="data")
                    except:
                        res[subject][session_id]["loaded"] = "FAILED"

                    print(
                        f"Loading of {session_id} from {str(i + 1)}/{len(sessions.keys())} sessions",
                        end="\n")

                    res[subject][session_id]["data"] = data

                    if outdir is not None:
                        if not os.path.exists(outdir):
                            os.makedirs(outdir)
                        outfile_name = os.path.join(outdir, f"{subject}_{session_id}.npz")
                        np.savez_compressed(outfile_name, data=data)

    return res


def load_all_mat_sessions(data_path, outdir=None):
    group_files = list_all_files(data_path)

    res = {}
    for subject, sessions in group_files.items():
        if subject not in res:
            res[subject] = {}
            for i, (session_id, session_path) in enumerate(sessions.items()):
                if session_id not in res[subject]:
                    res[subject][session_id] = {}
                    try:
                        data, timepoints, fs, clab, mnt, trial_info, metadata = load_data_from_mat(session_path, key="")
                    except:
                        res[subject][session_id]["loaded"] = "FAILED"

                    print(
                        f"Subject {subject} : Loading {session_id} from {str(i + 1)}/{len(sessions.keys())} sessions ",
                        end="\r")

                    res[subject][session_id]["data"] = data
                    res[subject][session_id]["timepoints"] = timepoints
                    res[subject][session_id]["fs"] = fs
                    res[subject][session_id]["clab"] = clab
                    res[subject][session_id]["mnt"] = mnt
                    res[subject][session_id]["trial_info"] = trial_info
                    res[subject][session_id]["metadata"] = metadata

                    if outdir is not None:
                        if not os.path.exists(outdir):
                            os.makedirs(outdir)
                        outfile_name = os.path.join(outdir, f"{subject}_{session_id}_all.mat")
                        scipy.io.savemat(outfile_name,
                                         {"data": data, "timepoints": timepoints, "fs": fs, "lab": clab, "mnt": mnt,
                                          "trial_info": trial_info, "metadata": metadata})

    return res


def load_single_mat_session(file_path):

    if os.path.exists(file_path):
        data, timepoints, fs, clab, mnt, trial_info, metadata = load_data_from_mat(file_path)
        return data, timepoints, fs, clab, mnt, trial_info, metadata

    else:
        raise FileNotFoundError(f"{file_path} file does not exist !")


def load_subjects_info(data_path, outdir=None):
    group_files = list_all_files(data_path)
    subjects_stats = {}
    for subject, sessions in group_files.items():
        if subject not in subjects_stats:
            subjects_stats[subject] = {}
            for session_path in sessions.values():
                try:
                    metadata = load_data_from_mat(session_path, key="metadata")
                    if len(metadata) != 0:
                        break
                except:
                    pass
            subjects_stats[subject]["metadata"] = metadata
        if outdir is not None:
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            outfile_name = os.path.join(outdir, f"{subject}.pkl")
            with open(outfile_name, "wb") as f:
                pickle.dump(subjects_stats, f, pickle.HIGHEST_PROTOCOL)
    return subjects_stats


def load_subjects_trials_stats(data_path, outdir=None):
    group_files = list_all_files(data_path)
    res = {}
    for subject, sessions in group_files.items():
        if subject not in res:
            res[subject] = {}
            for i, (session_id, session_path) in enumerate(sessions.items()):
                if session_id not in res[subject]:
                    res[subject][session_id] = {}
                    try:
                        trial_info = load_data_from_mat(session_path, key="trial_info")
                    except:
                        res[subject][session_id]["loaded"] = "FAILED"

                    res[subject][session_id]["result"] = np.array(trial_info["result"])
                    res[subject][session_id]["forcedresult"] = np.array(trial_info["forcedresult"])
                    res[subject][session_id]["triallength"] = np.array(trial_info["triallength"])

                if outdir is not None:
                    if not os.path.exists(outdir):
                        os.makedirs(outdir)
                    outfile_name = os.path.join(outdir, f"{subject}_{session_id}.pkl")
                    with open(outfile_name, "wb") as f:
                        pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
    return res
