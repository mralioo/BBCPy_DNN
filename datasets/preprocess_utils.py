import pickle

import numpy as np

from datasets.cont_smr import list_all_files


def load_dict_pkl(path_name):
    res_dict = {}
    group_files = list_all_files(path_name, pattern="*.pkl")
    for _, subject_value in group_files.items():
        if isinstance(subject_value, dict):
            for _, session_path in subject_value.items():
                with open(session_path, "rb") as f:
                    res_dict.update(pickle.load(f))
        else:
            with open(subject_value, "rb") as f:
                res_dict.update(pickle.load(f))
    return res_dict


def get_trial_stats(trial_info_dict):
    trial_stat_dict = {}
    for subject, sessions in trial_info_dict.items():
        trial_stat_dict[subject] = {}
        for session_id, values in sessions.items():
            trial_stat_dict[subject][session_id] = {}

            true_trials_idx = np.where(np.array(values["result"]) == True)[0]
            false_trials_idx = np.where(np.array(values["result"]) == False)[0]
            error_trials_idx = np.where(np.isnan(np.array(values["result"])))[0]

            trial_stat_dict[subject][session_id]["true"] = len(true_trials_idx)
            trial_stat_dict[subject][session_id]["false"] = len(false_trials_idx)
            trial_stat_dict[subject][session_id]["error"] = len(error_trials_idx)
            trial_stat_dict[subject][session_id]["mean_triallength"] = np.mean(np.array(values["triallength"]))
            trial_stat_dict[subject][session_id]["std_triallength"] = np.std(np.array(values["triallength"]))

    return trial_stat_dict


def get_subjects_stats(subjects_info_dict, filters):
    if filters == "gender":
        res_dict = {"M": 0, "F": 0}
        for subject, metadata in subjects_info_dict.items():
            res_dict[metadata["metadata"]["gender"]] += 1
    elif filters == "handedness":
        res_dict = {"R": 0, "L": 0}
        for subject, metadata in subjects_info_dict.items():
            res_dict[metadata["metadata"]["handedness"]] += 1
    elif filters == "handsport":
        res_dict = {"Y": 0, "N": 0}
        for subject, metadata in subjects_info_dict.items():
            res_dict[metadata["metadata"]["handsport"]] += 1
    elif filters == "MBSRsubject":
        res_dict = {True: 0, False: 0}
        for subject, metadata in subjects_info_dict.items():
            res_dict[metadata["metadata"]["MBSRsubject"]] += 1
    elif filters == "age":
        res_dict = {"Total": [], "F": [], "M": []}
        for subject, metadata in subjects_info_dict.items():
            res_dict["Total"].append(metadata["metadata"]["age"])
            if metadata["metadata"]["gender"] == "F":
                res_dict["F"].append(metadata["metadata"]["age"])
            else:
                res_dict["M"].append(metadata["metadata"]["age"])
    elif filters == "meditationpractice":
        res_dict = {"Total": [], "F": [], "M": []}
        for subject, metadata in subjects_info_dict.items():
            res_dict["Total"].append(metadata["metadata"]["meditationpractice"])
            if metadata["metadata"]["gender"] == "F":
                res_dict["F"].append(metadata["metadata"]["meditationpractice"])
            else:
                res_dict["M"].append(metadata["metadata"]["meditationpractice"])

    return res_dict
