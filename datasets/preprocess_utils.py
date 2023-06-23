import pickle

import numpy as np

from datasets.utils import list_all_files
from bbcpy.load.srm_eeg import *
from bbcpy.datatypes.utils import convert_numpy_types
import json
import logging
logging.getLogger().setLevel(logging.INFO)

def generate_session_metadata(srm_data_path):
    """ Create a metadata file for the session based on the trial_info.
    :param session_id:
    :type session_id:
    :param trial_info:
    :type trial_info:
    :return:
    :rtype:
    """
    target_map_dict = {1: "R", 2: "L", 3: "U", 4: "D", 'nan': "NT"}
    results_map_dict = {0: "failed", 1: "succeed", 'nan': "NT"}

    subject_group_path = list_all_files(srm_data_path, pattern="*.mat")

    for subject_id, subject_dict in subject_group_path.items():
        for session_id, session_path in subject_dict.items():
            # Load the data
            logging.info(
                f"Loading data from {session_path} for subject {subject_id} and session {session_id}")
            _, _, _, _, _, trial_info, subject_info = load_single_mat_session(session_path)

            results_trials = np.array(trial_info["result"])
            fresults_trials = np.array(trial_info["forcedresult"])
            task_trials = np.array(trial_info["targetnumber"])
            targethit_task_trials = np.array(trial_info["targethitnumber"])
            artifact_trials = np.array(trial_info["artifact"])
            length_trials = np.array(trial_info["triallength"])

            session_info_dict = {}

            session_info_dict[subject_id] = subject_info

            session_info_dict["result"] = {}
            unique_values, counts = np.unique(results_trials, return_counts=True)
            for res_int, count in zip(unique_values, counts):
                if np.isnan(res_int):
                    res_int = str(res_int)
                res_key = results_map_dict[res_int]
                session_info_dict["result"][res_key] = int(count)

            session_info_dict["targetnumber"] = {}
            unique_values, counts = np.unique(task_trials, return_counts=True)
            for task_int, count in zip(unique_values, counts):
                task_key = target_map_dict[task_int]
                session_info_dict["targetnumber"][task_key] = int(count)

            session_info_dict["targethitnumber"] = {}
            unique_values, counts = np.unique(targethit_task_trials, return_counts=True)
            for task_int, count in zip(unique_values, counts):
                if np.isnan(task_int):
                    task_int = str(task_int)
                task_key = target_map_dict[task_int]
                session_info_dict["targethitnumber"][task_key] = int(count)

            session_info_dict["triallength"] = {}
            session_info_dict["triallength"]["min"] = float(np.min(length_trials))
            session_info_dict["triallength"]["max"] = float(np.max(length_trials))
            session_info_dict["triallength"]["avg"] = float(np.average(length_trials))

            session_info_dict["artifact"] = {}
            unique_values, counts = np.unique(artifact_trials, return_counts=True)
            for trial_status, count in zip(unique_values, counts):
                if trial_status == True:
                    session_info_dict["artifact"]["yes"] = int(count)
                else:
                    session_info_dict["artifact"]["no"] = int(count)

            # Save the data to a JSON file
            metadata_file_name = f"{subject_id}_{session_id}.json"

            srm_metadata_path = os.path.join(Path(srm_data_path).parent, "metadata")
            if not os.path.exists(srm_metadata_path):
                os.makedirs(srm_metadata_path)

            metadata_file_path = os.path.join(srm_metadata_path, metadata_file_name)

            with open(metadata_file_path, 'w') as json_file:
                json.dump(session_info_dict, json_file, default=convert_numpy_types, indent=4)
                logging.info(f"Metadata file saved to {metadata_file_path}")


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
