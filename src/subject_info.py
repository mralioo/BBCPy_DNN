import argparse
import json
import os

import numpy as np

import bbcpy


def get_subject_info(subject_name, srm_data_path):
    task_name_dict = {"LR": 1.0, "UD": 2.0, "2D": 3.0}
    subject_dict = {}

    sessions_group_path = bbcpy.load.srm_eeg.list_all_files(srm_data_path,
                                                            pattern=f"{subject_name}_*.mat")[subject_name]

    for session_name in sessions_group_path.keys():
        _, _, _, _, _, trials_info, subject_info = \
            bbcpy.load.srm_eeg.load_single_mat_session(file_path=sessions_group_path[session_name])

        subject_dict[session_name] = {}
        subject_dict[session_name]["subject"] = subject_info
        subject_dict[session_name]["tasknumber"] = trials_info["tasknumber"]
        subject_dict[session_name]["resultind"] = trials_info["resultind"]
        subject_dict[session_name]["triallength"] = trials_info["triallength"]
        subject_dict[session_name]["targetnumber"] = trials_info["targetnumber"]
        subject_dict[session_name]["targethitnumber"] = trials_info["targethitnumber"]
        subject_dict[session_name]["result"] = trials_info["result"]
        subject_dict[session_name]["forcedresult"] = trials_info["forcedresult"]

        subject_dict[session_name]["valid_pvc"] = {}
        subject_dict[session_name]["forced_pvc"] = {}
        subject_dict[session_name]["all_pvc"] = {}

        for task_name, task_value in task_name_dict.items():
            task_ids = [id for id, i in enumerate(trials_info["tasknumber"]) if i == task_name_dict[task_name]]

            subject_dict[session_name]["valid_pvc"][task_name] = (
                    np.sum(np.array(trials_info["result"])[task_ids] == True) /
                    (np.sum(np.array(trials_info["result"])[task_ids] == True) + np.sum(
                        np.array(trials_info["result"])[task_ids] == False)))

            subject_dict[session_name]["forced_pvc"][task_name] = (
                    np.sum(np.array(trials_info["forcedresult"])[task_ids] == True) /
                    (np.sum(np.array(trials_info["forcedresult"])[task_ids] == True) + np.sum(
                        np.array(trials_info["forcedresult"])[task_ids] == False)))

            subject_dict[session_name]["all_pvc"][task_name] = (np.sum(
                np.array(trials_info["result"])[task_ids] == True)) / 150

    local_path = os.path.dirname(os.path.abspath(__file__))
    print(local_path)

    metadata_path = os.path.join(local_path, "metadata")

    if not os.path.exists(os.path.dirname(metadata_path)):
        os.makedirs(os.path.dirname(metadata_path))

    file_name = f"{subject_name}.json"

    # Convert numpy bools to Python bools
    def convert_bool(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, list):
            return [convert_bool(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_bool(value) for key, value in obj.items()}
        else:
            return obj

    # Apply the conversion
    data = convert_bool(subject_dict)

    with open(os.path.join(metadata_path, file_name), "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get subject info.')
    parser.add_argument('--subject_name', type=str, default="S1", help='Subject name')
    parser.add_argument('--srm_data_path', type=str, default="/input-data", help='Path to SRM data')
    args = parser.parse_args()

    get_subject_info(args.subject_name, args.srm_data_path, args.verbose)
