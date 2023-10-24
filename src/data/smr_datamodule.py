import gc
import logging

import numpy as np
import numpy.ma as ma
import pyrootutils
from omegaconf import OmegaConf

import bbcpy

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.srm_utils import transform_electrodes_configurations, remove_reference_channel, calculate_pvc_metrics, \
    normalize

logging.getLogger().setLevel(logging.INFO)


class SMR_Data():

    def __init__(self,
                 data_dir,
                 task_name,
                 trial_type,
                 subject_sessions_dict,
                 loading_data_mode,
                 ival,
                 bands,
                 chans,
                 fallback_neighbors,
                 transform,
                 normalize,
                 process_noisy_channels,
                 ignore_noisy_sessions,
                 ):

        """ Initialize the SMR datamodule
        """
        # FIXME : parameter are in type  omegaconf

        self.srm_data_path = data_dir
        self.task_name = task_name

        if self.task_name == "LR":
            self.classes = ["R", "L"]
        elif self.task_name == "UD":
            self.classes = ["U", "D"]
        elif self.task_name == "2D":
            self.classes = ["R", "L", "U", "D"]

        self.trial_type = trial_type

        self.subject_sessions_dict = subject_sessions_dict
        self.loaded_subjects_sessions = {}
        self.founded_subjects_sessions = {}

        self.loading_data_mode = loading_data_mode

        if isinstance(chans, str):
            self.select_chans = [chans]
        else:
            self.select_chans = OmegaConf.to_container(chans)

        self.select_timepoints = ival

        if isinstance(bands, list):
            self.bands = bands
        else:
            self.bands = OmegaConf.to_container(bands)

        self.fallback_neighbors = fallback_neighbors

        self.transform = transform

        if normalize is not None and not isinstance(normalize, dict):
            self.normalize = OmegaConf.to_container(normalize)
        elif normalize is not None and isinstance(normalize, dict):
            self.normalize = normalize
        else:
            self.normalize = None

        self.subjects_info_dict = {}

        self.process_noisy_channels = process_noisy_channels
        self.ignore_noisy_sessions = ignore_noisy_sessions

        # FIXME : walkaround to fix trial length
        self.trial_maxlen = 12000
        self.trial_len = []

    @property
    def num_classes(self):
        return len(self.classes)

    def collect_subject_sessions(self, subjects_dict):
        """ Collect all the sessions for the subjects in the list """

        # if not isinstance(subjects_dict, dict) or not isinstance(subjects_dict, omegaconf.dictconfig.DictConfig):
        #     raise Exception(f"subjects_dict must be a dictionary with S*:[1,2,3,4]")

        subjects_sessions_path_dict = {}
        for subject_name, sessions_ids in subjects_dict.items():
            sessions_group_path = bbcpy.load.srm_eeg.list_all_files(self.srm_data_path,
                                                                    pattern=f"{subject_name}_*.mat")[subject_name]

            self.founded_subjects_sessions[subject_name] = list(sessions_group_path.keys())

            if sessions_ids == "all":
                # select all sessions
                logging.info(f"Found sessions: {list(sessions_group_path.keys())} for subject {subject_name}")

                subjects_sessions_path_dict[subject_name] = sessions_group_path
            else:
                # select specific sessions
                if isinstance(sessions_ids, int):
                    sessions_names = ["Session_" + str(sessions_ids)]
                else:
                    sessions_names = ["Session_" + str(session_id) for session_id in sessions_ids]

                subjects_sessions_path_dict[subject_name] = {}
                for session_name in sessions_names:
                    if session_name not in sessions_group_path.keys():
                        raise Exception(f" {session_name} not found for subject {subject_name}")
                    else:
                        subjects_sessions_path_dict[subject_name][session_name] = sessions_group_path[session_name]

        return subjects_sessions_path_dict

    def preprocess_data(self, srm_obj):
        """ Reshape the SRM data object to the desired shape """

        if isinstance(srm_obj.chans, bbcpy.datatypes.eeg.ChanUnion):
            # for multiple channels
            if (self.classes is not None) and (self.select_chans is not None) and (self.select_timepoints is not None):
                if isinstance(self.classes, str):
                    self.classes = [self.classes]
                if self.select_timepoints == "all":
                    obj = srm_obj[self.classes][:, :, :].copy()
                else:
                    obj = srm_obj[self.classes][:, :, self.select_timepoints].copy()
            else:
                obj = srm_obj

        else:

            if (self.classes is not None) and (self.select_chans is not None) and (self.select_timepoints is not None):
                if isinstance(self.classes, str):
                    self.classes = [self.classes]
                if self.select_timepoints == "all":
                    obj = srm_obj[self.classes][:, self.select_chans, :].copy()
                else:
                    obj = srm_obj[self.classes][:, self.select_chans, self.select_timepoints].copy()
            else:
                obj = srm_obj

        return obj

    def referencing(self, srm_obj, noisy_channel_index, noisy_channels_indices, mode="average"):
        """Apply Laplacian referencing to a noisy channel.

        Parameters
        ----------
        srm_obj: bbcpy.datatypes.srm_eeg.SRM_Data
            SRM data object
        noisy_channel_index: int
            Index of the noisy channel currently being processed
        noisy_channels_indices: list
            List of indices of all noisy channels

        Returns
        -------
        srm_obj: bbcpy.datatypes.srm_eeg.SRM_Data
            SRM data object with the Laplacian referenced noisy channel
        chan_dict_info: dict
            Information about the neighbors used for Laplacian referencing
        """

        # Calculate distances from the current noisy channel to all other channels
        distances = np.linalg.norm(srm_obj.chans.mnt - srm_obj.chans.mnt[noisy_channel_index], axis=1)

        # Set the distances of all noisy channels to a high value so they are not chosen as neighbors
        for idx in noisy_channels_indices:
            distances[idx] = np.inf

        # Identify neighboring channels
        neighbors = np.argsort(distances)[1:self.fallback_neighbors + 1]  # Excluding the channel itself
        # Calculate the average signal of neighboring channels
        average_signal = np.mean(srm_obj[:, neighbors, :], axis=1)

        if mode == "average":
            # Replace the noisy channel's signal with the average signal
            srm_obj[:, noisy_channel_index, :] = np.expand_dims(average_signal, axis=1)
        elif mode == "laplacian":
            # Compute Laplacian: subtract the average of neighbors from the channel
            laplacian_signal = srm_obj[:, noisy_channel_index, :] - np.expand_dims(average_signal, axis=1)
            # Replace the noisy channel's signal with the Laplacian signal
            srm_obj[:, noisy_channel_index, :] = laplacian_signal

        # Save the neighbors info
        chan_dict_info = {"neighbors": srm_obj.chans[neighbors]}

        return srm_obj, chan_dict_info

    def load_session_runs(self, session_path):
        number_of_runs = 6
        task_name_dict = {"LR": 1.0, "UD": 2.0, "2D": 3.0}
        target_map_dict = {1: "R", 2: "L", 3: "U", 4: "D"}

        srm_data, timepoints, srm_fs, clab, mnt, trials_info, subject_info = \
            bbcpy.load.srm_eeg.load_single_mat_session(file_path=session_path)
        session_info_dict = {}
        session_info_dict["subject_info"] = subject_info
        # calculate pvc
        session_info_dict["pvc"] = calculate_pvc_metrics(trials_info, taskname=self.task_name)

        # set the EEG channels object, and remove the reference channel if exists
        chans = remove_reference_channel(clab, mnt)
        # Create SRM object for the given task
        task_trials_ids = [id for id, i in enumerate(trials_info["tasknumber"]) if i == task_name_dict[self.task_name]]

        raw_data = srm_data[task_trials_ids]
        time = np.arange(0, self.trial_maxlen)
        # Get the trials targets for each runs , we have 6 runs
        task_targets = np.array(trials_info["targetnumber"])[task_trials_ids]
        task_targets = task_targets.astype(int) - 1  # to start from 0

        new_srm_train_data = ma.zeros((len(raw_data), len(chans), self.trial_maxlen))
        # Set the mask for each row based on the sub ndarray size
        for i, sub_arr in enumerate(raw_data):
            new_srm_train_data[i, :, :sub_arr.shape[-1]] = sub_arr

        # FIXME : walkaround to fix trial length
        mrk = bbcpy.datatypes.srm_eeg.SRM_Marker(mrk_pos=task_trials_ids,
                                                 mrk_class=task_targets,
                                                 mrk_class_name=self.classes,
                                                 mrk_fs=1,
                                                 parent_fs=srm_fs)

        epo_data = bbcpy.datatypes.srm_eeg.SRM_Data(srm_data=new_srm_train_data,
                                                    timepoints=time,
                                                    fs=srm_fs,
                                                    mrk=mrk,
                                                    chans=chans)

        # if noisy channels are present, remove them from the data
        if subject_info["noisechan"] is not None:
            # shift to left becasue of REF. channel removal
            noisy_chans_id_list = [int(chan) - 1 for chan in subject_info["noisechan"]]
            if self.process_noisy_channels:
                logging.info(f"Noisy channels found: {subject_info['noisechan']}, "
                             f"each channel will be averaged with {self.fallback_neighbors} neighbors")
                session_info_dict["noisechans"] = {}
                for noisy_chans_id in noisy_chans_id_list:
                    epo_data, chan_dict_info = self.referencing(epo_data.copy(),
                                                                noisy_chans_id,
                                                                noisy_chans_id_list,

                                                                mode="average")

                    noise_chans_name = str(epo_data.chans[noisy_chans_id])
                    session_info_dict["noisechans"][noise_chans_name] = chan_dict_info
            else:
                session_info_dict["noisechans"] = noisy_chans_id_list
        else:
            session_info_dict["noisechans"] = None

        # channels configurations transformation for TSCeption model
        if self.transform == "TSCeption":
            epo_data = transform_electrodes_configurations(epo_data.copy())

        # FIXME bandpass filter the data bte 8-30 Hz in liter
        if self.bands is not None:
            logging.info(f"Bandpass filtering the data between {self.bands[0]}-{self.bands[1]} Hz")
            epo_data = epo_data.lfilter(self.bands)

        # preprocess the data
        logging.info(f"Preprocessing the data ...")
        logging.info(f"Data shape before preprocessing: {epo_data.shape}")
        epo_data = epo_data[:, self.select_chans, self.select_timepoints].copy()
        logging.info(f"Data shape after preprocessing: {epo_data.shape}")

        # divide the trials into 6 runs and get the trials ids for each run
        task_trials_ids_runs = np.array_split(task_trials_ids, 6)
        trial_ids_runs = np.array_split(np.arange(len(task_trials_ids)), 6)
        runs_data = {}
        for i, (task_idx, trial_idx) in enumerate(zip(task_trials_ids_runs, trial_ids_runs)):
            run_name = f"run_{i + 1}"
            runs_data[run_name] = {}
            if self.trial_type == "valid":
                task_results = np.array(trials_info["result"])[task_idx]
                # Convert NaN to False
                task_results[np.isnan(task_results)] = 0
                # Convert to boolean
                task_results = task_results.astype(bool)
            elif self.trial_type == "forced":
                task_results = np.array(trials_info["forcedresult"])[task_idx]
            else:
                raise ValueError("trial_type should be either valid or forced")

            # all trials
            runs_data[run_name]["ids"] = []
            runs_data[run_name]["tidx"] = []
            for idx, tidx, valid in zip(task_idx, trial_idx, task_results):
                if valid:
                    runs_data[run_name]["ids"].append(idx)
                    runs_data[run_name]["tidx"].append(tidx)

        for run_name, run_data in runs_data.items():
            runs_data[run_name]["data"] = epo_data[run_data["tidx"], :, :].copy()
            print(run_name, runs_data[run_name]["data"].shape)

        return runs_data, session_info_dict

    def load_sessions_runs(self, subject_path_dict):
        """ Create a subject object from the SRM data sessions"""
        # FIXME : not completed

        loaded_subject_sessions = {}
        subject_info = {"sessions_info": {}}
        for i, (session_name, session_path) in enumerate(subject_path_dict.items()):
            # load data
            logging.info(f"Loading session {session_name} ...")
            run_data_dict, session_info_dict = self.load_session_runs(session_path)

            logging.info(f"{i + 1}/{len(subject_path_dict)} sessions loaded")

            # add to the loaded sessions dict
            loaded_subject_sessions[session_name] = run_data_dict

            # add to the loaded sessions dict
            subject_info["sessions_info"][session_name] = {}
            subject_info["sessions_info"][session_name]["pvc"] = session_info_dict["pvc"]
            subject_info["sessions_info"][session_name]["NoisyChannels"] = session_info_dict["noisechans"]
            subject_info["subject_info"] = session_info_dict["subject_info"]

        return loaded_subject_sessions, subject_info

    def append_sessions_across_runs(self, sessions_data_dict, sessions_info_dict):
        """ Append all the subjects sessions , this includes train and test trials"""

        # FIXME : not completed: check how many noisy channels has a sessions

        run_1 = None
        run_2 = None
        run_3 = None
        run_4 = None
        run_5 = None
        run_6 = None

        for session_name, session_data in sessions_data_dict.items():

            # FIXME : preprocessing here if needed
            # check if the session has noisy channels
            if self.ignore_noisy_sessions and sessions_info_dict[session_name]["NoisyChannels"] is not None:
                logging.info(f"Session {session_name} has noisy channels, the session will be skipped")
                continue
            else:
                if run_1 is None and run_2 is None and run_3 is None and run_4 is None and run_5 is None and run_6 is None:
                    run_1 = session_data["run_1"]["data"]
                    run_2 = session_data["run_2"]["data"]
                    run_3 = session_data["run_3"]["data"]
                    run_4 = session_data["run_4"]["data"]
                    run_5 = session_data["run_5"]["data"]
                    run_6 = session_data["run_6"]["data"]
                else:
                    run_1 = run_1.append(session_data["run_1"]["data"], axis=0)
                    run_2 = run_2.append(session_data["run_2"]["data"], axis=0)
                    run_3 = run_3.append(session_data["run_3"]["data"], axis=0)
                    run_4 = run_4.append(session_data["run_4"]["data"], axis=0)
                    run_5 = run_5.append(session_data["run_5"]["data"], axis=0)
                    run_6 = run_6.append(session_data["run_6"]["data"], axis=0)

        logging.info(f"Run 1 shape: {run_1.shape}")
        logging.info(f"Run 2 shape: {run_2.shape}")
        logging.info(f"Run 3 shape: {run_3.shape}")
        logging.info(f"Run 4 shape: {run_4.shape}")
        logging.info(f"Run 5 shape: {run_5.shape}")
        logging.info(f"Run 6 shape: {run_6.shape}")

        return [run_1, run_2, run_3, run_4, run_5, run_6]

    def load_subjects_sessions(self, subject_sessions_path_dict):
        """ Load all the subjects sessions and concatenate them """
        subject_data_dict = {}
        subjects_info_dict = {}

        for subject_name, subject_path_dict in subject_sessions_path_dict.items():

            subject_data_dict[subject_name] = {}
            subjects_info_dict[subject_name] = {}

            logging.info(f"Subject {subject_name} loading...")

            subject_data_dict[subject_name], subject_info = \
                self.load_sessions_runs(subject_path_dict)

            # calculate the mean pvc for all the sessions
            pvc_list = []
            # ratio of noisy sessions are there with total number of sessions
            noisy_sessions = 0
            for session_name, session_info_dict in subject_info["sessions_info"].items():
                pvc_list.append(session_info_dict["pvc"])
                if session_info_dict["NoisyChannels"] is not None:
                    noisy_sessions += 1

            subjects_info_dict[subject_name]["pvc"] = np.mean(pvc_list)
            subjects_info_dict[subject_name]["ratio_noisy_sessions"] = noisy_sessions / len(
                subject_info["sessions_info"])
            subjects_info_dict[subject_name].update(subject_info)

        return subject_data_dict, subjects_info_dict

    def append_subjects(self, subjects_sessions_path_dict):
        """ Append all the subjects sessions """

        subjects_data_dict, subjects_info_dict = self.load_subjects_sessions(subjects_sessions_path_dict)

        runs_trials_list = [[] for _ in range(6)]
        runs_res_list = []
        for subject_name, subject_data in subjects_data_dict.items():
            loaded_subject_sessions_info = subjects_info_dict[subject_name]["sessions_info"]
            # append the sessions
            run_list = self.append_sessions_across_runs(subject_data,
                                                        loaded_subject_sessions_info)

            for i, run_data in enumerate(run_list):
                if run_data is not None:
                    runs_trials_list[i].append(run_data)

        for i, run_data in enumerate(runs_trials_list):

            run_obj = run_data[0]
            for run_obj_i in run_data[1:]:
                run_obj = run_obj.append(run_obj_i, axis=0)

            runs_res_list.append(run_obj)

        return runs_res_list, subjects_info_dict

    def prepare_dataloader(self):
        """ Prepare the data for the datamodule """

        # load subject paths
        subjects_sessions_path_dict = self.collect_subject_sessions(self.subject_sessions_dict)

        if self.loading_data_mode == "within_subject":
            # load the subject sessions dict
            subject_data_dict, subjects_info_dict = self.load_subjects_sessions(subjects_sessions_path_dict)
            subject_name = list(subject_data_dict.keys())[0]

            # append the sessions
            self.runs_data_list = self.append_sessions_across_runs(subject_data_dict[subject_name],
                                                                   subjects_info_dict[subject_name]["sessions_info"])

            logging.info("Preparing data...")
            # FIXME : take portion from run 3 and run 6 for test
            self.test_data = self.runs_data_list[-1]
            self.train_data_list = self.runs_data_list[0:-1]

            # subject info dict
            self.subjects_info_dict[subject_name] = subjects_info_dict[subject_name]

            # delete the subject data dict
            del subject_data_dict, subjects_info_dict
            gc.collect()


if "__main__" == __name__:
    # Using a raw string
    from pathlib import Path

    data_dir = Path("D:\\SMR\\")
    task_name = "LR"
    subject_sessions_dict = {"S4": "all"}
    loading_data_mode = "within_subject"
    ival = "2s:10s:10ms"
    bands = [8, 13]
    chans = "*"
    fallback_neighbors = 4
    transform = None
    normalize_dict = {"norm_type": "std", "norm_axis": 0}

    smr_datamodule = SMR_Data(data_dir=data_dir,
                              task_name=task_name,
                              subject_sessions_dict=subject_sessions_dict,
                              loading_data_mode=loading_data_mode,
                              ival=ival,
                              bands=bands,
                              chans=chans,
                              fallback_neighbors=fallback_neighbors,
                              transform=transform,
                              normalize=normalize_dict,
                              process_noisy_channels=False)

    subjects_sessions_path_dict = smr_datamodule.collect_subject_sessions(subject_sessions_dict)
    subject_data_dict, subjects_info_dict = smr_datamodule.load_subjects_sessions(subjects_sessions_path_dict)

    subject_name = list(subject_data_dict.keys())[0]
    loaded_subject_sessions = subject_data_dict[subject_name]
    loaded_subject_sessions_info = subjects_info_dict[subject_name]["sessions_info"]

    # append the sessions (FIXME : forced trials are not used)
    valid_trials = smr_datamodule.append_sessions(loaded_subject_sessions,
                                                  loaded_subject_sessions_info)

    from bbcpy.visual.scalp import map

    map(valid_trials, valid_trials.chans)

    # ival = [[160, 200], [230, 260], [300, 320], [380, 430]]
    #
    # rows = len(mrk_classname)
    # cols = len(ival)
    #
    # plt.figure(figsize=(18, 7))
    # for [klass_idx, klass_name] in enumerate(mrk_classname):
    #     for [interval_idx, [start, end]] in enumerate(ival):
    #         plot_idx = klass_idx * cols + interval_idx + 1
    #         plt.subplot(height, width, plot_idx)
    #
    #         indices = (epo_t >= start) & (epo_t <= end)
    #         mean = np.mean(epo[indices, :, :][:, :, mrk_class == klass_idx], axis=(0, 2))
    #         bci.scalpmap(mnt, mean)
