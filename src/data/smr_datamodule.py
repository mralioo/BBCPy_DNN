import gc
import logging

import numpy as np
import numpy.ma as ma
from omegaconf import OmegaConf
from sklearn.model_selection import KFold

import bbcpy
from src.utils.device import print_data_info
from src.utils.srm_utils import transform_electrodes_configurations, remove_reference_channel, calculate_pvc_metrics

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

    def load_session_trials(self, session_path):

        task_name_dict = {"LR": 1.0, "UD": 2.0, "2D": 3.0}
        target_map_dict = {1: "R", 2: "L", 3: "U", 4: "D"}
        # Define the runs for training and testing
        train_runs = [1, 2, 4, 5]
        test_runs = [3, 6]

        srm_data, timepoints, srm_fs, clab, mnt, trials_info, subject_info = \
            bbcpy.load.srm_eeg.load_single_mat_session(file_path=session_path)

        session_info_dict = {}
        session_info_dict["subject_info"] = subject_info
        # calculate pvc
        session_info_dict["pvc"] = calculate_pvc_metrics(trials_info, taskname=self.task_name)

        # set the EEG channels object, and remove the reference channel if exists
        chans = remove_reference_channel(clab, mnt)

        # Initialize empty lists for train and test data
        train_data = []
        test_data = []
        train_labels = []
        test_labels = []
        train_idx = []
        test_idx = []

        # Get the trials ids for the task
        task_trials_ids = [id for id, i in enumerate(trials_info["tasknumber"]) if i == task_name_dict[self.task_name]]

        # Get the trials results for the task
        if self.trial_type == "valid":
            task_results = np.array(trials_info["result"])[task_trials_ids]
            # Convert NaN to False
            task_results[np.isnan(task_results)] = 0
            # Convert to boolean
            task_results = task_results.astype(bool)

        elif self.trial_type == "forced":
            # forced_trials_ids= np.setdiff1d(np.where(np.array(trials_info["forcedresult"])[task_trials_ids] == np.bool_(True))[0],
            #                             np.array(trials_info["result"])[task_trials_ids])

            task_results = np.array(trials_info["forcedresult"])[task_trials_ids]
        else:
            raise ValueError("trial_type should be either valid or forced")

        # Get the trials targets for the task
        task_targets = np.array(trials_info["targetnumber"])[task_trials_ids]
        task_targets = task_targets.astype(int) - 1  # to start from 0
        class_names = np.array(["R", "L", "U", "D"])

        # all trials
        for idx, valid, target in zip(task_trials_ids, task_results, task_targets):
            if valid:
                trial_data = srm_data[idx]
                run_idx = (idx % 150) // 25 + 1  # Calculate the run index
                if run_idx in train_runs:
                    train_data.append(trial_data)
                    train_labels.append(target)
                    train_idx.append(idx)
                elif run_idx in test_runs:
                    test_data.append(trial_data)
                    test_labels.append(target)
                    test_idx.append(idx)

        # FIXME data need to be with even length
        # timepoints_train = timepoints[train_idx]
        # timepoints_test = timepoints[test_idx]
        # trial_maxlen_train = max([len(t[0]) for t in timepoints_train])
        # trial_maxlen_test = max([len(t[0]) for t in timepoints_test])
        # trial_maxlen = max(trial_maxlen_train, trial_maxlen_test)
        # self.trial_len.append(trial_maxlen)
        # FIXME : time with offset is not correct
        time = np.arange(0, self.trial_maxlen)

        # Create SRM_Data objects for the train data
        new_srm_train_data = ma.zeros((len(train_data), len(chans), self.trial_maxlen))
        # Set the mask for each row based on the sub ndarray size
        for i, sub_arr in enumerate(train_data):
            new_srm_train_data[i, :, :sub_arr.shape[-1]] = sub_arr

        mrk_train = bbcpy.datatypes.eeg.Marker(mrk_pos=train_idx,
                                               mrk_class=train_labels,
                                               mrk_class_name=class_names,
                                               mrk_fs=1,
                                               parent_fs=srm_fs)
        epo_train_data = bbcpy.datatypes.srm_eeg.SRM_Data(srm_data=new_srm_train_data,
                                                          timepoints=time,
                                                          fs=srm_fs,
                                                          mrk=mrk_train,
                                                          chans=chans)

        # preprocess the data
        epo_train_data = self.preprocess_data(epo_train_data)

        # Create SRM_Data objects for the test data
        new_srm_test_data = ma.zeros((len(test_data), len(chans), self.trial_maxlen))
        # Set the mask for each row based on the sub ndarray size
        for i, sub_arr in enumerate(test_data):
            new_srm_test_data[i, :, :sub_arr.shape[-1]] = sub_arr

        mrk_test = bbcpy.datatypes.eeg.Marker(mrk_pos=test_idx,
                                              mrk_class=test_labels,
                                              mrk_class_name=class_names,
                                              mrk_fs=1,
                                              parent_fs=srm_fs)

        epo_test_data = bbcpy.datatypes.srm_eeg.SRM_Data(srm_data=new_srm_test_data,
                                                         timepoints=time,
                                                         fs=srm_fs,
                                                         mrk=mrk_test,
                                                         chans=chans)
        # preprocess the data
        epo_test_data = self.preprocess_data(epo_test_data)

        # FIXME bandpass filter the data bte 8-30 Hz in liter
        if self.bands is not None:
            logging.info(f"Bandpass filtering the data between {self.bands[0]}-{self.bands[1]} Hz")
            epo_train_data = epo_train_data.lfilter(self.bands)
            epo_test_data = epo_test_data.lfilter(self.bands)

        # if noisy channels are present, remove them from the data
        if subject_info["noisechan"] is not None:
            # shift to left becasue of REF. channel removal
            noisy_chans_id_list = [int(chan) - 1 for chan in subject_info["noisechan"]]
            if self.process_noisy_channels:
                logging.info(f"Noisy channels found: {subject_info['noisechan']}, "
                             f"each channel will be averaged with {self.fallback_neighbors} neighbors")
                session_info_dict["noisechans"] = {}
                for noisy_chans_id in noisy_chans_id_list:
                    epo_train_data, chan_dict_info = self.referencing(epo_train_data.copy(),
                                                                      noisy_chans_id,
                                                                      noisy_chans_id_list,
                                                                      mode="average")
                    epo_test_data, _ = self.referencing(epo_test_data.copy(),
                                                        noisy_chans_id,
                                                        noisy_chans_id_list,
                                                        mode="average")

                    noise_chans_name = str(epo_train_data.chans[noisy_chans_id])
                    session_info_dict["noisechans"][noise_chans_name] = chan_dict_info
            else:
                session_info_dict["noisechans"] = noisy_chans_id_list

        else:
            session_info_dict["noisechans"] = None

        # channels configurations transformation for TSCeption model
        if self.transform == "TSCeption":
            epo_train_data = transform_electrodes_configurations(epo_train_data.copy())
            epo_test_data = transform_electrodes_configurations(epo_test_data.copy())

        # save subject info shapes
        session_info_dict["shapes"] = {"train": epo_train_data.shape, "test": epo_test_data.shape}

        return epo_train_data, epo_test_data, session_info_dict

    def reshape_raw_srm_data(self, srm_obj):
        pass

    def load_trials(self, session_path, load_forced_trials=False):

        srm_data, timepoints, srm_fs, clab, mnt, trial_info, subject_info = \
            bbcpy.load.srm_eeg.load_single_mat_session(file_path=session_path)

        session_info_dict = {}
        session_info_dict["subject_info"] = subject_info
        # calculate pvc
        session_info_dict["pvc"] = calculate_pvc_metrics(trial_info, taskname=self.task_name)

        # set the EEG channels object, and remove the reference channel if exists
        chans = remove_reference_channel(clab, mnt)

        # true labels
        # target_map_dict = {1: "R", 2: "L", 3: "U", 4: "D"}
        mrk_class = np.array(trial_info["targetnumber"])
        mrk_class = mrk_class.astype(int) - 1  # to start from 0
        class_names = np.array(["R", "L", "U", "D"])

        # all trials
        # FIXME : not sure what to pass for mrk_pos
        # mrk pos here is the results of all trials
        # 1 =correct target selected, 0=incorrect target selected,NaN=no target selected; timeout
        trialresult = trial_info["result"]

        mrk = bbcpy.datatypes.eeg.Marker(mrk_pos=trialresult,
                                         mrk_class=mrk_class,
                                         mrk_class_name=class_names,
                                         mrk_fs=1,
                                         parent_fs=srm_fs)

        # create SRM_Data object for the session
        epo_data = bbcpy.datatypes.srm_eeg.SRM_Data(srm_data=srm_data,
                                                    timepoints=timepoints.reshape(-1, 1),
                                                    fs=srm_fs,
                                                    mrk=mrk,
                                                    chans=chans)
        # print("before bandpass filter")
        # print_data_info(epo_data)

        # FIXME bandpass filter the data bte 8-30 Hz in liter
        if self.bands is not None:
            epo_data = epo_data.lfilter(self.bands)

        # print("after bandpass filter")
        # print_data_info(epo_data)

        # if noisy channels are present, remove them from the data
        if subject_info["noisechan"] is not None:
            # shift to left becasue of REF. channel removal
            noisy_chans_id_list = [int(chan) - 1 for chan in subject_info["noisechan"]]

            if self.process_noisy_channels:
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

        # print("after referencing")
        # print_data_info(epo_data)

        # channels configurations transformation
        if self.transform == "TSCeption":
            epo_data = transform_electrodes_configurations(epo_data.copy())

        # valid trials are defined as the trials where labeles are correct (true) in trialresult

        valid_trials_idx = np.where(epo_data.mrk == np.bool_(True))[0]
        valid_trials = epo_data[valid_trials_idx].copy()

        forced_trials = np.array(None)

        # preprocess the data
        valid_trials = self.preprocess_data(valid_trials)

        if load_forced_trials:
            # forced trials are defined as the trials where labeles are correct (true) in forcedresult
            # and not in valid trials
            forced_trials = trial_info["forcedresult"]

            forced_trials_idx = np.setdiff1d(np.where(forced_trials == np.bool_(True))[0], valid_trials_idx)

            # TODO IMPORTANT repalce y to forced y trials
            forced_trials = epo_data[forced_trials_idx].copy()

            forced_trials = self.preprocess_data(forced_trials)

        # save subject info
        session_info_dict["shapes"] = {"valid_trials": valid_trials.shape,
                                       "forced_trials": forced_trials.shape}

        return valid_trials, forced_trials, session_info_dict

    def load_sessions(self, subject_path_dict):
        """ Create a subject object from the SRM data sessions"""
        # FIXME : not completed

        loaded_subject_sessions = {}
        subject_info = {"sessions_info": {}}
        for i, (session_name, session_path) in enumerate(subject_path_dict.items()):
            # load data
            logging.info(f"Loading session {session_name} ...")
            epo_train_data, epo_test_data, session_info_dict = \
                self.load_session_trials(session_path)

            logging.info(f"Train trials shape: {epo_train_data.shape},"
                         f"Test trials shape: {epo_test_data.shape}")
            logging.info(f"{i + 1}/{len(subject_path_dict)} sessions loaded")

            # add to the loaded sessions dict
            loaded_subject_sessions[session_name] = [epo_train_data, epo_test_data]

            # add to the loaded sessions dict
            subject_info["sessions_info"][session_name] = {}
            subject_info["sessions_info"][session_name]["shapes"] = session_info_dict["shapes"]
            subject_info["sessions_info"][session_name]["pvc"] = session_info_dict["pvc"]
            subject_info["sessions_info"][session_name]["NoisyChannels"] = session_info_dict["noisechans"]
            subject_info["subject_info"] = session_info_dict["subject_info"]

        return loaded_subject_sessions, subject_info

    def append_sessions(self, sessions_data_dict, sessions_info_dict):
        """ Append all the subjects sessions , this includes train and test trials"""

        # FIXME : not completed: check how many noisy channels has a sessions
        train_trials = None
        test_trials = None
        for session_name, session_data in sessions_data_dict.items():
            # FIXME : preprocessing here if needed
            # check if the session has noisy channels
            if self.ignore_noisy_sessions and sessions_info_dict[session_name]["NoisyChannels"] is not None:
                logging.info(f"Session {session_name} has noisy channels, the data will be skipped")
                continue
            else:
                if train_trials is None and test_trials is None:
                    train_trials = session_data[0].copy()
                    test_trials = session_data[1].copy()
                else:
                    train_trials = train_trials.append(session_data[0], axis=0)
                    test_trials = test_trials.append(session_data[1], axis=0)

        return train_trials, test_trials

    def load_subjects_sessions(self, subject_sessions_path_dict):
        """ Load all the subjects sessions and concatenate them """
        subject_data_dict = {}
        subjects_info_dict = {}

        for subject_name, subject_path_dict in subject_sessions_path_dict.items():

            subject_data_dict[subject_name] = {}
            subjects_info_dict[subject_name] = {}

            logging.info(f"Subject {subject_name} loading...")

            subject_data_dict[subject_name], subject_info = \
                self.load_sessions(subject_path_dict)

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

        train_trials_list = []
        test_trials_list = []

        for subject_name, subject_data in subjects_data_dict.items():
            loaded_subject_sessions_info = subjects_info_dict[subject_name]["sessions_info"]
            # append the sessions
            train_trials, test_trials = self.append_sessions(subject_data,
                                                             loaded_subject_sessions_info)

            if train_trials is not None and test_trials is not None:
                train_trials_list.append(train_trials)
                test_trials_list.append(test_trials)

        train_trials = train_trials_list[0]
        test_trials = test_trials_list[0]
        for train_trials_i, test_trials_i in zip(train_trials_list[1:], test_trials_list[1:]):
            train_trials = train_trials.append(train_trials_i, axis=0)
            test_trials = test_trials.append(test_trials_i, axis=0)

        return train_trials, test_trials, subjects_info_dict

    def prepare_dataloader(self):
        """ Prepare the data for the classification """

        # load subject paths
        subjects_sessions_path_dict = self.collect_subject_sessions(self.subject_sessions_dict)

        if self.loading_data_mode == "within_subject":
            # load the subject sessions dict
            subject_data_dict, subjects_info_dict = self.load_subjects_sessions(subjects_sessions_path_dict)

            subject_name = list(subject_data_dict.keys())[0]

            loaded_subject_sessions = subject_data_dict[subject_name]
            loaded_subject_sessions_info = subjects_info_dict[subject_name]["sessions_info"]

            # append the sessions
            self.train_trials, self.test_trials = self.append_sessions(loaded_subject_sessions,
                                                                       loaded_subject_sessions_info)
            logging.info("Raw train trials")
            print_data_info(self.train_trials)
            logging.info("Raw test trials")
            print_data_info(self.test_trials)

            if self.normalize:
                self.train_trials, norm_params_train = normalize(self.train_trials,
                                                                 norm_type=self.normalize["norm_type"],
                                                                 axis=self.normalize["norm_axis"])

                # FIXME : normalize test trials with the same parameters as train trials
                self.test_trials, norm_params_test = normalize(self.test_trials,
                                                               norm_type=self.normalize["norm_type"],
                                                               axis=self.normalize["norm_axis"])

                logging.info("Normalized train trials")
                print_data_info(self.train_trials)
                logging.info("Normalized test trials")
                print_data_info(self.test_trials)

            # subject info dict
            self.subjects_info_dict[subject_name] = subjects_info_dict[subject_name]

            del subject_data_dict, subjects_info_dict
            gc.collect()


        elif self.loading_data_mode == "cross_subject_hpo":

            # append subjects data
            self.train_trials, self.test_trials, self.subjects_info_dict = self.append_subjects(
                subjects_sessions_path_dict)

            logging.info("Raw train trials")
            print_data_info(self.train_trials)
            logging.info("Raw test trials")
            print_data_info(self.test_trials)

            if self.normalize:
                self.train_trials, norm_params_train = normalize(self.train_trials,
                                                                 norm_type=self.normalize["norm_type"],
                                                                 axis=self.normalize["norm_axis"])

                self.test_trials, norm_params_test = normalize(self.test_trials,
                                                               norm_type=self.normalize["norm_type"],
                                                               axis=self.normalize["norm_axis"])

                logging.info("Normalized train trials")
                print_data_info(self.train_trials)
                logging.info("Normalized test trials")
                print_data_info(self.test_trials)

            gc.collect()

        elif self.loading_data_mode == "cross_subject":

            pass  # TODO : not implemented yet

        else:
            raise Exception(f"Loading data mode {self.loading_data_mode} not supported")


def train_valid_split(data, train_val_split_dict):
    """ Split the data into train and validation sets """

    random_seed = train_val_split_dict["random_seed"]
    val_ratio = train_val_split_dict["val_size"]

    # Set the random seed for reproducibility
    np.random.seed(random_seed)

    # Shuffle the data
    # TODO : Shuffle the SMR data object
    indices = np.random.permutation(len(data))
    data = data[indices]
    data.y = data.y[indices]

    # Compute the index where the validation set starts
    val_start_idx = int(len(data) * (1 - val_ratio))

    # Split the data into training and validation sets
    train_data = data[:val_start_idx]
    val_data = data[val_start_idx:]

    return train_data, val_data


def cross_validation(data, cross_validation_dict, kfold_idx):
    kf = KFold(n_splits=cross_validation_dict["num_splits"],
               shuffle=True,
               random_state=cross_validation_dict["split_seed"])

    all_splits_trial_kf = [k for k in kf.split(data)]

    train_indexes, val_indexes = all_splits_trial_kf[kfold_idx]

    train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

    train_data, val_data = (data[train_indexes],
                            data[val_indexes])

    return train_data, val_data


def normalize(data, norm_type="std", axis=None, keepdims=True, eps=10 ** -5, norm_params=None):
    """Normalize data along a given axis.

    Parameters
    ----------
    data : numpy.ndarray
        Data to normalize.
    norm_type : str
        Type of normalization. Can be 'std' or 'minmax'.
    axis : int
        Axis along which to normalize.
    keepdims : bool
        Whether to keep the dimensions of the original data.
    eps : float
        Epsilon to avoid division by zero.
    norm_params : dict
        Dictionary containing normalization parameters. If None, they will be computed.

    Returns
    -------
    data_norm : numpy.ndarray
        Normalized data.
    norm_params : dict
        Dictionary containing normalization parameters.

    """

    if norm_params is not None:
        if norm_params["norm_type"] == "std":
            data_norm = (data - norm_params["mean"]) / norm_params["std"]
        elif norm_params["norm_type"] == "minmax":
            data_norm = (data - norm_params["min"]) / (norm_params["max"] - norm_params["min"])
        else:
            raise RuntimeError("norm type {:} does not exist".format(norm_params["norm_type"]))

    else:
        if norm_type == "std":
            data_std = data.std(axis=axis, keepdims=keepdims)
            data_std[data_std < eps] = eps
            data_mean = data.mean(axis=axis, keepdims=keepdims)
            data_norm = (data - data_mean) / data_std
            norm_params = dict(mean=data_mean, std=data_std, norm_type=norm_type, axis=axis, keepdims=keepdims)
        elif norm_type == "minmax":
            data_min = data.min(axis=axis, keepdims=keepdims)
            data_max = data.max(axis=axis, keepdims=keepdims)
            data_max[data_max == data_min] = data_max[data_max == data_min] + eps
            data_norm = (data - data_min) / (data_max - data_min)
            norm_params = dict(min=data_min, max=data_max, norm_type=norm_type, axis=axis, keepdims=keepdims)
        elif norm_type is None:
            data_norm, norm_params = data, None
        else:
            data_norm, norm_params = None, None
            ValueError("Only 'std' and 'minmax' are supported")

    return data_norm, norm_params


def unnormalize(data_norm, norm_params):
    if norm_params["norm_type"] == "std":
        data = data_norm * norm_params["std"] + norm_params["mean"]
    elif norm_params["norm_type"] == "minmax":
        data = data_norm * (norm_params["max"] - norm_params["min"]) + norm_params["min"]
    return data


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
