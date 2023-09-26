import gc
import logging

import numpy as np
from omegaconf import OmegaConf

import bbcpy

logging.getLogger().setLevel(logging.INFO)


def transform_electrodes_configurations(epo_data):
    # FIXME : not implemented yet

    """
       This function will generate the channel order for TSception
       Parameters
       ----------
       original_order: list of the channel names

       Returns
       -------
       TS: list of channel names which is for TSception
       """

    original_order = epo_data.chans
    chan_name, chan_num, chan_final = [], [], []
    for channel in original_order:
        chan_name_len = len(channel)
        k = 0
        for s in [*channel[:]]:
            if s.isdigit():
                k += 1
        if k != 0:
            chan_name.append(channel[:chan_name_len - k])
            chan_num.append(int(channel[chan_name_len - k:]))
            chan_final.append(channel)
    chan_pair = []
    for ch, id in enumerate(chan_num):
        if id % 2 == 0:
            chan_pair.append(chan_name[ch] + str(id - 1))
        else:
            chan_pair.append(chan_name[ch] + str(id + 1))
    chan_no_duplicate = []
    [chan_no_duplicate.extend([f, chan_pair[i]]) for i, f in enumerate(chan_final) if
     f not in chan_no_duplicate]

    new_order = chan_no_duplicate[0::2] + chan_no_duplicate[1::2]
    # FIXME : not sure if this is the correct way to do it
    return epo_data[:, new_order, :]


def remove_reference_channel(clab, mnt, reference_channel="REF."):
    """ Set the EEG channels object, and remove the reference channel if exists

    Parameters
    ----------
    clab: numpy array
        List of channels names
    mnt: numpy array
        List of channels positions
    reference_channel: str
        Name of the reference channel

    Returns
    -------
    chans: bbcpy.datatypes.eeg.Chans

    """

    if reference_channel in clab:

        ref_idx = np.where(clab == "REF.")[0][0]
        clab = np.delete(clab, ref_idx)
        mnt = np.delete(mnt, ref_idx, axis=0)
        chans = bbcpy.datatypes.eeg.Chans(clab, mnt)

    else:
        logging.warning(f"Reference channel {reference_channel} not found in the data")
        chans = bbcpy.datatypes.eeg.Chans(clab, mnt)

    return chans


def calculate_pvc_metrics(trial_info, taskname="LR"):
    """PVC is metric introduced in the dataset paper and it descibes the percentage of valid correct trials
    formula: PVC = hits / (hits + misses)
    """
    task_num_dict = {"LR": 1.0, "UD": 2.0, "2D": 3.0}
    task_filter_idx = np.where(np.array(trial_info["tasknumber"]) == task_num_dict[taskname])[0]

    # get hits and misses TODO forcedresult
    trials_results = np.array(trial_info["forcedresult"])[task_filter_idx]
    res_dict = {"hits": np.sum(trials_results == True), "misses": np.sum(trials_results == False)}

    # calculate pvc
    pvc = res_dict["hits"] / (res_dict["hits"] + res_dict["misses"])
    return pvc


def print_data_info(srm_obj):
    print("max timepoints: ", np.max(srm_obj))
    print("min timepoints: ", np.min(srm_obj))
    print("mean timepoints: ", np.mean(srm_obj))
    print("std timepoints: ", np.std(srm_obj))


class SMR_Data():

    def __init__(self,
                 data_dir,
                 task_name,
                 subject_sessions_dict,
                 loading_data_mode,
                 ival,
                 bands,
                 chans,
                 fallback_neighbors,
                 transform,
                 normalize,
                 ):

        """ Initialize the SMR datamodule
        """
        # FIXME : parameter are in type  omegaconf

        self.srm_data_path = data_dir
        self.task_name = task_name

        if self.task_name == "RL":
            self.classes = ["R", "L"]
        elif self.task_name == "2D":
            self.classes = ["R", "L", "U", "D"]

        self.subject_sessions_dict = subject_sessions_dict
        self.loaded_subjects_sessions = {}
        self.founded_subjects_sessions = {}

        self.loading_data_mode = loading_data_mode

        self.select_chans = OmegaConf.to_container(chans)
        self.select_timepoints = ival
        self.bands = OmegaConf.to_container(bands)

        self.fallback_neighbors = fallback_neighbors

        self.transform = transform

        if normalize:
            self.normalize = OmegaConf.to_container(normalize)
        else:
            self.normalize = None

        self.subjects_info_dict = {}

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

    def load_forced_valid_trials_data(self, session_path):

        srm_data, timepoints, srm_fs, clab, mnt, trial_info, subject_info = \
            bbcpy.load.srm_eeg.load_single_mat_session(file_path=session_path)

        session_info_dict = {}
        session_info_dict["subject_info"] = subject_info
        # calculate pvc
        session_info_dict["pvc"] = calculate_pvc_metrics(trial_info, taskname=self.task_name)

        # set the EEG channels object, and remove the reference channel if exists
        chans = remove_reference_channel(clab, mnt)

        # true labels
        target_map_dict = {1: "R", 2: "L", 3: "U", 4: "D"}
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
            session_info_dict["noisechans"] = {}
            for noisy_chans_id in noisy_chans_id_list:
                epo_data, chan_dict_info = self.referencing(epo_data.copy(),
                                                            noisy_chans_id,
                                                            noisy_chans_id_list,
                                                            mode="average")

                noise_chans_name = str(epo_data.chans[noisy_chans_id])
                session_info_dict["noisechans"][noise_chans_name] = chan_dict_info
        else:
            session_info_dict["noisechans"] = None

        # print("after referencing")
        # print_data_info(epo_data)

        if self.transform == "TSCeption":
            epo_data = transform_electrodes_configurations(epo_data.copy())

        # valid trials are defined as the trials where labeles are correct (true) in trialresult

        valid_trials_idx = np.where(epo_data.mrk == np.bool_(True))[0]
        valid_trials = epo_data[valid_trials_idx].copy()

        # forced trials are defined as the trials where labeles are correct (true) in forcedresult
        # and not in valid trials
        forced_trials = trial_info["forcedresult"]

        forced_trials_idx = np.setdiff1d(np.where(forced_trials == np.bool_(True))[0], valid_trials_idx)

        # TODO IMPORTANT repalce y to forced y trials
        forced_trials = epo_data[forced_trials_idx].copy()

        # preprocess the data
        valid_trials = self.preprocess_data(valid_trials)
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
            valid_obj_new, forced_obj_new, session_info_dict = \
                self.load_forced_valid_trials_data(session_path)

            logging.info(f"valid trials shape: {valid_obj_new.shape},"
                         f"forced trials shape: {forced_obj_new.shape}")
            logging.info(f"{i + 1}/{len(subject_path_dict)} sessions loaded")

            loaded_subject_sessions[session_name] = [valid_obj_new, forced_obj_new]
            # add to the subject info dict

            # add to the loaded sessions dict
            subject_info["sessions_info"][session_name] = {}
            subject_info["sessions_info"][session_name]["shapes"] = session_info_dict["shapes"]
            subject_info["sessions_info"][session_name]["pvc"] = session_info_dict["pvc"]
            subject_info["sessions_info"][session_name]["NoisyChannels"] = session_info_dict["noisechans"]
            subject_info["subject_info"] = session_info_dict["subject_info"]

        return loaded_subject_sessions, subject_info

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

    def append_sessions(self, sessions_data_dict, sessions_info_dict, ignore_noisy_sessions=False):
        """ Append all the subjects sessions """

        valid_trials = None
        for session_name, session_data in sessions_data_dict.items():
            # check if the session has noisy channels
            if ignore_noisy_sessions and sessions_info_dict[session_name]["NoisyChannels"] is not None:
                logging.info(f"Session {session_name} has noisy channels, the data will be skipped")
                continue
            else:
                if valid_trials is None:
                    valid_trials = session_data[0].copy()
                    # forced_trials = session_data[1].copy()
                else:
                    valid_trials = valid_trials.append(session_data[0], axis=0)
                    # forced_trials = forced_trials.append(session_data[1], axis=0)

        return valid_trials
        # return valid_trials, forced_trials

    def append_subjects(self, subjects_sessions_path_dict):
        """ Append all the subjects sessions """

        subjects_data_dict, subjects_info_dict = self.load_subjects_sessions(subjects_sessions_path_dict)

        valid_trials_list = []

        for subject_name, subject_data in subjects_data_dict.items():
            loaded_subject_sessions_info = subjects_info_dict[subject_name]["sessions_info"]
            # append the sessions
            valid_trials = self.append_sessions(subject_data,
                                                loaded_subject_sessions_info,
                                                ignore_noisy_sessions=True)

            if valid_trials is not None:
                valid_trials_list.append(valid_trials)

        valid_trials = valid_trials_list[0]
        for valid_trials_i in valid_trials_list[1:]:
            valid_trials = valid_trials.append(valid_trials_i, axis=0)

        return valid_trials, subjects_info_dict

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

            # append the sessions (FIXME : forced trials are not used)
            self.valid_trials = self.append_sessions(loaded_subject_sessions,
                                                     loaded_subject_sessions_info)
            print("valid trials")
            print_data_info(self.valid_trials)

            if self.normalize:
                self.valid_trials, norm_params_valid = normalize(self.valid_trials,
                                                                 norm_type=self.normalize["norm_type"],
                                                                 axis=self.normalize["norm_axis"])

                print("Normliazed valid trials")
                print_data_info(self.valid_trials)

            # subject info dict
            self.subjects_info_dict[subject_name] = subjects_info_dict[subject_name]

            del subject_data_dict, subjects_info_dict
            gc.collect()


        elif self.loading_data_mode == "cross_subject_hpo":

            # append subjects data
            self.valid_trials, self.subjects_info_dict = self.append_subjects(subjects_sessions_path_dict)

            print("valid trials")
            print_data_info(self.valid_trials)

            if self.normalize:
                self.valid_trials, norm_params_valid = normalize(self.valid_trials,
                                                                 norm_type=self.normalize["norm_type"],
                                                                 axis=self.normalize["norm_axis"])
                print("Normliazed valid trials")
                print_data_info(self.valid_trials)

            gc.collect()

        elif self.loading_data_mode == "cross_subject":

            pass  # TODO : not implemented yet

        else:
            raise Exception(f"Loading data mode {self.loading_data_mode} not supported")

    def train_valid_split(self, data, train_val_split_dict):
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
