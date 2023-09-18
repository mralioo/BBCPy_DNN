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


class SMR_Data():

    def __init__(self,
                 data_dir,
                 task_name,
                 subject_sessions_dict,
                 concatenate_subjects,
                 loading_data_mode,
                 ival,
                 bands,
                 chans,
                 classes,
                 threshold_distance,
                 fallback_neighbors,
                 transform,
                 normalize,
                 ):

        """ Initialize the SMR datamodule
        """
        # FIXME : parameter are in type  omegaconf

        self.srm_data_path = data_dir
        self.task_name = task_name
        self.subject_sessions_dict = subject_sessions_dict
        self.loaded_subjects_sessions = {}
        self.founded_subjects_sessions = {}

        self.concatenate_subjects = concatenate_subjects
        self.loading_data_mode = loading_data_mode

        self.classes = OmegaConf.to_container(classes)
        self.select_chans = OmegaConf.to_container(chans)
        self.select_timepoints = ival
        self.bands = OmegaConf.to_container(bands)

        self.threshold_distance = threshold_distance
        self.fallback_neighbors = fallback_neighbors

        if transform:
            self.transform = OmegaConf.to_container(transform)
        else:
            self.transform = None

        if normalize:
            self.normalize = OmegaConf.to_container(normalize)
        else:
            self.normalize = None

        self.subjects_info_dict = {}
        self.subject_info_dict = {"noisechan": {}, "pvc": {self.task_name: {}}}
        self.subject_pvcs = []

    @property
    def num_classes(self):
        return len(self.classes)

    def collect_subject_sessions(self, subjects_dict):
        """ Collect all the sessions for the subjects in the list """

        # if not isinstance(subjects_dict, dict) or not isinstance(subjects_dict, omegaconf.dictconfig.DictConfig):
        #     raise Exception(f"subjects_dict must be a dictionary with S*:[1,2,3,4]")

        subjects_sessions_path_dict = {}
        for subject_name, sessions_ids in subjects_dict.items():
            logging.info(f"Collecting subject {subject_name} sessions from:  {self.srm_data_path}")

            sessions_group_path = bbcpy.load.srm_eeg.list_all_files(self.srm_data_path,
                                                                    pattern=f"{subject_name}_*.mat")[subject_name]

            self.founded_subjects_sessions[subject_name] = list(sessions_group_path.keys())

            if sessions_ids == "all":
                # select all sessions
                logging.info(f"Found sessions: {list(sessions_group_path.keys())}")

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

        if self.bands is not None:
            obj = obj.lfilter(self.bands)

        return obj

    def interpolate_noisy_channels(self, srm_obj, noisy_chans_idx, session_name):
        """ Interpolate noisy channels of 62-channel EEG cap (10-10 system), take the average of the neighboring channels

        Parameters
        ----------
        srm_obj: bbcpy.datatypes.srm_eeg.SRM_Data
            SRM data object
        noisy_chans_idx: int
            Index of the noisy channel

        Returns
        -------
        srm_obj: bbcpy.datatypes.srm_eeg.SRM_Data
            SRM data object with interpolated noisy channel


        """

        # Calculate distances from the noisy channel to all other channels
        distances = np.linalg.norm(srm_obj.chans.mnt - srm_obj.chans.mnt[noisy_chans_idx], axis=1)

        # Identify neighboring channels
        # neighbors = np.where(distances < self.threshold_distance)[0]

        # If no neighbors found within threshold, take the closest fallback_neighbors channels
        neighbors = np.argsort(distances)[1:self.fallback_neighbors + 1]  # Excluding the channel itself

        noise_chans_name = str(srm_obj.chans[noisy_chans_idx])
        self.subject_info_dict["noisechan"][session_name][noise_chans_name] = {"neighbors": srm_obj.chans[neighbors]}

        # Calculate the average signal of neighboring channels
        average_signal = np.mean(srm_obj[:, neighbors, :], axis=1)

        # Replace the noisy channel's signal with the average signal
        srm_obj[:, noisy_chans_idx, :] = np.expand_dims(average_signal, axis=1)

        return srm_obj

    def load_forced_valid_trials_data(self, session_name, sessions_group_path):
        session_path = sessions_group_path[session_name]

        srm_data, timepoints, srm_fs, clab, mnt, trial_info, subject_info = \
            bbcpy.load.srm_eeg.load_single_mat_session(file_path=session_path)

        # save subject info
        self.subject_info_dict["subject_info"] = subject_info
        self.subject_info_dict["noisechan"][session_name] = {}

        # save pvc

        sessions_pvc = calculate_pvc_metrics(trial_info, taskname=self.task_name)
        self.subject_pvcs.append(sessions_pvc)

        self.subject_info_dict["pvc"][self.task_name][session_name] = sessions_pvc

        # set the EEG channels object, and remove the reference channel if exists
        chans = remove_reference_channel(clab, mnt)
        # transform the channels order

        # true labels
        target_map_dict = {1: "R", 2: "L", 3: "U", 4: "D"}
        mrk_class = np.array(trial_info["targetnumber"])
        mrk_class = mrk_class.astype(int) - 1  # to start from 0

        class_names = np.array(["R", "L", "U", "D"])

        # all trials
        # FIXME : not sure what to pass for mrk_pos
        # mrk pos here is the results of all trials
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

        # remove noisy channels
        # if noisy channels are present, remove them from the data
        if subject_info["noisechan"] is not None:
            noisy_chans_id_list = [int(chan) for chan in subject_info["noisechan"]]
            for noisy_chans_id in noisy_chans_id_list:
                epo_data = self.interpolate_noisy_channels(epo_data, noisy_chans_id, session_name)

        if self.transform == "TSCeption":
            epo_data = transform_electrodes_configurations(epo_data)

        # valid trials are defined as the trials where labeles are correct (true) in trialresult

        valid_trials_idx = np.where(epo_data.mrk == np.bool_(True))[0]
        valid_trials = epo_data[valid_trials_idx]

        # forced trials are defined as the trials where labeles are correct (true) in forcedresult
        # and not in valid trials
        forced_trials = trial_info["forcedresult"]
        forced_trials_idx = np.setdiff1d(np.where(forced_trials == np.bool_(True))[0], valid_trials_idx)
        forced_trials = epo_data[forced_trials_idx]

        # preprocess the data
        valid_trials = self.preprocess_data(valid_trials)
        forced_trials = self.preprocess_data(forced_trials)

        logging.info(f"{session_name} loaded;"
                     f" valid trails shape: {valid_trials.shape},"
                     f" forced trials shape: {forced_trials.shape}")
        return valid_trials, forced_trials

    def load_subject_sessions(self, subject_name, subject_dict, distributed_mode=False):

        """ Create a subject object from the SRM data sessions , concatenating the sessions over trails"""
        # FIXME : not completed

        sessions_path_dict = self.collect_subject_sessions(subject_dict)
        self.loaded_subjects_sessions[subject_name] = {}

        # sessions_list = sessions_path_dict[subject_name]
        sessions_list = list(sessions_path_dict[subject_name].keys())

        logging.info(f"Prepare to Load : {sessions_list} sessions")

        if len(sessions_list) > 1:
            init_session_name = sessions_list[0]

            # load the first session to init the object
            valid_obj_new, forced_obj_new = self.load_forced_valid_trials_data(session_name=init_session_name,
                                                                               sessions_group_path=sessions_path_dict[
                                                                                   subject_name])

            logging.info(f"Loading {init_session_name} finalized (1 from {str(len(sessions_list))})")
            self.loaded_subjects_sessions[subject_name][init_session_name] = [valid_obj_new.shape, forced_obj_new.shape]

            for i, session_name in enumerate(sessions_list[1:]):
                logging.info(
                    f"Loading {session_name} finalized ({str(i + 2)} from {str(len(sessions_list))})")

                try:
                    valid_obj, forced_obj = self.load_forced_valid_trials_data(session_name=session_name,
                                                                               sessions_group_path=sessions_path_dict[
                                                                                   subject_name])
                    # check if the valid data has the same datapoints
                    if valid_obj.shape[2] != valid_obj_new.shape[2]:
                        # reshape the data
                        valid_obj_len = valid_obj_new.shape[2]
                        valid_obj = valid_obj[:, :, :valid_obj_len]
                    valid_obj_new = valid_obj_new.append(valid_obj, axis=0)

                    # check if the forced data has the same datapoints
                    if forced_obj.shape[2] != forced_obj_new.shape[2]:
                        # reshape the data
                        forced_obj_len = forced_obj_new.shape[2]
                        forced_obj = forced_obj[:, :, :forced_obj_len]

                    # check if the forced data exists
                    if forced_obj.shape[0] == 0:
                        logging.info(f"Session {session_name} has no forced trials")
                        continue

                    forced_obj_new = forced_obj_new.append(forced_obj, axis=0)

                    # append successfully loaded session
                    self.loaded_subjects_sessions[subject_name][session_name] = [valid_obj.shape, forced_obj.shape]

                except Exception as e:
                    logging.info(f"Session {session_name} not loaded")
                    logging.warning(f"Exception occurred: {e}")
                    continue
        else:
            init_session_name = sessions_list[0]

            valid_obj_new, forced_obj_new = self.load_forced_valid_trials_data(session_name=init_session_name,
                                                                               sessions_group_path=sessions_path_dict[
                                                                                   subject_name])

            self.loaded_subjects_sessions[subject_name][init_session_name] = [valid_obj_new.shape, forced_obj_new.shape]
            logging.info(f"Loading sessions: {init_session_name} finalized (1 from 1)")

        return valid_obj_new, forced_obj_new

    def load_all_subjects_sessions(self, subject_dict, concatenate_subjects=False):
        """ Load all the subjects sessions and concatenate them """

        # FIXME : not completed

        subjects_sessions_path_dict = self.collect_subject_sessions(subject_dict)

        subject_data_valid_dict = {}
        subject_data_forced_dict = {}

        for subject_name, sessions_ids in subjects_sessions_path_dict.items():
            logging.info(f"Loading subject {subject_name} sessions")
            valid_obj_new, forced_obj_new = self.load_subject_sessions(subject_name=subject_name,
                                                                       subject_dict=subject_dict)

            # TODO : calculate subject pvc

            subject_data_valid_dict[subject_name] = valid_obj_new
            subject_data_forced_dict[subject_name] = forced_obj_new

            # TODO : calculate subject pvc
            # calculate subject pvc
            pvc_mean = np.mean(self.subject_pvcs)
            self.subject_info_dict["pvc"][self.task_name]["mean"] = pvc_mean
            self.subjects_info_dict[subject_name] = self.subject_info_dict

        # concatenate all the subjects data
        if concatenate_subjects:
            init_subject_name = list(subject_data_valid_dict.keys())[0]
            valid_trials = subject_data_valid_dict[init_subject_name]
            forced_trials = subject_data_forced_dict[init_subject_name]

            for subject_name in subject_data_valid_dict.keys():
                if subject_name == init_subject_name:
                    continue
                valid_trials = valid_trials.append(subject_data_valid_dict[subject_name], axis=0)
                forced_trials = forced_trials.append(subject_data_forced_dict[subject_name], axis=0)

            # delete the subject data dict
            del subject_data_valid_dict
            del subject_data_forced_dict

            return valid_trials, forced_trials

        else:
            return subject_data_valid_dict, subject_data_forced_dict

    def prepare_dataloader(self):
        """ Prepare the data for the classification """

        if self.loading_data_mode == "within_subject":

            subject_name = list(self.subject_sessions_dict.keys())[0]
            self.valid_trials, self.forced_trials = self.load_subject_sessions(subject_name=subject_name,
                                                                               subject_dict=self.subject_sessions_dict)

            if not self.normalize:
                self.valid_trials, norm_params_valid = normalize(self.valid_trials,
                                                                 norm_type=self.normalize["norm_type"],
                                                                 axis=self.normalize["norm_axis"])

                self.forced_trials, norm_params_forced = normalize(self.forced_trials,
                                                                   norm_type=self.normalize["norm_type"],
                                                                   axis=self.normalize["norm_axis"])

            # calculate subject pvc
            pvc_mean = np.mean(self.subject_pvcs)
            self.subject_info_dict["pvc"][self.task_name]["mean"] = pvc_mean
            self.subjects_info_dict[subject_name] = self.subject_info_dict


        elif self.loading_data_mode == "cross_subject_hpo":

            self.valid_trials, self.forced_trials = self.load_all_subjects_sessions(self.subject_sessions_dict,
                                                                                    concatenate_subjects=True)

            if not self.normalize:
                self.valid_trials, norm_params_valid = normalize(data=self.valid_trials,
                                                                 norm_type=self.normalize["norm_type"],
                                                                 axis=self.normalize["norm_axis"])

                self.forced_trials, norm_params_forced = normalize(data=self.forced_trials,
                                                                   norm_type=self.normalize["norm_type"],
                                                                   axis=self.normalize["norm_axis"])

        elif self.loading_data_mode == "cross_subject":
            self.valid_trials, self.forced_trials = self.load_subject_sessions(self.subject_sessions_dict,
                                                                               self.concatenate_subjects)

        else:
            raise Exception(f"Loading data mode {self.loading_data_mode} not supported")

    def train_valid_split(self, data, train_val_split):
        """ Split the data into train and validation sets """

        # Shuffle the data
        assert np.isclose(sum(train_val_split), 1.0), "Split ratios should sum up to 1.0"

        # TODO : Shuffle the SMR data object
        indices = np.random.permutation(len(data))
        data = data[indices]
        data.y = data.y[indices]

        splits = []
        start_idx = 0

        for ratio in train_val_split:
            end_idx = start_idx + int(len(data) * ratio)
            splits.append(data[start_idx:end_idx])
            start_idx = end_idx

        return splits[0], splits[1]


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
