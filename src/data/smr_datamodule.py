# from bbcpy.datatypes.srm_eeg import *
import logging

import numpy as np
from omegaconf import OmegaConf
from sklearn.model_selection import KFold

import bbcpy

logging.getLogger().setLevel(logging.INFO)


def prepare_subject_data(data_dir,
                         ival,
                         bands,
                         chans,
                         classes,
                         subject_dict):
    """ Prepare the data for the classification """
    srm_data = SMR_Data(data_dir=data_dir,
                        bands=bands,
                        classes=classes,
                        chans=chans,
                        ival=ival)

    obj = srm_data.load_data(subject_dict=subject_dict)

    return obj


class SMR_Data():

    def __init__(self,
                 data_dir,
                 subject_sessions_dict,
                 concatenate_subjects,
                 loading_data_mode,
                 train_val_split,
                 ival,
                 bands,
                 chans,
                 classes,
                 threshold_distance,
                 fallback_neighbors
                 ):

        """ Initialize the SMR datamodule

        Parameters
        ----------
        data_dir: str
            Path to the data directory
        ival: list
            Timepoints to be selected
        bands: list
            Frequency bands to be selected
        chans: list
            Channels to be selected
        classes: list
            Classes to be selected
        """
        # FIXME : parameter are in type  omegaconf

        self.srm_data_path = data_dir
        self.subject_sessions_dict = subject_sessions_dict
        self.loaded_subjects_sessions = {}
        self.founded_subjects_sessions = {}

        self.concatenate_subjects = concatenate_subjects
        self.loading_data_mode = loading_data_mode
        self.train_val_split = OmegaConf.to_container(train_val_split)

        self.classes = OmegaConf.to_container(classes)
        self.select_chans = OmegaConf.to_container(chans)
        self.select_timepoints = ival
        self.bands = OmegaConf.to_container(bands)

        self.threshold_distance = threshold_distance
        self.fallback_neighbors = fallback_neighbors

        self.subject_info_dict = {}

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

    def load_session_raw_data(self, session_path):
        """ Load session raw data  """
        srm_data, timepoints, srm_fs, clab, mnt, trial_info, subject_info = \
            bbcpy.load.srm_eeg.load_single_mat_session(file_path=session_path)
        return srm_data, timepoints, srm_fs, clab, mnt, trial_info, subject_info

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

    def interpolate_noisy_channels(self, srm_obj, noisy_chans_idx):
        """ Interpolate noisy channels
         62-channel EEG cap (10-10 sytem) """

        # Calculate distances from the noisy channel to all other channels
        distances = np.linalg.norm(srm_obj.chans.mnt - srm_obj.chans.mnt[noisy_chans_idx], axis=1)

        # Identify neighboring channels
        neighbors = np.where(distances < self.threshold_distance)[0]

        # If no neighbors found within threshold, take the closest fallback_neighbors channels
        if len(neighbors) == 0 or (len(neighbors) == 1 and noisy_chans_idx in neighbors):
            neighbors = np.argsort(distances)[1:self.fallback_neighbors + 1]  # Excluding the channel itself

        # Exclude the noisy channel itself from the neighbors
        neighbors = neighbors[neighbors != noisy_chans_idx]

        # Calculate the average signal of neighboring channels
        average_signal = np.mean(srm_obj[neighbors], axis=0)

        # Replace the noisy channel's signal with the average signal
        srm_obj[noisy_chans_idx] = average_signal

        return srm_obj

    def remove_noisy_channels(self, srm_obj, noisy_chans):
        """ Remove noisy channels """
        pass

    def load_forced_valid_trials_data(self, session_name, sessions_group_path):

        session_path = sessions_group_path[session_name]
        self.subject_info_dict["noisy_chans"] = {}
        srm_data, timepoints, srm_fs, clab, mnt, trial_info, subject_info = \
            self.load_session_raw_data(session_path)


        # save subject info
        self.subject_info_dict["subject_info"] = subject_info

        # init channels object
        chans = bbcpy.datatypes.eeg.Chans(clab, mnt)

        # true labels
        target_map_dict = {1: "R", 2: "L", 3: "U", 4: "D"}
        mrk_class = np.array(trial_info["targetnumber"])
        mrk_class = mrk_class.astype(int) - 1  # to start from 0

        # Split data into train and test , where test contained forced trails
        trialresult = trial_info["result"]
        valid_trials_idx = np.where(trialresult == np.bool_(True))[0]

        # select all valid trials to train
        valid_data = srm_data[valid_trials_idx]
        valid_mrk_class = mrk_class[valid_trials_idx]
        valid_timepoints = timepoints[valid_trials_idx]
        class_names = np.array(["R", "L", "U", "D"])

        valid_mrk = bbcpy.datatypes.eeg.Marker(mrk_pos=valid_trials_idx,
                                               mrk_class=valid_mrk_class,
                                               mrk_class_name=class_names,
                                               mrk_fs=1,
                                               parent_fs=srm_fs)

        # create SRM_Data object for the session
        valid_trials = bbcpy.datatypes.srm_eeg.SRM_Data(srm_data=valid_data,
                                                        timepoints=valid_timepoints.reshape(-1, 1),
                                                        fs=srm_fs,
                                                        mrk=valid_mrk,
                                                        chans=chans)

        # select all forced trials to test
        forced_trials = trial_info["forcedresult"]
        forced_trials_idx = np.setdiff1d(np.where(forced_trials == np.bool_(True))[0], valid_trials_idx)

        # if len(forced_trials_idx) == 0:
        #     raise Exception(f"No forced trials found for session {session_name}")

        forced_data = srm_data[forced_trials_idx]
        forced_mrk_class = mrk_class[forced_trials_idx]
        forced_timepoints = timepoints[forced_trials_idx]
        forced_mrk = bbcpy.datatypes.eeg.Marker(mrk_pos=forced_trials_idx,
                                                mrk_class=forced_mrk_class,
                                                mrk_class_name=class_names,
                                                mrk_fs=1,
                                                parent_fs=srm_fs)
        # # create SRM_Data object for the session
        forced_trials = bbcpy.datatypes.srm_eeg.SRM_Data(srm_data=forced_data,
                                                         timepoints=forced_timepoints.reshape(-1, 1),
                                                         fs=srm_fs,
                                                         mrk=forced_mrk,
                                                         chans=chans)

        # remove noisy channels
        # if noisy channels are present, remove them from clab
        if len(subject_info["noisechan"]) > 0:
            noisy_chans_id_list = [int(chan) for chan in subject_info["noisechan"]]
            noisy_chans_name = [clab[chan] for chan in noisy_chans_id_list]
            self.subject_info_dict["noisy_chans"][session_name] = noisy_chans_name
            for noisy_chans_id in noisy_chans_id_list:
                valid_trials = self.interpolate_noisy_channels(valid_trials, noisy_chans_id)
                forced_trials = self.interpolate_noisy_channels(forced_trials, noisy_chans_id)

        # preprocess the data

        valid_trials = self.preprocess_data(valid_trials)
        forced_trials = self.preprocess_data(forced_trials)

        logging.info(
            f"{session_name} loaded;  valid trails shape: {valid_trials.shape},"
            f" forced trials shape: {forced_trials.shape}")
        return valid_trials, forced_trials

    def load_valid_trials_data(self, session_name, sessions_group_path):
        """ Create a session object from the SRM data sessions """

        session_path = sessions_group_path[session_name]

        srm_data, timepoints, srm_fs, clab, mnt, trial_info, subject_info = \
            self.load_session_raw_data(session_path)

        # init channels object
        chans = bbcpy.datatypes.eeg.Chans(clab, mnt)

        # true labels
        target_map_dict = {1: "R", 2: "L", 3: "U", 4: "D"}
        mrk_class = np.array(trial_info["targetnumber"])
        mrk_class = mrk_class.astype(int) - 1  # to start from 0

        # Split data into train and test , where test contained forced trails
        trialresult = trial_info["result"]
        valid_trials_idx = np.where(trialresult)[0]

        # select all valid trials to train
        valid_data = srm_data[valid_trials_idx]
        valid_mrk_class = mrk_class[valid_trials_idx]
        valid_timepoints = timepoints[valid_trials_idx]
        class_names = np.array(["R", "L", "U", "D"])

        valid_mrk = bbcpy.datatypes.eeg.Marker(mrk_pos=valid_trials_idx,
                                               mrk_class=valid_mrk_class,
                                               mrk_class_name=class_names,
                                               mrk_fs=1,
                                               parent_fs=srm_fs)

        # create SRM_Data object for the session
        obj = bbcpy.datatypes.srm_eeg.SRM_Data(srm_data=valid_data,
                                               timepoints=valid_timepoints.reshape(-1, 1),
                                               fs=srm_fs,
                                               mrk=valid_mrk,
                                               chans=chans)

        logging.info(f"Data original shape: {obj.shape}")
        # preprocess the data
        logging.info(f"Preprocessing data..")
        obj = self.preprocess_data(obj)
        logging.info(f"{session_name} loaded; has the shape: {obj.shape}")

        return obj

    def load_subject_sessions(self, subject_name, subject_dict, distributed_mode=False):

        """ Create a subject object from the SRM data sessions , concatenating the sessions over trails"""
        # FIXME : not completed

        sessions_path_dict = self.collect_subject_sessions(subject_dict)
        self.loaded_subjects_sessions[subject_name] = []

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
            self.loaded_subjects_sessions[subject_name].append(init_session_name)

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
                    self.loaded_subjects_sessions[subject_name].append(session_name)

                except Exception as e:
                    logging.info(f"Session {session_name} not loaded")
                    logging.warning(f"Exception occurred: {e}")
                    continue


        else:

            init_session_name = sessions_list[0]
            self.loaded_subjects_sessions[subject_name].append(init_session_name)
            valid_obj_new, forced_obj_new = self.load_forced_valid_trials_data(session_name=init_session_name,
                                                                               sessions_group_path=sessions_path_dict[
                                                                                   subject_name])
            logging.info(f"Loading sessions: {init_session_name} finalized (1 from 1)")

        return valid_obj_new, forced_obj_new

    def load_subjects(self, subject_dict, concatenate_subjects=True):
        """ Prepare the data for the classification """

        subjects_data = {"valid": {}, "forced": {}}
        subjects_sessions_path_dict = self.collect_subject_sessions(subject_dict)
        subjects_key = list(subjects_sessions_path_dict.keys())

        if len(subjects_key) > 1:
            init_subject_name = subjects_key[0]
            logging.info(f"Loading subject: {init_subject_name} finalized (1 from  {str(len(subjects_key))})")
            valid_Sobj_new, forced_Sobj_new = self.load_subject_sessions(subject_name=init_subject_name,
                                                                         sessions_group_path=subjects_sessions_path_dict)

            subjects_data["valid"][init_subject_name] = valid_Sobj_new.copy()
            subjects_data["forced"][init_subject_name] = forced_Sobj_new.copy()

            for i, subject_name in enumerate(subjects_key[1:]):
                logging.info(f"Loading subject: {subject_name} finalized ({str(i + 2)} from {str(len(subjects_key))})")

                valid_Sobj, forced_Sobj = self.load_subject_sessions(subject_name=subject_name,
                                                                     sessions_group_path=subjects_sessions_path_dict)

                valid_Sobj_new = valid_Sobj_new.append(valid_Sobj, axis=0)
                forced_Sobj_new = forced_Sobj_new.append(forced_Sobj, axis=0)

                subjects_data["valid"][subject_name] = valid_Sobj_new.copy()
                subjects_data["forced"][subject_name] = forced_Sobj_new.copy()
        else:
            init_subject_name = subjects_key[0]
            logging.info(f"Loading subject: {init_subject_name} finalized (1 from 1)")

            valid_Sobj_new, forced_Sobj_new = self.load_subject_sessions(subject_name=init_subject_name,
                                                                         sessions_group_path=subjects_sessions_path_dict)

            subjects_data["valid"][init_subject_name] = valid_Sobj_new.copy()
            subjects_data["forced"][init_subject_name] = forced_Sobj_new.copy()

        if not concatenate_subjects:
            # remove object from memory FIXME
            del valid_Sobj_new
            del forced_Sobj_new

            return subjects_data

        del subjects_data

        logging.info(f"Concatenating subject valid data, shape: {valid_Sobj_new.shape}")
        logging.info(f"Concatenating subject forced data, shape: {forced_Sobj_new.shape}")

        return valid_Sobj_new, forced_Sobj_new

    def prepare_dataloader(self):
        """ Prepare the data for the classification """

        if self.loading_data_mode == "within_subject":
            subject_name = list(self.subject_sessions_dict.keys())[0]
            valid_trial_trainset, forced_trial_testset = self.load_subject_sessions(subject_name=subject_name,
                                                                                    subject_dict=self.subject_sessions_dict)

        elif self.loading_data_mode == "within_subject_all":
            valid_trial_trainset, forced_trial_testset = self.load_subjects(self.subject_sessions_dict,
                                                                            self.concatenate_subjects)

            valid_trial_trainset = valid_trial_trainset.append(forced_trial_testset, axis=0)

        elif self.loading_data_mode == "cross_subject":
            valid_trial_trainset, forced_trial_testset = self.load_subjects(self.subject_sessions_dict,
                                                                            self.concatenate_subjects)

        else:
            raise Exception(f"Loading data mode {self.loading_data_mode} not supported")

        return valid_trial_trainset, forced_trial_testset

    def kfolding_within_subject(self, subject_dict, cv_mode, prepare_test=False):

        subject_id = subject_dict.keys()[0]
        sessions_ids = subject_dict[subject_id]

        if len(sessions_ids) < 2:
            raise Exception(f"Subject {subject_id} has only one session, take at least two sessions for k-folding")

        sessions_group_path = self.collect_subject_sessions(subject_dict)

        sessions_data = {}

        kfold = KFold(n_splits=len(sessions_ids), shuffle=True, random_state=42)

        for session_name in sessions_ids:
            logging.info(f"Loading session: {session_name} finalized")
            sessions_data[session_name] = self.load_valid_trials_data(session_name=session_name,
                                                                      sessions_group_path=sessions_group_path[
                                                                          subject_id])

        channels_kfolded = []

        for fold, (train_index, test_index) in enumerate(kfold.split(sessions_ids)):
            foldNum = fold + 1

        pass

    def calculate_class_weights(self, data):
        """ Calculate the class weights for the data """

        pass

    def kfolding_cross_subjects(self, subjects_dict, cv_mode, prepare_test=False):
        pass

    def split_data(self, data):
        pass


if __name__ == "__main__":
    srm_raw_path = "../../data/SMR/raw/"

    obj = prepare_subject_data(data_dir=srm_raw_path,
                               ival="2s:8s:10ms",
                               bands=[8, 13],
                               chans=['C*', 'FC*'],
                               classes=["R", "L", "U"],
                               subject_dict={"S1": [1, 2, 3, 4]})

    print(obj)
