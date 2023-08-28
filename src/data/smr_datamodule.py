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
                 train_subjects_sessions_dict,
                 test_subjects_sessions_dict,
                 concatenate_subjects,
                 loading_data_mode,
                 train_val_split,
                 ival,
                 bands,
                 chans,
                 classes,
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
        self.train_subjects_sessions_dict = train_subjects_sessions_dict
        self.test_subjects_sessions_dict = test_subjects_sessions_dict

        self.concatenate_subjects = concatenate_subjects
        self.loading_data_mode = loading_data_mode
        self.train_val_split = OmegaConf.to_container(train_val_split)

        self.classes = OmegaConf.to_container(classes)
        self.select_chans = OmegaConf.to_container(chans)
        self.select_timepoints = ival
        self.bands = OmegaConf.to_container(bands)

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

            logging.info(f"Found sessions: {list(sessions_group_path.keys())}")

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

    def load_forced_valid_trials_data(self, session_name, sessions_group_path):

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

        valid_trials = self.preprocess_data(valid_trials)

        # select all forced trials to test
        forced_trials = trial_info["forcedresult"]
        forced_trials_idx = np.setdiff1d(np.where(forced_trials == np.bool_(True))[0], valid_trials_idx)
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

        # sessions_list = sessions_path_dict[subject_name]
        sessions_list = list(sessions_path_dict[subject_name].keys())

        logging.info(f"Prepare to Load : {sessions_list} sessions")

        if len(sessions_list) > 1:
            init_session_name = sessions_list[0]

            valid_obj_new, forced_obj_new = self.load_forced_valid_trials_data(session_name=init_session_name,
                                                                               sessions_group_path=sessions_path_dict[
                                                                                   subject_name])

            logging.info(f"Loading {init_session_name} finalized (1 from {str(len(sessions_list))})")

            for i, session_name in enumerate(sessions_list[1:]):
                logging.info(
                    f"Loading {session_name} finalized ({str(i + 2)} from {str(len(sessions_list))})")
                valid_obj, forced_obj = self.load_forced_valid_trials_data(session_name=session_name,
                                                                           sessions_group_path=sessions_path_dict[
                                                                               subject_name])

                valid_obj_new = valid_obj_new.append(valid_obj, axis=0)
                forced_obj_new = forced_obj_new.append(forced_obj, axis=0)

        else:

            init_session_name = sessions_list[0]
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
            subject_name = list(self.train_subjects_sessions_dict.keys())[0]
            valid_trial_trainset, forced_trial_testset = self.load_subject_sessions(subject_name=subject_name,
                                                                                    subject_dict=self.train_subjects_sessions_dict)

        elif self.loading_data_mode == "within_subject_all":
            valid_trial_trainset, forced_trial_testset = self.load_subjects(self.train_subjects_sessions_dict,
                                                                            self.concatenate_subjects)

            valid_trial_trainset = valid_trial_trainset.append(forced_trial_testset, axis=0)

        elif self.loading_data_mode == "cross_subject":
            valid_trial_trainset, forced_trial_testset = self.load_subjects(self.train_subjects_sessions_dict,
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
