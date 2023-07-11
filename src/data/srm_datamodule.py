# from bbcpy.datatypes.srm_eeg import *
import logging

import numpy as np

import bbcpy

logging.getLogger().setLevel(logging.INFO)


class SRMDatamodule():

    def __init__(self,
                 data_dir,
                 ival,
                 bands,
                 chans,
                 classes):
        """ Initialize the SRM datamodule

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

        if not isinstance(classes, list):
            classes = list(classes)
        if not isinstance(chans, list):
            chans = list(chans)
        if not isinstance(bands, list):
            bands = list(bands)

        self.classes = classes
        self.select_chans = chans
        self.select_timepoints = ival
        self.bands = bands

    def collect_subject_sessions(self, subjects_dict):
        """ Collect all the sessions for the subjects in the list """

        if not isinstance(subjects_dict, dict):
            raise Exception(f"subjects_dict must be a dictionary with S*:[1,2,3,4]")

        subjects_sessions_path_dict = {}
        for subject_name, sessions_ids in subjects_dict.items():
            logging.info(f"Collecting subject {subject_name} sessions from:  {self.srm_data_path}")
            sessions_group_path = bbcpy.load.srm_eeg.list_all_files(self.srm_data_path,
                                                                    pattern=f"{subject_name}_*.mat")[subject_name]

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
            obj = srm_obj.copy()

        if self.bands is not None:
            obj = obj.lfilter(self.bands).copy()

        return obj

    def load_session_data(self, session_name, sessions_group_path):
        """ Create a session object from the SRM data sessions """

        session_path = sessions_group_path[session_name]

        srm_data, timepoints, srm_fs, clab, mnt, trial_info, subject_info = \
            self.load_session_raw_data(session_path)

        mrk_class = np.array(trial_info["targetnumber"])
        mrk_class = mrk_class.astype(int) - 1  # to start from 0

        mrk_class_name = ["R", "L", "U", "D"]

        trialresult = trial_info["result"]
        # initialize SRM_Marker object
        mrk = bbcpy.datatypes.srm_eeg.SRM_Marker(mrk_class, mrk_class_name, trialresult, srm_fs)
        # initialize SRM_Chans object
        chans = bbcpy.datatypes.eeg.Chans(clab, mnt)

        # create SRM_Data object for the session
        obj = bbcpy.datatypes.srm_eeg.SRM_Data(srm_data,
                                               timepoints.reshape(-1, 1),
                                               srm_fs,
                                               mrk=mrk,
                                               chans=chans)

        # preprocess the data
        logging.info(f"Preprocessing data..")
        obj = self.preprocess_data(obj)
        logging.info(f"{session_name} loaded; has the shape: {obj.shape}")
        return obj

    def load_subject_data(self, sessions_group_path):
        """ Create a subject object from the SRM data sessions , concatenating the sessions over trails"""

        sessions_key = list(sessions_group_path.keys())
        logging.info(f"Prepare to Load : {sessions_key} sessions")
        if len(sessions_key) > 1:
            init_session_name = sessions_key[0]
            logging.info(
                f"Loading {init_session_name} finalized (1 from {str(len(sessions_key))})")
            obj_new = self.load_session_data(session_name=init_session_name,
                                             sessions_group_path=sessions_group_path)

            for i, session_name in enumerate(sessions_key[1:]):
                logging.info(
                    f"Loading {session_name} finalized ({str(i + 2)} from {str(len(sessions_key))})")
                obj = self.load_session_data(session_name=session_name,
                                             sessions_group_path=sessions_group_path)
                obj_new = obj_new.append(obj, axis=0)

        else:
            init_session_name = sessions_key[0]
            obj_new = self.load_session_data(session_name=init_session_name,
                                             sessions_group_path=sessions_group_path)
            logging.info(f"Loading sessions: {init_session_name} finalized (1 from 1)")

        return obj_new

    def load_data(self, subjects_dict, concatenate_subjects=True):
        """ Prepare the data for the classification """

        subjects_data = {}
        subjects_sessions_path_dict = self.collect_subject_sessions(subjects_dict)
        subjects_key = list(subjects_sessions_path_dict.keys())

        if len(subjects_key) > 1:
            init_subject_name = subjects_key[0]
            logging.info(f"Loading subject: {init_subject_name} finalized (1 from  {str(len(subjects_key))})")
            obj_new = self.load_subject_data(sessions_group_path=subjects_sessions_path_dict[init_subject_name])
            subjects_data[init_subject_name] = obj_new.copy()

            for i, subject_name in enumerate(subjects_key[1:]):
                logging.info(
                    f"Loading subject: {subject_name} finalized ({str(i + 2)} from {str(len(subjects_key))})")
                obj = self.load_subject_data(sessions_group_path=subjects_sessions_path_dict[subject_name])
                obj_new = obj_new.append(obj, axis=0)

                subjects_data[subject_name] = obj_new.copy()
        else:
            init_subject_name = subjects_key[0]
            logging.info(f"Loading subject: {init_subject_name} finalized (1 from 1)")
            obj_new = self.load_subject_data(sessions_group_path=subjects_sessions_path_dict[init_subject_name])

            subjects_data[init_subject_name] = obj_new.copy()

        # if not concatenate_subjects:
        #     # remove object from memory FIXME
        #     del obj_new
        #     return subjects_data

        return obj_new
