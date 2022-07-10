import os
import random
import sys
from datasets.preprocess_utils import *
from datimport os
import random
import sys
from datasets.preprocess_utils import *
from datasets.cont_smr import *
from plotly.offline import init_notebook_mode

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

init_notebook_mode(connected=True)  ## plotly init
seed = 123
random.seed = seed

from datasets.cont_smr import *
from bokeh.io import output_notebook

output_notebook()asets.cont_smr import *
from plotly.offline import init_notebook_mode

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

init_notebook_mode(connected=True)  ## plotly init
seed = 123
random.seed = seed

from datasets.cont_smr import *
from bokeh.io import output_notebook

output_notebook()


def sort_by_gender(subjects_dict):
    res_dict = {"M": [], "F": []}
    for subject, sessions in subjects_dict.items():
        sessions_id = list(sessions.keys())
        gender = sessions[sessions_id[0]]["metadata"]["gender"]
        res_dict[gender].append(subjects_dict[subject])
    return res_dict


def sort_by_handedness(subjects_dict):
    res_dict = {"R": [], "L": []}
    for subject, sessions in subjects_dict.items():
        sessions_id = list(sessions.keys())
        handedness = sessions[sessions_id[0]]["metadata"]["handedness"]
        res_dict[handedness].append(subjects_dict[subject])
    return res_dict


def get_trial_stats(session_dict):
    trial_stat_dict = {}
    true_trials_idx = np.where(np.array(session_dict["trial_info"]["result"]) == True)[0]
    false_trials_idx = np.where(np.array(session_dict["trial_info"]["result"]) == False)[0]
    error_trials_idx = np.where(np.isnan(np.array(session_dict["trial_info"]["result"])))[0]
    trial_stat_dict["true"] = len(true_trials_idx)
    trial_stat_dict["false"] = len(false_trials_idx)
    trial_stat_dict["error"] = len(error_trials_idx)
    trial_stat_dict["mean_triallength"] = np.mean(np.array(session_dict["trial_info"]["triallength"]))
    trial_stat_dict["std_triallength"] = np.std(np.array(session_dict["trial_info"]["triallength"]))
    return trial_stat_dict


def get_subjects_stats(subjects_dict):
    subjects_stats = {}
    for subject, sessions in subjects_dict.items():
        sessions_id = list(sessions.keys())
        subjects_stats[subject] = {}
        subjects_stats[subject]["gender"] = sessions[sessions_id[0]]["metadata"]["gender"]
        subjects_stats[subject]["age"] = sessions[sessions_id[0]]["metadata"]["age"]
        subjects_stats[subject]["handedness"] = sessions[sessions_id[0]]["metadata"]["handedness"]
        subjects_stats[subject]["handsport"] = sessions[sessions_id[0]]["metadata"]["handsport"]
        subjects_stats[subject]["instrument"] = sessions[sessions_id[0]]["metadata"]["instrument"]
        subjects_stats[subject]["meditationpractice"] = sessions[sessions_id[0]]["metadata"]["meditationpractice"]
        subjects_stats[subject]["MBSRsubject"] = sessions[sessions_id[0]]["metadata"]["MBSRsubject"]
    return subjects_stats
