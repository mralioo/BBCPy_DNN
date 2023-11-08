import os
import sys
from pathlib import Path

module_path = os.path.abspath(os.path.join('../../'))
if module_path not in sys.path:
    sys.path.append(module_path)
from src.data.smr_datamodule import SMR_Data
from src.utils.device import print_data_info
import mne
import bbcpy

data_dir = Path("D:\\SMR\\")
task_name = "LR"
subject_sessions_dict = {"S1": "all"}
loading_data_mode = "within_subject"
ival = "2s:10s:1ms"
# bands = [8, 13]
bands = None
chans = "*"
fallback_neighbors = 4
transform = None
normalize = None

process_noisy_channels = True
ignore_noisy_sessions = False

trial_type = "valid"

smr_datamodule = SMR_Data(data_dir=data_dir,
                          task_name=task_name,
                          trial_type=trial_type,
                          subject_sessions_dict=subject_sessions_dict,
                          loading_data_mode=loading_data_mode,
                          ival=ival,
                          bands=bands,
                          chans=chans,
                          fallback_neighbors=fallback_neighbors,
                          transform=transform,
                          normalize=normalize,
                          process_noisy_channels=process_noisy_channels,
                          ignore_noisy_sessions=ignore_noisy_sessions)

subjects_sessions_path_dict = smr_datamodule.collect_subject_sessions(subject_sessions_dict)



if __name__ == "__main__" :


    subjects_sessions_path_dict = smr_datamodule.collect_subject_sessions({"S1": "all"})

    data, timepoints, fs, clab, mnt, trial_info, metadata = (
        bbcpy.load.srm_eeg.load_single_mat_session(file_path=subjects_sessions_path_dict["S1"]["Session_1"]))

    epo_data, session_info = smr_datamodule.load_session_all_runs(subjects_sessions_path_dict["S1"]["Session_1"])


    smr_datamodule.prepare_dataloader_1()
    train_data = smr_datamodule.train_data
    eeg_data = train_data.data
    info = mne.create_info(ch_names=train_data.chans, sfreq=train_data.fs, ch_types=['eeg'] * train_data.nCh)
    # Create a RawArray object
    # Create an EpochsArray object
    epochs = mne.EpochsArray(eeg_data, info)

    import numpy as np
    import matplotlib.pyplot as plt

    # We'll also plot a sample time onset for each trial
    plt_times = np.linspace(0, 0.2, len(epochs))

    plt.close("all")
    mne.viz.plot_epochs_image(
        epochs,
        ["C3", "C4"],
        sigma=0.5,
        overlay_times=plt_times,
        show=True,
        block=True,
    )


    epochs.plot_image(picks=["C3", "C4"],combine="mean")

