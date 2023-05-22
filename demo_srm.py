import bbcpy
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import scipy.io as sio
from datasets import utils as dutils
from data.SMR.eeg import *
from bbcpy.datatypes.eeg import Data, Chans, Marker, Epo
from datasets.datatypes import SRM_Data

if __name__ == "__main__":



    imagVPaw = bbcpy.load.eeg.data('./local/data/imagVPaw/imagVPaw.npz')

    ival = [1000, 4500]
    band = [10.5, 13]

    cnt_bp = imagVPaw.lfilter(band)
    epo_bp = cnt_bp.epochs(ival)
    print(data.shape)
    #select all but some (here exclude all frontal and fronto-central) channels:
    print(imagVPaw['~F*'].shape)
    #can also be combined - selecting all frontal channels except for 7 and 8s e.g.:
    print(imagVPaw[['F?', '~*7', '~*8']].shape)
    print(imagVPaw[['F?,~*7,~*8']].chans)
    #adding advanced slicing with units - e.g. 2s to 5s in steps of 100ms:
    print(imagVPaw['~F*', '2s:5s:100ms'].shape)
    #you may also select single time points:
    print(imagVPaw[:, '2s,5s,100ms'].shape)
    #the same can be separated in a list:
    print(imagVPaw[:, ['2s', '5s', '100ms']].shape)



    data, timepoints, fs, clab, mnt, trial_info, subject_info = dutils.load_single_mat_session(subjectno=1,
                                                                                           sessionno=1,
                                                                                           data_path="./local/data/SMR/raw")
    s1_obj = SRM_Data(data, fs, trial_info, subject_info, clab)

    # smart indexing trail, channel, time
