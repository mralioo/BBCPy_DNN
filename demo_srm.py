import bbcpy
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import scipy.io as sio
from datasets import utils as dutils

if __name__ == "__main__":
    data, timepoints, fs, clab, mnt, trial_info, metadata = dutils.load_single_mat_session(subjectno=1, sessionno=1,
                                                                                           data_path="./data/SMR/raw")


    print(data.shape)