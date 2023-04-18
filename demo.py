import bbcpy
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import scipy.io as sio

if __name__ == "__main__":
    vp_fb_parameters = sio.loadmat('local/data/SMR/raw/S1_Session_1.mat')
