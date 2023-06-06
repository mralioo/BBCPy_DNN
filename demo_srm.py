import numpy as np
from datasets import utils as dutils
from datasets.datatypes import SRM_Data, Chans, SRM_Marker, makeepochs_srm, gettimeindices

srm_data_root_path = "./local/data/SMR/raw"


def load_srm_subject_data(subjectno=1, sessions_list=[1, 2, 3]):
    pass


if __name__ == "__main__":
    srm_data, timepoints, fs, clab, mnt, trial_info, subject_info = dutils.load_single_mat_session(subjectno=1,
                                                                                                   sessionno=1,
                                                                                                   data_path=srm_data_root_path)

    mrk_className = ["R", "L", "U", "D"]

    mrk_class = trial_info["targetnumber"]
    trialresult = trial_info["result"]
    triallength = trial_info["triallength"]
    mrk_class_name = np.unique(mrk_class)

    mrk_obj = SRM_Marker(mrk_class, mrk_class_name, trialresult, triallength, fs)
    zz = mrk_obj.select_classes(["R", "L"])
    # rr = mrk_obj[["R", "L"]] # should this work ?
    # channel class
    chans = Chans(clab, mnt)
    # reshape srm_data to even size trials

    ival_orig = [1000, 4500]
    ival = gettimeindices(ival_orig, 1000)

    data, time = makeepochs_srm(srm_data, timepoints, ival, fs, mrk_obj)

    # # TODO subject all sessions
    # # TODO trail info and subject info combine to epo class
    s1_obj = SRM_Data(data, time, fs, mrk=mrk_obj, chans=chans)
    #
    test_slice = s1_obj[['R', 'L']][:, ['C?,~*3,~*4', 'FC*,~*3,~*4'], '2s:4s:100ms']
    print(test_slice.shape)
    # smart indexing trail, channel, time

    #
