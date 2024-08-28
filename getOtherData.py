import warnings
import numpy as np
import scipy.io as sio

warnings.filterwarnings('ignore')
# 필요에 따라 설정
DATASET_ROOT = "D:/EEG datasets/"


# MI, 250Hz, C3,Cz,C4: 7,9,11 l/r hand: 0,1 foot, tongue: 2,3
# subj: 1-9, 0.5-100Hz, 10-20 system, returns 7.5s data, 2s before and 1.5s after the trial
def get_data_2a(subject, training, root_path=DATASET_ROOT+'BCICIV_2a_mat/'):
    '''	Loads the dataset 2a of the BCI Competition IV
    available on http://bnci-horizon-2020.eu/database/data-sets

    Keyword arguments:
    subject -- number of subject in [1, .. ,9]
    training -- if True, load training data
                if False, load testing data

    Return:	data_return 	numpy matrix 	size = NO_valid_trial x 22 x 1825
            class_return 	numpy matrix 	size = NO_valid_trial
    '''
    # two sessions per subject, trial: 4s, 250Hz

    # reference: left mastoid, ground: right mastoid
    # sampling rate: 250Hz
    # bandpassed: 0.5-100Hz
    NO_channels = 22
    NO_tests = 6 * 48
    Window_Length = int(7.5 * 250)

    class_return = np.zeros(NO_tests)
    data_return = np.zeros((NO_tests, NO_channels, Window_Length))

    NO_valid_trial = 0
    if training:
        a = sio.loadmat(root_path + 'A0' + str(subject) + 'T.mat')
    else:
        a = sio.loadmat(root_path + 'A0' + str(subject) + 'E.mat')
    a_data = a['data']
    for ii in range(0, a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0, 0]]
        a_data3 = a_data2[0]
        a_X = a_data3[0] # entire data (refer to comment before for loop)
        a_trial = a_data3[1]
        a_y = a_data3[2]
        a_fs = a_data3[3]
        a_classes = a_data3[4]
        a_artifacts = a_data3[5]
        a_gender = a_data3[6]
        a_age = a_data3[7]
        # a_trial.size > 0 means there is data in this run
        for trial in range(0, a_trial.size):
            # remove bad trials
            # if (a_artifacts[trial] == 0):
            data_return[NO_valid_trial, :, :] = np.transpose(
                a_X[int(a_trial[trial]):(int(a_trial[trial]) + Window_Length), :NO_channels]) # 4s pre-trial and 4s post-trial
            class_return[NO_valid_trial] = int(a_y[trial])
            NO_valid_trial += 1

    # index 500~1500 is the imagery time
    data = data_return[0:NO_valid_trial, :, :]
    class_return = class_return[0:NO_valid_trial]-1

    return data, class_return