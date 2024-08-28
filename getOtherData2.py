import warnings
import numpy as np
import scipy.io as sio

warnings.filterwarnings('ignore')
DATASET_ROOT = "D:/EEG datasets/"

def get_data_2a(subject, root_path=DATASET_ROOT + 'BCICIV_2a_mat/'):
    NO_channels = 22
    NO_tests = 6 * 48
    Window_Length = int(7.5 * 250)

    class_return_train = np.zeros(NO_tests)
    data_return_train = np.zeros((NO_tests, NO_channels, Window_Length))
    class_return_test = np.zeros(NO_tests)
    data_return_test = np.zeros((NO_tests, NO_channels, Window_Length))

    NO_valid_trial_train = 0
    NO_valid_trial_test = 0

    a_train = sio.loadmat(root_path + 'A0' + str(subject) + 'T.mat')
    a_test = sio.loadmat(root_path + 'A0' + str(subject) + 'E.mat')

    def process_data(a, data_return, class_return, NO_valid_trial):
        a_data = a['data']
        for ii in range(0, a_data.size):
            a_data1 = a_data[0, ii]
            a_data2 = [a_data1[0, 0]]
            a_data3 = a_data2[0]
            a_X = a_data3[0]
            a_trial = a_data3[1]
            a_y = a_data3[2]
            a_artifacts = a_data3[5]

            for trial in range(0, a_trial.size):
                data_return[NO_valid_trial, :, :] = np.transpose(
                    a_X[int(a_trial[trial]):(int(a_trial[trial]) + Window_Length), :NO_channels])
                class_return[NO_valid_trial] = int(a_y[trial])
                NO_valid_trial += 1

        return data_return, class_return, NO_valid_trial

    data_return_train, class_return_train, NO_valid_trial_train = process_data(a_train, data_return_train, class_return_train, NO_valid_trial_train)
    data_return_test, class_return_test, NO_valid_trial_test = process_data(a_test, data_return_test, class_return_test, NO_valid_trial_test)

    dataT = data_return_train[0:NO_valid_trial_train, :, :]
    labelsT = class_return_train[0:NO_valid_trial_train] - 1
    dataE = data_return_test[0:NO_valid_trial_test, :, :]
    labelsE = class_return_test[0:NO_valid_trial_test] - 1

    return dataT, labelsT, dataE, labelsE
