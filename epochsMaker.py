import numpy as np
import json
import mne

def import_EEG(file_name):
    """
    :param file_name: Path to the text file containing EEG data
    :return: EEG_array (numpy array: epoch * timestamp * channel), label_array (numpy array of labels)
    """
    with open(file_name, 'r') as f:
        lines = f.readlines()
    
    label_type = '0123'
    EEG_array = []
    label_array = []

    for l in lines:
        l = l.strip()
        if l in label_type:
            label_array.append(np.int64(l))
        else:  # append the data for the label
            data = json.loads(l)
            EEG_array.append(np.float64(data))

    EEG_array = np.array(EEG_array) * (10**(-8))
    label_array = np.array(label_array)
    EEG_array = np.transpose(EEG_array, (0, 2, 1))
    
    return EEG_array, label_array

def print_list_structure(lst, depth=0):
    if isinstance(lst, list):
        print(f"{'  ' * depth}[\n")
        for item in lst:
            print_list_structure(item, depth + 1)
        print(f"{'  ' * depth}]")
    else:
        print(f"{'  ' * depth}{lst}")

def EEG_array_modifier(eeg_array, label_array):
    """
    Receives the EEG array and the label array to modify into a more suitable form to switch it into EpochsArray

    :param eeg_array: (numpy array) EEG data of each trial
    :param label_array: (numpy array) Labels of each trial
    :return: Modified EEG array and events array
    """
    X, y, event_timepoints = [], [], []
    for i, label in enumerate(label_array):
        X.append(np.array(eeg_array[i]))
        y.append(label)
        event_timepoints.append(i)
    
    events_array = np.array([[event_timepoints[i], 0, y[i]] for i in range(len(y))])
    return np.array(X), events_array

def EEG_to_epochs(eeg_array, label_array, sfreq=500, event_id={'Rest': 0, 'Right Hand': 1, 'Left Hand': 2, 'Feet': 3}):
    channels = ['F5', 'FC5', 'C5', 'CP5', 'P5', 'FC3', 'C3', 'CP3', 'P3', 'F1', 'FC1', 'C1', 'CP1', 'P1', 'Cz', 'CPz',
                'Pz', 'F2', 'FC2', 'C2', 'CP2', 'P2', 'FC4', 'C4', 'CP4', 'P4', 'F6', 'FC6', 'C6', 'CP6', 'P6']

    n_channels = len(channels)
    ch_types = ['eeg'] * n_channels
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=channels, sfreq=sfreq, ch_types=ch_types)
    info.set_montage(montage)
    data, events = EEG_array_modifier(eeg_array, label_array)
    epochs = mne.EpochsArray(data, info, events, tmin=0, event_id=event_id)

    return epochs


