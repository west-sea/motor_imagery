import numpy as np
import mne

def EEG_to_raw(eeg_array, label_array, n, sfreq=500):
    channels = ['F5', 'FC5', 'C5', 'CP5', 'P5', 'FC3', 'C3', 'CP3', 'P3', 'F1', 'FC1', 'C1', 'CP1', 'P1', 'Cz', 'CPz',
                'Pz', 'F2', 'FC2', 'C2', 'CP2', 'P2', 'FC4', 'C4', 'CP4', 'P4', 'F6', 'FC6', 'C6', 'CP6', 'P6']

    n_channels = len(channels)
    ch_types = ['eeg'] * n_channels
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=channels, sfreq=sfreq, ch_types=ch_types)
    info.set_montage(montage)

    # Select the data for the n-th trial
    data = eeg_array[n]
    
    # Transpose data to (channels, samples) assuming your data is already in (y, z)
    #data = np.transpose(data, (1, 0))
    
    # Check if the data length matches the number of channels
    assert data.shape[0] == len(channels), f"Number of channels in data ({data.shape[0]}) does not match len(channels) ({len(channels)})"

    # Create raw object
    raw = mne.io.RawArray(data, info)

    return raw
