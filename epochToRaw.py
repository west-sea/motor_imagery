import mne
import numpy as np
def epoch_to_raw(epochs, sfreq, ch_names):
    """
    Convert the first epoch in the epochs array to a RawArray object.
    
    :param epochs: numpy array of shape (n_epochs, n_channels, n_times) representing the epochs data
    :param sfreq: Sampling frequency of the data
    :param ch_names: List of channel names
    :return: mne.io.RawArray object
    """
    ##if epochs.ndim != 3:
     #   raise ValueError(f"Expected a 3D array, but got shape {epochs.shape}")
    
    # Extract the first epoch (shape: (n_channels, n_times))
    first_epoch = epochs[0]
    
    # Ensure the extracted epoch is 2D
    #if first_epoch.ndim != 2:
     #   raise ValueError(f"Expected a 2D array after extraction, but got shape {first_epoch.shape}")
    
    # Create RawArray object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'] * len(ch_names))
    raw = mne.io.RawArray(first_epoch, info)
    return raw