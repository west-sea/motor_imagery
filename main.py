from notchFilterEpoch import import_EEG, filter_EEG, EEG_to_epochs

def main(file_name):
    # Import EEG data
    eeg_array, label_array = import_EEG(file_name)
    
    # Define sampling frequency and channel names
    sfreq = 500  # Example sampling frequency
    ch_names = ['F5', 'FC5', 'C5', 'CP5', 'P5', 'FC3', 'C3', 'CP3', 'P3', 'F1', 'FC1', 'C1', 'CP1', 'P1', 'Cz', 'CPz',
                'Pz', 'F2', 'FC2', 'C2', 'CP2', 'P2', 'FC4', 'C4', 'CP4', 'P4', 'F6', 'FC6', 'C6', 'CP6', 'P6']

    # Apply notch filtering
    filtered_eeg_array = filter_EEG(eeg_array, sfreq, ch_names)
    
    # Convert filtered EEG data to epochs
    epochs = EEG_to_epochs(filtered_eeg_array, label_array, sfreq=sfreq)
    
    # Plot the first epoch for visualization
    epochs[0].plot(n_channels=len(ch_names))
    
    return epochs

if __name__ == "__main__":
    file_name = '[HRS]MI_four_3.txt'
    main(file_name)
