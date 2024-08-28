import json
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
from arrowMaker import add_arrows
from epochsMaker import import_EEG, EEG_to_epochs
from epochToRaw import epoch_to_raw
from rawMaker import EEG_to_raw

channels = ['F5', 'FC5', 'C5', 'CP5', 'P5', 'FC3', 'C3', 'CP3', 'P3', 'F1', 'FC1', 'C1', 'CP1', 'P1', 'Cz', 'CPz',
            'Pz', 'F2', 'FC2', 'C2', 'CP2', 'P2', 'FC4', 'C4', 'CP4', 'P4', 'F6', 'FC6', 'C6', 'CP6', 'P6']

file_name = '[HRS]MI_four_1.txt'
eeg_array, label_array = import_EEG(file_name)
epochs = EEG_to_epochs(eeg_array, label_array)
raw = EEG_to_raw(eeg_array, label_array, 0)
#raw.plot(picks=[10])
#epochs[0].plot(n_channels=32)
#raw.plot(n_channels=32)

# Plotting the epochs

#epochs.plot(duration=60, proj=False, n_channels=32, remove_dc=False)
#epochs.plot(n_channels=32)

#Slow drift filtering
'''for cutoff in (0.1, 0.2):
    raw_highpass = epochs.copy().filter(l_freq=cutoff, h_freq=None)
    with mne.viz.use_browser_backend("matplotlib"):
        fig = raw_highpass.plot(
            n_channels=32
        )
    fig.subplots_adjust(top=0.9)
    fig.suptitle(f"High-pass filtered at {cutoff} Hz", size="xx-large", weight="bold")'''
    

#arrow marker
'''fig = raw.compute_psd(fmax=250).plot(
    average=True, amplitude=False, picks="data", exclude="bads"
)
add_arrows(fig.axes[:2])'''

#notch filtering
'''meg_picks = mne.pick_types(raw.info, eeg=True)
freqs = (60, 120, 180, 240)
raw_notch = raw.copy().notch_filter(freqs=freqs, picks=meg_picks)
for title, data in zip(["Un", "Notch "], [raw, raw_notch]):
    fig = data.compute_psd(fmax=250).plot(
        average=True, amplitude=False, picks="data", exclude="bads"
    )
    fig.suptitle(f"{title}filtered", size="xx-large", weight="bold")
    add_arrows(fig.axes[:2])'''

    
#Down sampling
'''raw_downsampled = epochs.copy().resample(sfreq=300)
# choose n_fft for Welch PSD to make frequency axes similar resolution
n_ffts = [4096, int(round(4096 * 200 / epochs.info["sfreq"]))]
fig, axes = plt.subplots(2, 1, sharey=True, layout="constrained", figsize=(10, 6))
for ax, data, title, n_fft in zip(
    axes, [epochs, raw_downsampled], ["Original", "Downsampled"], n_ffts
):
    fig = data.compute_psd(method='welch', n_fft=n_fft).plot(
        average=True, amplitude=False, picks="data", exclude="bads", axes=ax
    )
    ax.set(title=title, xlim=(0, 300))'''
    

'''reconst_epoch = epochs.copy()
ica.apply(reconst_epoch)
regexp = r"(MEG [12][45][123]1|EEG 00.)"
artifact_picks = mne.pick_channels_regexp(epochs.ch_names, regexp=regexp)
epochs.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False)
reconst_epoch.plot(
    order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False
)
del reconst_epoch'''

#ica
'''filt_raw = epochs.copy().filter(l_freq=1.0, h_freq=None)
ICA
ica = ICA(n_components=15, max_iter="auto", random_state=97)
ica.fit(filt_raw)
ica'''

###data distribution by ICA
'''explained_var_ratio = ica.get_explained_variance_ratio(filt_raw)
for channel_type, ratio in explained_var_ratio.items():
    print(
        f"Fraction of {channel_type} variance explained by all components: " f"{ratio}"
    )'''

###brain distribution field
#ica.plot_components()

###graph?
#epochs[36].load_data()
#ica.plot_sources(epochs[36], show_scrollbars=False)

###overlay
'''filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None)
ICA
ica = ICA(n_components=15, max_iter="auto", random_state=97)
ica.fit(filt_raw)
ica

raw.load_data()
ica.plot_sources(raw, show_scrollbars=False)'''
#ica.plot_components()
#raw = epochs.average().to_raw()
# blinks
#ica.plot_overlay(raw, exclude=[1,2,4], picks="eeg")
# right hand
#ica.plot_overlay(raw, exclude=[7], picks="0")

###various graphs
#ica.plot_properties(raw, picks=[0,1,2,3,4,5,6,7])

#regraph without exclude
'''ica.exclude = [0, 1]
reconst_raw = raw.copy()
ica.apply(reconst_raw)
raw.plot(n_channels=32)
reconst_raw.plot(n_channels=32)
del reconst_raw'''


#finding component automatically
'''ica.exclude = []
raw1 = EEG_to_raw(eeg_array, label_array, 1)
# find which ICs match the ECG pattern
ecg_indices, ecg_scores = ica.find_bads_eog(raw, ch_name="C1")
corrmap(raw1, template=(0, ecg_indices[0]))
ica.exclude = ecg_indices

for index, (ica, raw) in enumerate(zip(raw1, raw)):
    with mne.viz.use_browser_backend("matplotlib"):
        fig = ica.plot_sources(raw, show_scrollbars=False)
    fig.subplots_adjust(top=0.9)  # make space for title
    fig.suptitle(f"Subject {index}")
# barplot of ICA component "ECG match" scores
ica.plot_scores(ecg_scores)
# plot diagnostics
ica.plot_properties(raw, picks=ecg_indices)
# plot ICs applied to raw data, with ECG matches highlighted
ica.plot_sources(raw, show_scrollbars=False)
# plot ICs applied to the averaged ECG epochs, with ECG matches highlighted
#ica.plot_sources(ecg_evoked)'''



#epochs.plot(n_channels=32)
#raw_highpass[0].plot(n_channels=2)
evoked = epochs[0].average()
evoked.plot_joint(picks="eeg")
evoked.plot_topomap(times=[0., 0.08, 0.1, 0.12, 0.2], ch_type='eeg')
plt.show()
#epochs.plot(n_channels=32, scalings='auto', title='EEG Epochs', show=True, block=True)
#epochs.plot_image(n_channels=32, scalings='auto', title='EEG Epochs', show=True, block=True)