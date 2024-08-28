import matplotlib.pyplot as plt
import numpy as np
import mne
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs

from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.io import concatenate_raws, read_raw_edf
from epochsMaker import import_EEG, EEG_to_epochs
from epochToRaw import epoch_to_raw
from rawMaker import EEG_to_raw

#데이터 epoch 불러오기
file_name = '[HRS]MI_four_1.txt'
eeg_array, label_array = import_EEG(file_name)
epochs = EEG_to_epochs(eeg_array, label_array)
raw = EEG_to_raw(eeg_array, label_array, 0)

#라벨링
# 0: 오른손, 1: 휴식, 2: 왼손, 3: 양발
#labels = np.array([0, 1, 2, 3] * (epochs_data[0] // 4))
labels = np.array([i % 4 for i in range(len(epochs))])
#labels = epochs.events[:, -1] - 2

#epoch를 원하는 배열로 다시 생성
# Create MNE info structure
'''n_channels = 32
sfreq = 500  # Set this to your actual sampling frequency
ch_names = [f'EEG {i+1}' for i in range(n_channels)]
ch_types = ['eeg'] * n_channels
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
epochs = mne.EpochsArray(epochs_data, info, tmin=0, events=None)'''

# Define a monte-carlo cross-validation generator (reduce variance)
#
'''epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)
epochs_data = epochs.get_data(copy=False)
epochs_data_train = epochs_train.get_data(copy=False)
cv = ShuffleSplit(10, test_size=0.2, random_state=42)
cv_split = cv.split(epochs_data)

# Assemble a classifier
lda = LinearDiscriminantAnalysis()
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

# Use scikit-learn Pipeline with cross_val_score function
clf = Pipeline([("CSP", csp), ("LDA", lda)])
scores = cross_val_score(clf, epochs_data, labels, cv=cv, n_jobs=None)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1.0 - class_balance)
print(f"raw, Classification accuracy: {np.mean(scores)} / Chance level: {class_balance}")

# plot CSP patterns estimated on full data for visualization
csp.fit_transform(epochs_data, labels)

csp.plot_patterns(epochs.info, ch_type="eeg", units="Patterns (AU)", size=1.5)'''

########################################
#주파수 filter 후
# 필터링
epochs_filtered = epochs.copy().filter(l_freq=0.5, h_freq=40.0) #0.6625
#epochs_filtered = epochs.copy().filter(l_freq=8, h_freq=30.0)
'''epochs_filter_data = epochs_filtered.get_data(copy=False)
cv = ShuffleSplit(10, test_size=0.2, random_state=42)
cv_split = cv.split(epochs_filter_data)

# Assemble a classifier
lda = LinearDiscriminantAnalysis()
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

# Use scikit-learn Pipeline with cross_val_score function
clf = Pipeline([("CSP", csp), ("LDA", lda)])
scores = cross_val_score(clf, epochs_filter_data, labels, cv=cv, n_jobs=None)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1.0 - class_balance)
print(f"filtered, Classification accuracy: {np.mean(scores)} / Chance level: {class_balance}")

# plot CSP patterns estimated on full data for visualization
csp.fit_transform(epochs_data, labels)

csp.plot_patterns(epochs.info, ch_type="eeg", units="Patterns (AU)", size=1.5)'''

########################################
##ICA 후
# Define a monte-carlo cross-validation generator (reduce variance)
#
filt_raw = epochs_filtered.copy().filter(l_freq=1.0, h_freq=None)
ICA
ica = ICA(n_components=15, max_iter="auto", random_state=97)
ica.fit(epochs_filtered)
ica

ica.exclude = [0, 1, 3, 4, 8, 9, 14]
reconst_raw = epochs_filtered.copy()
ica.apply(reconst_raw)
#raw.plot(n_channels=32)
#reconst_raw.plot(n_channels=32)
#del reconst_raw


icaEpochs_data = reconst_raw.get_data(copy=False)
cv = ShuffleSplit(10, test_size=0.2, random_state=42)
cv_split = cv.split(icaEpochs_data)

# Assemble a classifier
lda = LinearDiscriminantAnalysis()
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

# Use scikit-learn Pipeline with cross_val_score function
clf = Pipeline([("CSP", csp), ("LDA", lda)])
scores = cross_val_score(clf, icaEpochs_data, labels, cv=cv, n_jobs=None)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1.0 - class_balance)
print(f"ica, Classification accuracy: {np.mean(scores)} / Chance level: {class_balance}")

# plot CSP patterns estimated on full data for visualization
csp.fit_transform(icaEpochs_data, labels)

csp.plot_patterns(reconst_raw.info, ch_type="eeg", units="Patterns (AU)", size=1.5)

















