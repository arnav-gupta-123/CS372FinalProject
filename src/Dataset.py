#Load the data and preprocess it for training and testing. 

import mne
import numpy as np
import warnings
import glob
from scipy.stats import ttest_ind
from typing import List, Optional
from sklearn.model_selection import cross_val_score, train_test_split
from scipy.stats import t
from sklearn.metrics import roc_auc_score as get_auc

#shut off warnings and mne logs 
warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")

class Dataset:
    def __init__(self,
        glob_path: str, 
        batch_size: int = 1,
        notch_filter: Optional[float] = None,
        sample_rate: Optional[int] = None,
        important_channels: Optional[List[str]] = None,
        use_car: bool=True,
        tmin: float = 0.0,
        tmax: float = 0.8):
            self.glob_path: str = glob_path
            self.batch_size: int = batch_size
            self.notch_filter: Optional[float] = notch_filter
            self.sample_rate: Optional[int] = sample_rate
            self.important_channels: Optional[List[str]] = important_channels
            self.data_paths: List[str] = glob.glob(glob_path)
            self.tmin = tmin
            self.tmax = tmax
            self.use_car=use_car
            self.current_target_events = None
            self.character_channels = None
            self.target_string = None
   

    def _load(self, path: str) -> mne.io.BaseRaw:
        raw: mne.io.BaseRaw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        return raw

    def _process(self, raw: mne.io.BaseRaw) -> mne.Epochs:
        # Find stimulus and target stimulus events
        self.stim_events = mne.find_events(raw, stim_channel='StimulusBegin', verbose=False)
        self.targstim_events = mne.find_events(raw, stim_channel='StimulusType', verbose=False)
        
        # Label Target and non-target epochs
        targstim_indices = np.isin(self.stim_events[:, 0], self.targstim_events[:, 0])
        self.stim_events[~targstim_indices, 2] = 0
        self.current_target_events = mne.find_events(raw, stim_channel='CurrentTarget', verbose=False)

        # Pick EEG channels if not set
        if self.important_channels is None:
            self.important_channels = mne.pick_channels_regexp(raw.info['ch_names'], 'EEG')
            
        #handle outliers

        # Apply notch filter and bandpass filter
        if self.notch_filter:
            raw.notch_filter(freqs=self.notch_filter, picks=self.important_channels, verbose=False)
            raw.filter(l_freq=0.5, h_freq=30.0, picks=self.important_channels, verbose=False)

        # Define event_id mapping
        event_dict = {'target': 1, 'non_target': 0}

    
        self.character_channels = [
            ch for ch in raw.info['ch_names']
            if "_" in ch and ch.split("_")[-1].isdigit() and ch.split("_")[-2].isdigit()
        ]
        if self.current_target_events is not None and len(self.current_target_events) > 0:
            letters = self.current_target_events[:, 2]

            if self.character_channels and len(self.character_channels) > 0:
                try:
                    self.target_string = "".join(
                        self.character_channels[code - 1][0] for code in letters
                    )
                except Exception:
                    self.target_string = None
            else:
                self.target_string = None

        # Create epochs
        epochs = mne.Epochs(
            raw, self.stim_events, tmin=self.tmin, tmax=self.tmax,
            event_id=event_dict, preload=True, baseline=None,
           #reject=reject_criteria,
            verbose=False, proj=False, picks=self.important_channels
        )

        if self.use_car:
            # Use MNE's CAR
            epochs.set_eeg_reference('average', projection=True)
            epochs.apply_proj()
            #print(f"Applied CAR ")

        # Resample
        if self.sample_rate:
            # Store original sample rate for conversion
            original_sfreq = epochs.info['sfreq']
            epochs.resample(self.sample_rate, verbose=False)

            # Scale epoch event times to match resampled data
            resample_ratio = self.sample_rate / original_sfreq
            expected_max_time = int(raw.n_times * resample_ratio)
            
            # Check if event times are still in original scale
            if epochs.events[:, 0].max() > expected_max_time * 2:  # Allow some margin
                # Store the scaling factor for later use
                self.event_time_scale_factor = resample_ratio
            else:
                self.event_time_scale_factor = 1.0

        return epochs

    def _extract(self, epochs: mne.Epochs) -> tuple[np.ndarray, np.ndarray]:
        X = epochs.get_data()
        y = epochs.events[:, 2]
        return X, y

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        path: str = self.data_paths[idx]
        raw = self._load(path)
        epochs = self._process(raw)
        
        # Store raw and epochs for later use
        self.raw = raw
        self.epochs = epochs
        
        return self._extract(epochs)

    def plot_raw(self, idx: int = 0):
        raw = self._load(self.data_paths[idx])

        picks = self.important_channels
        if picks is None:
            picks = mne.pick_channels_regexp(raw.info['ch_names'], 'EEG')
        elif isinstance(picks[0], int):
            picks = [raw.info['ch_names'][i] for i in picks]

        raw.plot(picks=picks)