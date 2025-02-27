import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import mne
import os 
import numpy as np
from sklearn.model_selection import KFold, train_test_split

class EEGSleepDataset(Dataset):
    def __init__(self, 
                 data_path, 
                 split=0,
                 n_samples = None,
                 dataset = 'train', # 'train', 'val', 'test'
                 use_edf_20 = False,
                 epoch_length=30, 
                 sample_rate=100, 
                 channels=['EEG Fpz-Cz', 'EEG Pz-Oz'], 
                 normalize = False):
        """
        Initialize the dataset.

        Parameters:
        - data_path: str, path to the top-level data folder
        - subjects: list, list of subject identifiers to include (e.g., ["SC401", "SC402"])
        - epoch_length: int, length of each epoch in seconds
        - sample_rate: int, sampling rate to resample the EEG data
        """
        # create splits
        subjs = os.listdir(data_path)
        subjs = [s for s in subjs if not s.startswith('.')]
        if use_edf_20:
            subjs = np.sort(subjs)
            subjs = subjs[:20]
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(kfold.split(subjs))
        train_subjs = [subjs[i] for i in splits[split][0]]
        test_subjects = [subjs[i] for i in splits[split][1]]
        # split subjects into train and validation
        train_subjs, val_subjs = train_test_split(train_subjs, test_size=0.1, random_state=42)
        
        if dataset == 'train':
            subjects = train_subjs
        elif dataset == 'val':
            subjects = val_subjs
        elif dataset == 'test':
            subjects = test_subjects

        self.data_path = Path(data_path)
        self.subjects = subjects
        self.epoch_length = epoch_length
        self.sample_rate = sample_rate
        self.data = []
        self.labels = []
        self.subject_id = []
        self.channels = channels
        self.normalize = normalize
        
        self._load_data()
        self.std = np.std(np.concatenate(self.data))
        
        if n_samples is not None:
            np.random.seed(42)
            idx = np.random.choice(len(self.data), n_samples, replace=False)
            self.data = [self.data[i] for i in idx]
            self.labels = [self.labels[i] for i in idx]
            self.subject_id = [self.subject_id[i] for i in idx]

    def _load_data(self):
        """Load and preprocess data for the specified subjects."""
        for subject in self.subjects:
            subject_path = self.data_path / subject
            if not subject_path.exists():
                raise FileNotFoundError(f"Subject folder {subject} not found in {self.data_path}")

            fif_files = list(subject_path.glob("*.fif"))
            if not fif_files:
                raise FileNotFoundError(f"No FIF files found for subject {subject}")

            for fif_file in fif_files:
                # Load raw EEG data
                epoched = mne.read_epochs(str(fif_file), preload=True, verbose=False)
                # get only EEG channels
                epoched.pick(self.channels)
                data = epoched.get_data()
                labels = epoched.events[:, -1]
                # standardize data along the time axis
                if self.normalize:
                    data = (data - data.mean(axis=-1, keepdims=True)) / data.std(axis=-1, keepdims=True)
                self.data.extend(data)
                self.labels.extend(labels)
                self.subject_id.extend([subject] * len(data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)
    

def get_dataloader(data_path, train_test_val, split, batch_size=32, shuffle=True, num_workers=0, balanced=True, channels = ['EEG Fpz-Cz', 'EEG Pz-Oz'], normalize = False):
    """
    Create a DataLoader for the EEG sleep dataset.

    Parameters:
    - data_path: str, path to the top-level data folder
    - train_test_val: str, whether to use train, test, or validation data
    - batch_size: int, number of samples per batch
    - shuffle: bool, whether to shuffle the data
    - num_workers: int, number of subprocesses to use for data loading

    Returns:
    - DataLoader instance
    """
    dataset = EEGSleepDataset(data_path, dataset=train_test_val, split = split, channels=channels, normalize=normalize)
    # sample balanced dataset based on labels
    labels = torch.tensor(dataset.labels)
    class_sample_count = torch.tensor(
        [(labels == t).sum() for t in torch.unique(labels, sorted=True)])
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in labels])
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
    if balanced:
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler, persistent_workers=num_workers > 0)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, persistent_workers=num_workers > 0)
    return dataloader       