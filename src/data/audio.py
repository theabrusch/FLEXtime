import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import h5py as h5

class TSloader(Dataset):
    def __init__(self, data, freqs = None, target = None):
        super().__init__()
        data = torch.tensor(data).float()
        self.data = data.unsqueeze(1)
        self.data_length = self.data.shape[2]
        if freqs is None:
            freqs = np.zeros(self.data.shape[0])
        self.freqs = freqs
        if target is not None:
            self.target = torch.tensor(target).float()
        else:
            self.target = None

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        data_1 = self.data[idx,:, :self.data_length//2]
        data_2 = self.data[idx,:, self.data_length//2:]
        if self.target is not None:
            return data_1, data_2, self.freqs[idx], self.target[idx]
        return data_1, data_2, self.freqs[idx]

class AudioNetDataset(Dataset):
    def __init__(self, path, preload, dataset, labeltype = 'digit', splits = [0], subsample = False, seed = 0, add_noise = False, noiselevel = 1):
        self.path = path
        self.preload = preload
        self.dataset = dataset
        self.labeltype = labeltype
        self.splits = splits
        self.subsample = subsample
        self.data = []
        self.digitlabels = []
        self.genderlabels = []
        self.data_paths = self.get_data_paths()
        self.add_noise = add_noise
        self.noiselevel = noiselevel
        if self.subsample:
            np.random.seed(seed)
            np.random.shuffle(self.data_paths)
            self.data_paths = self.data_paths[:subsample]
        self.gender_dict = {'female': 0, 'male':1}
        with open(path + 'data/audioMNIST_meta.txt', 'r') as f:
            self.meta = eval(f.read())
        if preload:
            self.load_data()
            self.std = torch.std(torch.cat(self.data))
       

    def get_data_paths(self):
        if self.labeltype == 'both':
            label = 'gender'
        else:
            label = self.labeltype
        files = glob.glob(f"{self.path}preprocessed_data/AudioNet_{label}_{self.splits}_{self.dataset}.txt")
        data_paths = []
        for file in files:
            with open(file, 'r') as f:
                data_paths += f.read().splitlines()
        return data_paths
    
    def load_data(self):
        for dpath in self.data_paths:
            split_path = dpath.split('/')[-3:]
            datapath = self.path + '/'.join(split_path)
            with h5.File(datapath, 'r') as f:
                data = torch.tensor(f['data'][0])
                if self.add_noise:
                    noise = torch.randn_like(data) * self.noiselevel
                    data += noise
                self.data.append(data)
                lab = torch.tensor(f['label'][0][0])
                self.digitlabels.append(lab)
                subject = datapath.split('_')[-2]
                lab = self.gender_dict[self.meta[subject]['gender']]
                self.genderlabels.append(lab)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        if self.preload:
            if self.labeltype == 'digit':
                return (self.data[idx], self.digitlabels[idx])
            elif self.labeltype == 'gender':
                return (self.data[idx], self.genderlabels[idx])
            else:
                return (self.data[idx], self.digitlabels[idx], self.genderlabels[idx])
        else:
            with h5.File(self.data_paths[idx], 'r') as f:
                return (torch.tensor(f['data'][0]), torch.tensor(f['label'][0][0]))