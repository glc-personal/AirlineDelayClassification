import multiprocessing as mp
import pandas as pd
import numpy as np
# PyTorch
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler

# GLC Modules
import prepocess as pp

class AirlineDataset(Dataset):
    def __init__(self, csv_fname, cores=1, jobspercore=1, preprocess=True, transform=None):
        '''
        Args:
            csv_fname (string): filename of the airline data csv.
            transform (callable, optional): Optional transform to be applied on a sample.
        '''
        self._index_delay = 5
        self.cores = cores
        self.jobspercore = jobspercore
        self.preprocess = preprocess
        self.pool = mp.Pool(processes=self.cores)
        if self.preprocess == True:
            self.data = pp.preprocess(csv_fname, n=self.cores*self.jobspercore, pool=self.pool)
        else:
            df = pd.read_csv(csv_fname)
            self.data = Tensor(df.values)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        flight = self.data[i,:].tolist()
        label = flight.pop(self._index_delay)
        sample = {'flight': flight, 'delay': label}
        return sample

    def split_to_be_deprecated(self, by=0.8, shuffle=True, random_seed=42):
        '''
        Args:
            split (float): split amount, default is 80% of the data goes in the first, and 20% in the second.
        '''
        # Get the size of the Dataset.
        size = len(self)

        # Get a list of indices with size equal to the number of entries in the Dataset.
        indices = list(range(size))
        split = int(np.floor(by * size))

        # Shuffle the indices if requested.
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        # Split into two sets.
        first_indices, second_indices = indices[split:], indices[:split]

        # Create a subset samplers for the two Datasets.
        first_sampler = SubsetRandomSampler(first_indices)
        second_sampler = SubsetRandomSampler(second_indices)
        return first_sampler, second_sampler

    def load(self, split=True, percentages=[80, 10, 10], shuffle=False, random_seed=42, batch_size=1):
        '''
        Args:
            split (bool): split the data if needed.
            percentages (list): list of percentages to be split into.
            shuffle (bool): shuffle the data.
            random_seed (int): seed for randomization of data if needed.
            batch_size (int): batch size for the DataLoader objects.
        '''
        if split:
            # Assert that percentages adds up to 100%.
            assert int(sum(percentages)) == 100, f"Percentages must add together to 100%, got {int(sum(percentages))}"

            # Create an empty list with the correct number of entries for how much the user wants to split the dataset.
            split_indices = [0 for i in percentages]
            samplers = [0 for i in percentages]
            dataloaders = [0 for i in percentages]
            
            # Get the size of the Dataset.
            size = len(self)

            # Create a list of indices for each entry in the Dataset.
            indices = list(range(size))

            # Shuffle the indices if desired.
            if shuffle:
                np.random.seed(random_seed)
                np.random.shuffle(indices)

            # Split the data into Samplers and DataLoaders.
            last_index = 0
            for i in range(len(percentages)):
                # Split the indices
                split_indices[i] = indices[last_index:(last_index + int(np.floor(0.01 * percentages[i] * size)))]
                last_index = last_index + int(np.floor(0.01 * percentages[i] * size))

                # Create Samplers for each new set.
                samplers[i] = SubsetRandomSampler(split_indices[i])

                # Create a DataLoader for each new set.
                dataloaders[i] = DataLoader(self, batch_size=batch_size, sampler=samplers[i])

            # Rerturn the DataLoaders list.
            return dataloaders

        else:
            dataloader = DataLoader(self, batch_size=batch_size, shuffle=shuffle)
            return dataloader
