# Airline Delay Classification Problem
# - PyTorch
# - Kaggle - https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay?select=Airlines.csv

# Time
import time
import multiprocessing as mp

# PyTorch
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader

# GLC Modules
from AirlineDataset import AirlineDataset

# Define the model.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

    def forward(self, x):
        logits = x
        return logits

if __name__ == '__main__':
    start = time.time()
    ncores = mp.cpu_count()

    # Instantiate the airline Dataset.
    #dataset = AirlineDataset('short.csv', cores=ncores, jobspercore=10)
    dataset = AirlineDataset('Airlines.csv', cores=ncores, jobspercore=10)

    # Split the data and load into DataLoader objects.
    #train_dataloader, test_dataloader, val_dataloader = dataset.load(percentages=[80,10,10], shuffle=True)

    # Create a model for training with the train_datalaoder *********

    end = time.time()
    print(f"Elapsed Time: {end - start} (s)")
