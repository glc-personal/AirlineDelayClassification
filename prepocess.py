# Preprocess: prepare the data for PyTorch.

import multiprocessing as mp
import pandas as pd
import numpy as np
# PyTorch
from torch import Tensor
from torch.utils.data import DataLoader

def preprocess(csv_fname, n, pool):
    # Read in the csv file.
    df = pd.read_csv(csv_fname)

    # Split the DataFrame into n smaller DataFrames.
    dfs = []
    dfs = _split_dataframe(df, n)

    # One-hot encode the original data.
    #df_one_hot_encoded = _one_hot_encode(df)
    
    dfs_one_hot_encoded = pool.map(_one_hot_encode, dfs)
    print(len(dfs_one_hot_encoded))

    # Convert to a PyTorch Tensor object.
    #x = Tensor(df_one_hot_encoded.values)
    #return x

def _split_dataframe(df, n):
    dfs = []
    df_size = len(df)
    nchunks = np.floor(df_size / float(n))

    # Split the DataFrame into n DataFrames.
    iend = 0
    for i in range(n):
        istart = int(iend)
        iend = int(nchunks * (i + 1))
        dfs.append(df.iloc[istart:iend,:])
    return dfs

def _one_hot_encode(df):
    '''
    One-Hot Encodes a Dataframe object for given column names.
    '''
    arr_col_names = ['Airline', 'AirportFrom', 'AirportTo']

    # Get the number of entries in the Dataframe.
    n_entries = len(df)

    # Initialize the one-hot encoded Dataframe object.
    df_one_hot_encoded = df

    # For each column determine the number of categories.
    for col_name in arr_col_names:
        # Get the unique categories.
        arr_unique_categories = df[col_name].unique()

        # Get the number of unique categories.
        n_unique_categories = len(arr_unique_categories)

        # Create a Dataframe object for the categories.
        for category in arr_unique_categories:
            if col_name == 'AirportFrom':
                appendage = '_FROM'
            elif col_name == 'AirportTo':
                appendage = '_TO'
            else:
                appendage = ''
        df_categories = pd.DataFrame(index=np.arange(0, n_entries), columns=[(category + appendage) for category in arr_unique_categories]).fillna(0)

        # Fill the categories Dataframe according to the original Dataframe.
        i = 0
        for value in df[col_name]:
            # One-hot encode the categories Dataframe.
            for category in df_categories:
                if (value + appendage) == category:
                    df_categories.loc[i, category] = 1.0
            i = i + 1

        # Remove the column from the original data.
        del df_one_hot_encoded[col_name]

        # Concat the one-hot encoded Dataframe objects to the original data.        
        df_one_hot_encoded = pd.concat([df_one_hot_encoded, df_categories], axis=1, ignore_index=False, sort=False)
    return df_one_hot_encoded
