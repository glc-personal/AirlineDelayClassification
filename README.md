# AirlineDelayClassification
Using the Airline Delay Dataset from Kaggle to better understand and learn Machine Learning with PyTorch. 

# (Kaggle) - Airline Dataset
  Description: Airlines dataset has 539383 instances and 8 different features
  Link: https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay
  
# Goals:
  - Predict weather or not a flight will be delayed or not.
  - Build / Train a model for this prediction of flight delays.
  - Use parallel processing (multiprocessing module) for CPU optimization.
  
 # AirlineDataset:
  - torch.utils.data.Dataset subclass
  - Args:
    - csv_fname (string): name of the csv file containing the airline data.
    - preprocess (bool): determines whether or not to preprocess the airline data for training or not.
    - transform (optional): transformations functions to be applied on a sample if needed.
  - __getitem__
    - Args:
      - i (int): row index to return the desired flight data and label.
    - Output:
      - sample (dict): returns a dictionary with the 'flight' data (list) and the 'label' data which is the delay (int)
  - load:
    - Description: 
    - Args:
