import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from data_utils import BreastCancerDataset

# Define the list of genes of interest
genes_21 = ['ESR1', 'PGR', 'ERBB2', 'MKI67', 'AURKA', 'BIRC5', 'CCNB1', 'MYBL2', 'MMP11',
             'GSTP1', 'BAG1', 'GAPDH', 'TFRC', 'GUSB', 'BCL2', 'SCUBE2', 'BAG1', 'ACTB']
# usually it is use genes whatever is available for all datasets

# Define the file paths to your datasets
file_paths = [
    'brightness_input/GSE164458/GSE164458_carboplatin_paclitaxel.csv',
    'brightness_input/GSE164458/GSE164458_paclitaxel.csv',
    'brightness_input/GSE164458/GSE164458_veliparib_carboplatin_paclitaxel.csv'
]

# Create the dataset
dataset = BreastCancerDataset(file_paths=file_paths, genes_of_interest=genes_21)

# create a DataLoader for batching
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

print()