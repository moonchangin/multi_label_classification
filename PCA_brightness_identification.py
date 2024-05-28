import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Load one file to examine its structure
file_path = 'brightness_input/GSE164458/GSE164458_paclitaxel.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
df.head()
# List of the gene symbols from the Oncotype DX 21 gene panel
genes_21 = ['ESR1', 'PGR', 'ERBB2', 'MKI67', 'AURKA', 'BIRC5', 'CCNB1', 'MYBL2', 'MMP11', 'GSTP1', 'BAG1', 'GAPDH', 'TFRC', 'GUSB', 'ACTB', 'BCL2', 'SCUBE2', 'BAG1', 'ACTB', 'GAPDH', 'TFRC']
# Subsetting the dataframe to include only the relevant gene columns
# We need to adjust gene symbols if they are represented differently in the dataset
# Some genes might have alternative names or may not be present

# Adjusting gene symbols to match the dataset
adjusted_genes = {
    "ERBB2": "HER2",  # As seen in the head of the dataframe, HER2 might be used instead of ERBB2
    "STK15": "AURKA",  # Adjusting based on common alternative names
    "Survivin": "BIRC5",  # Using the gene symbol instead of common name
    # Note: For some genes, if they are not found directly, we may need to check for alternative representations
}

# Try to find the exact match or the adjusted match in the dataframe columns
subset_genes = [gene if gene in df.columns else adjusted_genes.get(gene, gene) for gene in genes_21]

# Filter out genes that are not found in the dataframe
valid_genes = [gene for gene in subset_genes if gene in df.columns]

# Creating a subset dataframe with the valid genes found
df_subset = df[['Sample'] + valid_genes]  # Including 'Sample' for reference

print(df_subset.head(), f"Valid genes found: {len(valid_genes)} out of {len(genes_21)}", valid_genes)