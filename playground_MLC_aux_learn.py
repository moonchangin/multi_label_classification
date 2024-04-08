import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import roc_auc_score
from data_utils import BreastCancerDataset
from models import MLCModel

# import auxlearn scripts
from task_weighter import task_weighter

# Define the list of genes of interest
genes_21 = ['ESR1', 'PGR', 'ERBB2', 'MKI67', 'AURKA', 'BIRC5', 'CCNB1', 'MYBL2', 'MMP11',
             'GSTP1', 'BAG1', 'GAPDH', 'TFRC', 'GUSB', 'BCL2', 'SCUBE2', 'BAG1', 'ACTB']
# usually it is use genes whatever is available for all datasets

# Define the file paths to your datasets
file_paths = [
    'brightness_input/GSE164458/GSE164458_paclitaxel.csv',
    'brightness_input/GSE164458/GSE164458_carboplatin_paclitaxel.csv',
    'brightness_input/GSE164458/GSE164458_veliparib_carboplatin_paclitaxel.csv'
]

immune_paths = ['immune_profile/immune_GSE164458_paclitaxel.csv',
                'immune_profile/immune_GSE164458_carboplatin_paclitaxel.csv',
                'immune_profile/immune_GSE164458_veliparib_carboplatin_paclitaxel.csv']

# Create the dataset
dataset = BreastCancerDataset(oncotype_file_paths=file_paths, cibersort_file_path= immune_paths, genes_of_interest=genes_21)

# create a DataLoader for batching
# data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Calculate the sizes for training and validation sets (80% - 20% split)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Split the dataset
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

# Create DataLoaders for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Instantiate the model; adjust num_features based on your dataset
num_features =  dataset.features.shape[1]  # Make sure this matches your dataset's feature structure
num_labels = len(file_paths) # Adjust based on the number of files (treatments)
model = MLCModel(num_features, num_labels)
# Assuming you're using a device-agnostic approach for model training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize Task Weighter
task_weighter = task_weighter(num_labels).to(device)
optimizer_task_weighter = optim.SGD(task_weighter.parameters(), lr=0.01)  # Adjust the learning rate as needed

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Original Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        optimizer_task_weighter.zero_grad()

        outputs = model(features)

        # Assuming the first column is the main task and the rest are auxiliary
        main_loss = criterion(outputs[:, 0], labels[:, 0]).unsqueeze(0)  # Make it a single-element tensor

        # Auxiliary losses
        if num_labels > 1:
            aux_losses = torch.stack([criterion(outputs[:, i], labels[:, i]) for i in range(1, num_labels)])
        else:
            # If no auxiliary losses, create a dummy tensor to avoid breaking the computation
            aux_losses = torch.zeros(1, device=device)

        losses = list(torch.cat([main_loss, aux_losses]))

        # Pass the concatenated losses to the task weighter
        weighted_loss = task_weighter(losses)
        weighted_loss.backward()

        optimizer.step()
        optimizer_task_weighter.step()

        total_loss += weighted_loss.item()      
        
        # loss reweighting seems to be working.
        # We still need to implement the gradient variance estimator and the noise injection for every 20 iterations

    print(f'Epoch {epoch+1}/{num_epochs}, Total Loss: {total_loss}')

model.eval()  # Set the model to evaluation mode
y_true = []
y_pred = []

# Collect all labels and predictions
for features, labels in val_loader:
    with torch.no_grad():  # No need to track gradients
        output = model(features)
        y_pred.append(torch.sigmoid(output).numpy())  # Apply sigmoid to get probabilities
        y_true.append(labels.numpy())

# Stack to create a single numpy array for true labels and predictions
y_true = np.vstack(y_true)
y_pred = np.vstack(y_pred)

# Calculate AUROC for each label
auroc_per_label = [roc_auc_score(y_true[:, i], y_pred[:, i]) for i in range(y_pred.shape[1])]

# Calculate the average AUROC across all labels
average_auroc = np.mean(auroc_per_label)

print(f"AUROC per label: {auroc_per_label}")
print(f"Average AUROC: {average_auroc}")