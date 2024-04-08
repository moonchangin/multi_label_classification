import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import roc_auc_score, average_precision_score
from data_utils import BreastCancerDataset
from models import STLModel
import time

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
# dataset labels for main task only
original_labels = dataset.labels
modified_labels = original_labels[:, 0]
modified_labels = modified_labels.reshape(-1, 1)
# Assign the modified labels back to the dataset's labels attribute
dataset.labels = modified_labels
# dataset.test_indices contains the indices for the main task test set
test_indices = dataset.test_indices # Test indices for the main task (30%)
# Indices in for only the main task
max_indices = len(pd.read_csv(file_paths[0]))
main_task_indices = np.arange(max_indices)  # All indices in the dataset

# Exclude test indices to find the remaining ones for train/validation
remaining_indices = np.setdiff1d(main_task_indices, test_indices)

# Shuffle the remaining indices to ensure randomness
np.random.seed(42)  # Ensure reproducibility
np.random.shuffle(remaining_indices)

# Calculate the split size for 50% train, 50% validation
num_remaining = len(remaining_indices)
num_train = num_remaining // 2

# Split indices for training and validation
train_indices = remaining_indices[:num_train]
val_indices = remaining_indices[num_train:]
# Create subset datasets for train, validation, and test
train_subset = torch.utils.data.Subset(dataset, train_indices)
val_subset = torch.utils.data.Subset(dataset, val_indices)
test_subset = torch.utils.data.Subset(dataset, test_indices)

# Create DataLoaders for train, validation, and test sets
train_loader = DataLoader(train_subset, batch_size=100, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=100, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=9999 , shuffle=False) # Batch size is the size of the test set

# Instantiate the model
num_features = dataset.features.shape[1]
# num_labels = len(file_paths)
num_labels = 1 # we will only use the first label for this example
# Model, loss, and optimizer
model = STLModel(num_features, num_labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1)
# Add L2 regularization via weight_decay
optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4)
# Define a learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Training loop
# Early stopping parameters
patience = 10  # Number of epochs to wait for improvement before stopping
best_val_auroc = 0.0  # Best validation AUROC seen so far
epochs_without_improvement = 0  # Tracks epochs without improvement
best_model_state = copy.deepcopy(model.state_dict())  # To save the best model state
num_epochs = 100
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        label = labels[:, 0].unsqueeze(1)  # Use only the first label
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    # Validation
    model.eval()  # Set model to evaluation mode
    val_targets = []
    val_outputs = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            label = labels[:, 0].unsqueeze(1)  # Use only the first label
            outputs = model(inputs)
            val_outputs.extend(outputs.sigmoid().cpu().numpy())
            val_targets.extend(label.cpu().numpy())

    # Calculate validation metrics
    val_auroc = roc_auc_score(val_targets, val_outputs, average='macro')
    print(f'Epoch {epoch+1}, Validation AUROC: {val_auroc:.4f}')

    # Early Stopping Check
    if val_auroc > best_val_auroc:
        best_val_auroc = val_auroc
        epochs_without_improvement = 0
        best_model_state = copy.deepcopy(model.state_dict())  # Save the best model state
        print(f"Validation AUROC improved to {val_auroc:.4f}, saving model...")
    else:
        epochs_without_improvement += 1
        print(f"Validation AUROC did not improve. Patience: {epochs_without_improvement}/{patience}")
        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break
# Test evaluation
test_targets = []
test_outputs = []
with torch.no_grad():
    model.eval()  # Ensure model is in evaluation mode
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        label = labels[:, 0].unsqueeze(1)  # Use only the first label
        outputs = model(inputs)
        test_outputs.extend(outputs.sigmoid().cpu().numpy())
        test_targets.extend(label.cpu().numpy())

# Calculate test metrics
test_auroc = roc_auc_score(test_targets, test_outputs, average='macro')  # Adjust as necessary
test_aupr = average_precision_score(test_targets, test_outputs, average='macro')  # Adjust as necessary
print(f'Test AUROC: {test_auroc:.4f}, Test AUPR: {test_aupr:.4f}')