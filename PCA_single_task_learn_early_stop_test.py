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
import glob
import copy
import argparse

parser = argparse.ArgumentParser(description='Run model with different seeds.')
parser.add_argument('--mccv', type=int, default=1, help='Seed for the random number generator', required=True)
parser.add_argument('--main_task_file', type=str, default='GSE164458_paclitaxel.csv' ,help='File name for the main task', required=True)
args = parser.parse_args()
def generate_test_indices(first_file_path, test_size_fraction,seed):
        """
        Generate indices for a test set based on the first file.

        Args:
            first_file_path (str): The file path for the first dataset.
            test_size_fraction (float): The fraction of the first file's samples to use for the test set.

        Returns:
            np.array: An array of indices for the test set.
        """
        # Read the first file to get the sample size
        df_first_file = pd.read_csv(first_file_path)
        total_samples = len(df_first_file)
        
        # Calculate the number of test samples
        test_samples_count = int(total_samples * test_size_fraction)
        
        # Generate random indices for the test set
        # Ensure reproducibility with np.random.seed
        np.random.seed(seed)  # Or any seed of your choice
        test_indices = np.random.choice(range(total_samples), size=test_samples_count, replace=False)
        
        return test_indices

# Define the list of genes of interest
genes_21 = ['ESR1', 'PGR', 'ERBB2', 'MKI67', 'AURKA', 'BIRC5', 'CCNB1', 'MYBL2', 'MMP11',
             'GSTP1', 'BAG1', 'GAPDH', 'TFRC', 'GUSB', 'BCL2', 'SCUBE2', 'BAG1', 'ACTB']
# usually it is use genes whatever is available for all datasets

# Define the file paths to your datasets
# Collect all CSV file paths from the specified directory
file_paths_unsorted = glob.glob('./clindb_breast/*.csv')
# Sort the file paths to ensure "GSE164458_paclitaxel.csv" is first [any main task file name you would like to use]
file_paths = sorted(file_paths_unsorted, key=lambda x: (args.main_task_file not in x, x))
# Generate immune_paths based on the file_paths
immune_paths = ['./immune_profile/immune_' + fp.split('/')[-1] for fp in file_paths]
# file_paths = [
#     'brightness_input/GSE164458/GSE164458_paclitaxel.csv',
#     'brightness_input/GSE164458/GSE164458_carboplatin_paclitaxel.csv',
#     'brightness_input/GSE164458/GSE164458_veliparib_carboplatin_paclitaxel.csv'
# ]

# immune_paths = ['immune_profile/immune_GSE164458_paclitaxel.csv',
#                 'immune_profile/immune_GSE164458_carboplatin_paclitaxel.csv',
#                 'immune_profile/immune_GSE164458_veliparib_carboplatin_paclitaxel.csv']

# Create the dataset
dataset = BreastCancerDataset(oncotype_file_paths=file_paths, cibersort_file_path= immune_paths, genes_of_interest=genes_21)
# Initialize lists to store AUROC and AUPR values for each MCCV iteration
auroc_list = []
aupr_list = []
# test_indices = dataset.test_indices # Test indices for the main task (30%)
test_indices = generate_test_indices(file_paths[0], test_size_fraction=0.3, seed=args.mccv)

# All indices in the main task dataset for single task learning
all_indices = np.arange(len(pd.read_csv(file_paths[0])))

# Exclude test indices to find the remaining ones for train/validation
remaining_indices = np.setdiff1d(all_indices, test_indices)

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
train_loader = DataLoader(train_subset, batch_size=50, shuffle=True) # default is 128
val_loader = DataLoader(val_subset, batch_size=50, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=9999 , shuffle=False) # Batch size is the size of the test set

# Instantiate the model
num_features = dataset.features.shape[1]
num_labels = dataset.labels.shape[1]
model = STLModel(num_features, num_labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the criterion and optimizer
criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1)
# Add L2 regularization via weight_decay
optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4)
# Define a learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=1e-3)

# Training loop
# Initialize parameters and lists to track losses
num_epochs = 100
patience = 10
epochs_without_improvement = 0
best_val_auroc = 0.0
best_model_state = None
train_loss_list = []
val_loss_list = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    epoch_train_loss = 0  # Track training loss for the epoch
    num_batches = 0  # Count of batches for averaging loss
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        label = labels.squeeze(1)  # Use only the first label
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        epoch_train_loss += loss.item()  # Add batch loss to epoch loss
        num_batches += 1
    # Calculate average training loss for the epoch
    avg_train_loss = epoch_train_loss / num_batches
    train_loss_list.append(avg_train_loss)
    # Validation
    model.eval()  # Set model to evaluation mode
    epoch_val_loss = 0
    val_targets = []
    val_outputs = []
    with torch.no_grad():
        num_batches = 0
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            label = labels.squeeze(1)  # Use only the first label
            outputs = model(inputs)
            val_outputs.extend(outputs.sigmoid().cpu().numpy())
            val_targets.extend(label.cpu().numpy())
            loss = criterion(outputs, labels)  # Validation loss
            epoch_val_loss += loss.item()  # Accumulate validation loss
            num_batches += 1
    # Calculate average validation loss for the epoch
    avg_val_loss = epoch_val_loss / num_batches
    val_loss_list.append(avg_val_loss)
    

    # the validation AUROC for the main task
    # Initialize lists to store true and predicted values
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for data in val_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device).float(), targets.to(device).float()

            # Forward pass
            outputs = model(inputs)
            # Apply sigmoid since BCEWithLogitsLoss was used
            probabilities = torch.sigmoid(outputs)

            # Store probabilities and targets to compute AUROC later
            all_outputs.append(probabilities.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batch outputs and targets to compute overall metrics
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute AUROC for the main task
    auroc = roc_auc_score(all_targets[:, 0], all_outputs[:, 0])
    print(f'Epoch {epoch+1}, Validation AUROC: {auroc:.4f}')

    # Early Stopping Check
    if auroc > best_val_auroc:
        best_val_auroc = auroc
        epochs_without_improvement = 0
        best_model_state = copy.deepcopy(model.state_dict())
        print(f"Validation AUROC improved to {auroc:.4f}, saving model...")
    if auroc > 0.7:
        print("Validation AUROC is greater than 0.7. Stopping early.") # prevent overfitting
        break
    else:
        epochs_without_improvement += 1
        print(f"Validation AUROC did not improve. Patience: {epochs_without_improvement}/{patience}")
        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

epochs = list(range(1, len(train_loss_list) + 1))  # List of epoch numbers
# Create a new plot
plt.figure(figsize=(10, 6))  # Optional: Set the figure size for the plot
# Plot train loss
plt.plot(epochs, train_loss_list, label='Train Loss', linestyle='-', marker='o', color='b')  # Customize as needed
# Plot validation loss
plt.plot(epochs, val_loss_list, label='Validation Loss', linestyle='-', marker='o', color='r')  # Customize as needed
# Add labels and a title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss over Epochs')
# add subtitle "loss is BCEWithLogitsLoss"
plt.suptitle('Loss = Binary cross entropy w/ logit loss')
# Add a legend to distinguish between train and validation loss
plt.legend()
# save as png
plt.savefig(f'./output/MCCV_oncotype_STL_learn_{args.main_task_file.replace(".csv", "")}_{args.mccv}.png')

# After completing the training loop, or if early stopping was triggered,
# you may want to load the best model state for further use or evaluation:
model.load_state_dict(best_model_state)
# test
print("### Test Evaluation ###")
test_targets = []
test_outputs = []
with torch.no_grad():
    model.eval()  # Ensure model is in evaluation mode
    # Iterate over the test dataset
    for data in test_loader:
        inputs, targets = data
        inputs = inputs.to(device).float()
        targets = targets.to(device).float()
        # Forward pass to get outputs
        outputs = model(inputs)
        # Apply sigmoid to the outputs to get probabilities since BCEWithLogitsLoss was used during training
        probabilities = torch.sigmoid(outputs)
        # Collect the probabilities and true labels
        test_outputs.append(probabilities.cpu().numpy())
        test_targets.append(targets.cpu().numpy())
    # Convert lists of arrays into single numpy arrays
    test_outputs_np = np.vstack(test_outputs)  # Stack arrays vertically
    test_targets_np = np.vstack(test_targets)
        
    # Compute AUROC and AUPR for only the main task
    test_auroc = roc_auc_score(test_targets_np[:, 0], test_outputs_np[:, 0])
    test_aupr = average_precision_score(test_targets_np[:, 0], test_outputs_np[:, 0])

    print(f'AUROC for main task: {test_auroc:.4f}')
    print(f'AUPR for main task: {test_aupr:.4f}')

# Once all iterations are complete, create a DataFrame from the lists
results_df = pd.DataFrame({
    'AUROC': [test_auroc],  
    'AUPR': [test_aupr]  
}, index=[0]) 

# # Save the DataFrame to a CSV file
# remove .csv
main_task_name = args.main_task_file.replace('.csv', '')
results_df.to_csv(f'./output/MCCV_oncotype_STL_learn_{main_task_name}_{args.mccv}.csv', index=False)



    # Calculate validation metrics
#     val_auroc = roc_auc_score(val_targets, val_outputs, average='macro')
#     print(f'Epoch {epoch+1}, Validation AUROC: {val_auroc:.4f}')

#     # Early Stopping Check
#     if val_auroc > best_val_auroc:
#         best_val_auroc = val_auroc
#         epochs_without_improvement = 0
#         best_model_state = copy.deepcopy(model.state_dict())  # Save the best model state
#         print(f"Validation AUROC improved to {val_auroc:.4f}, saving model...")
#     else:
#         epochs_without_improvement += 1
#         print(f"Validation AUROC did not improve. Patience: {epochs_without_improvement}/{patience}")
#         if epochs_without_improvement >= patience:
#             print("Early stopping triggered.")
#             break
# # Test evaluation
# test_targets = []
# test_outputs = []
# with torch.no_grad():
#     model.eval()  # Ensure model is in evaluation mode
#     for inputs, labels in test_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         label = labels.squeeze(1)  # Use only the first label
#         outputs = model(inputs)
#         test_outputs.extend(outputs.sigmoid().cpu().numpy())
#         test_targets.extend(label.cpu().numpy())

# # Calculate test metrics
# test_auroc = roc_auc_score(test_targets, test_outputs, average='macro')  # Adjust as necessary
# test_aupr = average_precision_score(test_targets, test_outputs, average='macro')  # Adjust as necessary
# print(f'Test AUROC: {test_auroc:.4f}, Test AUPR: {test_aupr:.4f}')