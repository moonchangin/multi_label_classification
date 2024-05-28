import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class BreastCancerDataset(Dataset):
    """
    A custom dataset class for breast cancer classification.

    Args:
        oncotype_file_paths (list): A list of file paths containing the dataset.
        cibersort_file_path (str): The file path containing the immune CIBERSORT dataset.
        genes_of_interest (list): A list of genes of interest.

    Attributes:
        samples (DataFrame): A DataFrame to store the samples.
        labels (ndarray): An array to store the labels.
        features (DataFrame): A DataFrame to store the normalized features.

    Methods:
        find_common_genes: Find the common genes present in all datasets.
        normalize_features: Normalize the features using Z-normalization.
        __len__: Get the length of the dataset.
        __getitem__: Get a specific sample and its corresponding label.

    """

    def __init__(self, oncotype_file_paths, cibersort_file_path, genes_of_interest):
        # First, determine the common genes present in all datasets [oncotype DX genes]
        common_genes = self.find_common_genes(oncotype_file_paths, genes_of_interest)
        
        # Initialize an empty DataFrame for samples and an empty array for labels
        self.samples = pd.DataFrame()
        self.labels = np.array([]).reshape(0, len(oncotype_file_paths))  # No rows initially, but len(oncotype_file_paths) columns

        # Load and normalize the Oncotype DX data
        oncotype_data, oncotype_labels = self.load_and_normalize_oncotype_data(oncotype_file_paths, common_genes)
        # Load and normalize the CIBERSORT data
        cibersort_data = self.load_and_normalize_cibersort_data(cibersort_file_path)
        
        # Concatenate the normalized features from both datasets
        self.features = pd.concat([oncotype_data, cibersort_data], axis=1)
        self.labels = oncotype_labels  # Assuming labels are only relevant from the Oncotype dataset

        # After loading and processing all data, create test indices
        self.test_indices = self.generate_test_indices(oncotype_file_paths[0], test_size_fraction=0.3)

    def load_and_normalize_oncotype_data(self, oncotype_file_paths, common_genes):
         # Initialize an empty DataFrame for collecting samples from all files
        all_samples = pd.DataFrame()
        # Dictionary to map Sample to its index in self.samples and self.labels
        sample_id_to_index = {}
        # Process each file
        for file_idx, file_path in enumerate(oncotype_file_paths):
            df = pd.read_csv(file_path)
            # Assuming the second column contains the actual labels
            actual_labels = df.iloc[:, 1].values
            # Filter columns based on common_genes, ensuring 'Sample' is included
            df = df[['Sample'] + common_genes]
            for i, row in df.iterrows():
                sample_id = row['Sample']
                if sample_id not in sample_id_to_index:
                    # New sample, append it
                    self.samples = pd.concat([self.samples, pd.DataFrame([row])], ignore_index=True)
                    # Initialize a new label row with zeros and set the label for the current file
                    new_label_row = np.zeros(len(oncotype_file_paths))
                    new_label_row[file_idx] = actual_labels[i]
                    self.labels = np.vstack([self.labels, new_label_row])
                    
                    # Map the Sample to its new index
                    sample_id_to_index[sample_id] = len(sample_id_to_index)
                else:
                    # Existing sample, update its label for the current file
                    idx = sample_id_to_index[sample_id]
                    self.labels[idx, file_idx] = actual_labels[i]
        
        # Drop 'Sample' column after use
        self.samples.drop('Sample', axis=1, inplace=True)
        
        # Normalize features
        all_samples = self.samples
        labels = self.labels
        normalized_features = (all_samples - all_samples.mean()) / all_samples.std()
        return normalized_features, labels

    def load_and_normalize_cibersort_data(self, file_paths):
        # code to load CIBERSORT data, apply normalization, and return it
        # Initialize an empty DataFrame for collecting samples from all files
        all_samples = pd.DataFrame()
        
        # Process each file
        for file_path in file_paths:
            # Load the dataset from the current file
            df = pd.read_csv(file_path)
            cibersort_features = [col for col in df.columns if col.startswith("CIBERSORT")]
            
            # Filter columns based on cibersort_features, ensuring 'Sample' is included
            filtered_df = df[['Sample'] + cibersort_features]

            # Append the current file's data and labels to the collective DataFrame and list
            all_samples = pd.concat([all_samples, filtered_df], ignore_index=True)
           # After concatenating data from all files, drop the 'Sample' column
        all_samples.drop('Sample', axis=1, inplace=True)

        # Normalize the features using Z-score normalization
        normalized_features = (all_samples - all_samples.mean()) / all_samples.std()
        return normalized_features
            
    def find_common_genes(self, oncotype_file_paths, genes_of_interest):
        """
        Find the common genes present in all datasets.

        Args:
            oncotype_file_paths (list): A list of file paths containing the dataset.
            genes_of_interest (list): A list of genes of interest.

        Returns:
            list: A list of common genes.

        """
        common_genes = set(genes_of_interest)
        for file_path in oncotype_file_paths:
            df = pd.read_csv(file_path)
            common_genes &= set(df.columns)
        return list(common_genes)

    def normalize_features(self):
        """
        Normalize the features using Z-normalization.

        """
        self.features = (self.features - self.features.mean()) / self.features.std()

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.

        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        Get a specific sample and its corresponding label.

        Args:
            idx (int): The index of the sample.

        Returns:
            tuple: A tuple containing the features and label tensors.

        """
        # Convert features and labels to tensors
        features = torch.tensor(self.features.iloc[idx].values, dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return features, label
    
    def generate_test_indices(self, first_file_path, test_size_fraction):
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
        np.random.seed(42)  # Or any seed of your choice
        test_indices = np.random.choice(range(total_samples), size=test_samples_count, replace=False)
        
        return test_indices
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, chkpoint_name='best.pt'):
            """
            Initializes the EarlyStopping object.

            Args:
                patience (int): How long to wait after last time validation loss improved.
                                Default: 7
                verbose (bool): If True, prints a message for each validation loss improvement. 
                                Default: False
                delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                                Default: 0
                chkpoint_name (str): Name of the checkpoint file to save the best model.
                                     Default: 'best.pt'
            """
            self.patience = patience
            self.verbose = verbose
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            self.val_loss_min = np.Inf
            self.delta = delta
            self.chkpoint_name = chkpoint_name

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.chkpoint_name)
        self.val_loss_min = val_loss
