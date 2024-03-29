import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class BreastCancerDataset(Dataset):
    """
    A custom dataset class for breast cancer classification.

    Args:
        file_paths (list): A list of file paths containing the dataset.
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

    def __init__(self, file_paths, genes_of_interest):
        # First, determine the common genes present in all datasets [oncotype DX genes]
        common_genes = self.find_common_genes(file_paths, genes_of_interest)
        
        # Initialize an empty DataFrame for samples and an empty array for labels
        self.samples = pd.DataFrame()
        self.labels = np.array([]).reshape(0, len(file_paths))  # No rows initially, but len(file_paths) columns
        
        # Dictionary to map Sample to its index in self.samples and self.labels
        sample_id_to_index = {}
        
        # Process each file
        for file_idx, file_path in enumerate(file_paths):
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
                    new_label_row = np.zeros(len(file_paths))
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
        self.features = self.samples
        self.normalize_features()

    def find_common_genes(self, file_paths, genes_of_interest):
        """
        Find the common genes present in all datasets.

        Args:
            file_paths (list): A list of file paths containing the dataset.
            genes_of_interest (list): A list of genes of interest.

        Returns:
            list: A list of common genes.

        """
        common_genes = set(genes_of_interest)
        for file_path in file_paths:
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