# converted 20 to 10 for PCA

import torch.nn as nn

class MLCModel(nn.Module):
    def __init__(self, num_features, num_labels):
        super(MLCModel, self).__init__()
        self.layer1 = nn.Linear(num_features, 10)
        self.batchnorm1 = nn.BatchNorm1d(10)  # Batch Normalization after the first linear layer
        self.relu = nn.ReLU()  # Non-linear activation function
        self.dropout = nn.Dropout(0.5)  # Dropout layer to prevent overfitting
        self.layer2 = nn.Linear(10, num_labels)  # Second layer maps to the number of labels
    
    def forward(self, x):
        x = self.layer1(x)  # Pass input through the first layer
        x = self.batchnorm1(x)  # Apply Batch Normalization
        x = self.relu(x)  # Apply non-linearity
        x = self.dropout(x)  # Apply dropout
        x = self.layer2(x)  # Pass through the second layer
        return x
    
class STLModel(nn.Module):
    def __init__(self, num_features, num_labels=1):
        super(STLModel, self).__init__()
        self.layer1 = nn.Linear(num_features, 10)
        self.batchnorm1 = nn.BatchNorm1d(10)  # Batch Normalization after the first linear layer
        self.relu = nn.ReLU()  # Non-linear activation function
        self.dropout = nn.Dropout(0.5)  # Dropout layer to prevent overfitting
        self.layer2 = nn.Linear(10, num_labels)  # Second layer maps to the number of labels
    
    def forward(self, x):
        x = self.layer1(x)  # Pass input through the first layer
        x = self.batchnorm1(x)  # Apply Batch Normalization
        x = self.relu(x)  # Apply non-linearity
        x = self.dropout(x)  # Apply dropout
        x = self.layer2(x)  # Pass through the second layer
        return x


# class MLCModel(nn.Module):
#     def __init__(self, num_features, num_labels):
#         super(MLCModel, self).__init__()
#         self.layer1 = nn.Linear(num_features, 20)
#         self.relu = nn.ReLU()  # Non-linear activation function
#         self.dropout = nn.Dropout(0.5)  # Dropout layer to prevent overfitting
#         self.layer2 = nn.Linear(20, num_labels)  # Second layer maps to the number of labels
    
#     def forward(self, x):
#         x = self.layer1(x)  # Pass input through the first layer
#         x = self.relu(x)  # Apply non-linearity
#         x = self.dropout(x)  # Apply dropout
#         x = self.layer2(x)  # Pass through the second layer
#         return x

# Previous model
# class MLCModel(nn.Module):
#     def __init__(self, num_features, num_labels):
#         super(MLCModel, self).__init__()
#         self.layer1 = nn.Linear(num_features, 10)
#         self.relu = nn.ReLU()  # Non-linear activation function
#         self.layer2 = nn.Linear(10, num_labels)  # Second layer maps to the number of labels
    
#     def forward(self, x):
#         x = self.layer1(x)  # Pass input through the first layer
#         x = self.relu(x)  # Apply non-linearity
#         x = self.layer2(x)  # Pass through the second layer
#         return x