import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

model = nn.Linear(20, 5) # 20 in-features, 5 out-features
x = torch.randn(1, 20) # 1 sample with 20 features
y = torch.tensor([[1., 0., 1., 0., 0.]]) # 1 sample with 5 labels; 1st and 3rd labels are active

criterion = nn.BCEWithLogitsLoss() # Binary Cross-Entropy with logit Loss
optimizer = optim.SGD(model.parameters(), lr=1e-1) # stochastic gradient descent

for epoch in range(20):
    optimizer.zero_grad() # Zeroes the gradients to prevent accumulation.
    output = model(x) # Computes the model's output for the input sample.
    loss = criterion(output, y) # Calculates the loss between the output and the target labels. # custom loss function will be used with aux_learn
    loss.backward() # Backpropagates the loss to compute gradients.
    optimizer.step() # Updates the model's weights.
    print('Loss: {:.3f}'.format(loss.item()))