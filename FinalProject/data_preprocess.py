import os
import json
import numpy as np
import pandas as pd
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Define model layers here
        nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Conv2d(512,128,3),
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        # Define forward pass here
        return x
    


if __name__ == "__main__":

    train_data = pd.read_csv('./data/train.csv')
    mappings = {}
    for col in train_data.columns:
        if train_data[col].dtype == 'object':
            cat = train_data[col].astype('category')
            mappings[col] = list(cat.cat.categories)
            train_data[col] = cat.cat.codes
    
    print(train_data.head())
    print(mappings)