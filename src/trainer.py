import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append("./src/")

class Trainer():
    def __init__(self, dataloader:DataLoader = None, epochs: int = 100):
        self.dataloader = dataloader
        self.epochs = epochs
        
    def train(self):
        pass
    
if __name__ == "__main__":
    pass