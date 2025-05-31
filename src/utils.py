import os
import sys
import yaml
import torch

def device_init(device: str = "cuda"):
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        return torch.device("cpu")
    
    
def config_files():
    with open(file="./config.yml", mode="r") as file:
        return yaml.safe_load(file)