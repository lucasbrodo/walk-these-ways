import pickle
import numpy as np
import torch

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def detailed_compare(data1, data2, path=""):
    if isinstance(data1, np.ndarray) and isinstance(data2, np.ndarray):
        if np.array_equal(data1, data2):
            return True
        else:
            print(f"Different values at {path}: Array sizes or elements differ")
            return False
    elif isinstance(data1, torch.Tensor) and isinstance(data2, torch.Tensor):
        if torch.equal(data1, data2):
            return True
        else:
            print(f"Different values at {path}: Tensor sizes or elements differ")
            return False
    elif isinstance(data1, dict) and isinstance(data2, dict):
        keys1 = set(data1.keys())
        keys2 = set(data2.keys())
        shared_keys = keys1.intersection(keys2)
        different_keys = keys1.symmetric_difference(keys2)

        for key in different_keys:
            if key in data1:
                print(f"Different or missing in second: {path}/{key} - Value 1: {data1[key]}")
            else:
                print(f"Different or missing in first: {path}/{key} - Value 2: {data2[key]}")

        for key in shared_keys:
            if not detailed_compare(data1[key], data2[key], path=f"{path}/{key}"):
                print(f"Difference found at {path}/{key}")
    elif isinstance(data1, list) and isinstance(data2, list):
        for i in range(max(len(data1), len(data2))):
            if i < len(data1) and i < len(data2):
                if not detailed_compare(data1[i], data2[i], path=f"{path}/[{i}]"):
                    print(f"Difference found at {path}/[{i}]")
            else:
                if i < len(data1):
                    print(f"Extra element in first: {path}/[{i}] - {data1[i]}")
                else:
                    print(f"Extra element in second: {path}/[{i}] - {data2[i]}")
    else:
        if data1 == data2:
            return True
        else:
            print(f"Different values at {path}: {data1} vs {data2}")
            return False

    return True

# Paths to your .pkl files
file_path1 = '/home/lucas/github/walk-these-ways/runs/gait-conditioned-agility/pretrain-v0/train/025417.456545/parameters.pkl'
file_path2 = '/home/lucas/github/walk-these-ways/runs/gait-conditioned-agility/pretrain-v0/train/025417.456545/parameters2.pkl'

# Load the data from each file
data1 = load_pickle(file_path1)
data2 = load_pickle(file_path2)

# Compare the data in detail
detailed_compare(data1, data2)
