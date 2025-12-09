"""Debug script to check layer names in saved models"""
import h5py
import os

model_dir = "models"
metadata_dir = "metadata"

# Check one model
metadata_file = "wind_immediate_metadata.json"
metadata_path = os.path.join(metadata_dir, metadata_file)

import json
with open(metadata_path, 'r') as f:
    md = json.load(f)

model_path = md.get('model_path_best') or md.get('model_path')
if not os.path.isabs(model_path):
    model_path = os.path.join(model_dir, os.path.basename(model_path))

print(f"Checking model: {model_path}")
print("="*70)

def print_structure(name, obj, indent=0):
    """Recursively print HDF5 structure"""
    prefix = "  " * indent
    if isinstance(obj, h5py.Group):
        print(f"{prefix}{name}/ (Group)")
        for key in obj.keys():
            print_structure(key, obj[key], indent + 1)
    elif isinstance(obj, h5py.Dataset):
        print(f"{prefix}{name}: shape={obj.shape}, dtype={obj.dtype}")

with h5py.File(model_path, 'r') as f:
    if 'model_weights' in f:
        model_weights = f['model_weights']
        print("Full structure of model_weights:")
        print_structure("model_weights", model_weights)
    else:
        print("No 'model_weights' found")
        print("Keys in file:", list(f.keys()))

