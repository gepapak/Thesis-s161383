import h5py
import os

model_path = "models/wind_immediate_model_best.h5"

print(f"Checking: {model_path}")
print("="*70)

with h5py.File(model_path, 'r') as f:
    if 'model_weights' in f:
        mw = f['model_weights']
        print("Layer names:", list(mw.keys()))
        print()
        
        for layer_name in ['dense', 'dense_1', 'dense_2']:
            if layer_name in mw:
                print(f"{layer_name}:")
                layer = mw[layer_name]
                print(f"  Keys: {list(layer.keys())}")
                if 'sequential' in layer:
                    seq = layer['sequential']
                    print(f"  sequential keys: {list(seq.keys())}")
                    for key in seq.keys():
                        sub = seq[key]
                        if isinstance(sub, h5py.Group):
                            print(f"    {key}/ (Group): {list(sub.keys())}")
                            for subkey in sub.keys():
                                if isinstance(sub[subkey], h5py.Dataset):
                                    print(f"      {subkey}: shape={sub[subkey].shape}, dtype={sub[subkey].dtype}")
                        elif isinstance(sub, h5py.Dataset):
                            print(f"    {key}: shape={sub.shape}, dtype={sub.dtype}")
                print()

