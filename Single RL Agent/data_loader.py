# data_loader.py
import pandas as pd

def load_energy_data(path="sample.csv"):
    """
    Loads and filters the energy dataset for 2024.
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df[(df["timestamp"] >= "2024-01-01") & (df["timestamp"] <= "2024-12-31")].reset_index(drop=True)
    return df
