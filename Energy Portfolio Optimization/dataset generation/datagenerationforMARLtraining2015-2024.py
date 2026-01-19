"""
Port of `Dataset Generation 2015-2024.ipynb` (verbatim logic).

Goal: Generate 20 half-year scenarios (2015–2024, H1/H2) at 10-minute resolution
using:
  - real wind speed time series (from `windspeed.csv`)
  - real Elspot prices (Energi Data Service, DK1)
  - synthetic solar/hydro/load profiles
  - PyPSA optimization per scenario
  - a simple heuristic battery energy simulation

IMPORTANT (by design, to match the notebook):
  - Randomness is NOT seeded by default (solar/hydro/load/carbon_tax vary per run).
  - Wind + price alignment uses `reindex(..., method="nearest")`.
  - Wind + price series are resampled to 10min and linearly interpolated.

If you want reproducibility, you may pass `--seed`, but the default is identical
to the notebook (no explicit seed).
"""

import argparse
import json
import os
from datetime import datetime  # noqa: F401 (kept to mirror notebook imports)

import numpy as np
import pandas as pd
import pypsa
import requests


# === CONFIG (matches notebook defaults) ===
OUTPUT_DIR = "dataset"
RESOLUTION = "10min"
FREQ_MINUTES = 10


# === Fixed Scenario Dates (matches notebook) ===
SCENARIO_DATES = []
for year in range(2015, 2025):
    SCENARIO_DATES.append({"start": f"{year}-01-01 00:00", "end": f"{year}-06-30 23:50"})
    SCENARIO_DATES.append({"start": f"{year}-07-01 00:00", "end": f"{year}-12-31 23:50"})


def fetch_elspot_prices(start_date: str, end_date: str, price_area: str = "DK1") -> pd.DataFrame:
    url = "https://api.energidataservice.dk/dataset/Elspotprices"
    params = {
        "start": f"{start_date}",
        "end": f"{end_date}",
        "filter": json.dumps({"PriceArea": [price_area]}),
        "columns": "HourDK,SpotPriceDKK",
        "sort": "HourDK ASC",
        "timezone": "dk",
    }
    response = requests.get(url, params=params)
    records = response.json().get("records", [])
    df = pd.DataFrame(records)
    df["HourDK"] = pd.to_datetime(df["HourDK"])
    df = df.drop_duplicates("HourDK").set_index("HourDK").sort_index()
    df = df.resample(RESOLUTION).interpolate()
    return df.rename(columns={"SpotPriceDKK": "market_price"})


def realistic_wind_power_curve(speed: np.ndarray) -> np.ndarray:
    cut_in, rated, cut_out = 3, 12, 25
    output = np.zeros_like(speed)
    output[(speed >= cut_in) & (speed <= rated)] = (
        (speed[(speed >= cut_in) & (speed <= rated)] - cut_in) / (rated - cut_in)
    )
    output[(speed > rated) & (speed < cut_out)] = 1
    return np.clip(output, 0, 1)


def build_network(
    timestamps: pd.DatetimeIndex,
    wind_pu: np.ndarray,
    solar_pu: np.ndarray,
    hydro_pu: np.ndarray,
    load_profile: np.ndarray,
    carbon_tax: float,
    wind_capacity: float,
) -> pypsa.Network:
    network = pypsa.Network()
    network.set_snapshots(timestamps)

    for bus in ["Wind", "Solar", "Hydro", "Grid"]:
        network.add("Bus", bus, carrier="electricity")

    network.add(
        "Generator",
        "Wind",
        bus="Wind",
        p_nom=wind_capacity,
        p_max_pu=wind_pu,
        capital_cost=1000000,
        marginal_cost=2 + carbon_tax,
    )
    network.add(
        "Generator",
        "Solar",
        bus="Solar",
        p_nom=1000,
        p_max_pu=solar_pu,
        capital_cost=600000,
        marginal_cost=1 + carbon_tax,
    )
    network.add(
        "Generator",
        "Hydro",
        bus="Hydro",
        p_nom=1000,
        p_max_pu=hydro_pu,
        capital_cost=1200000,
        marginal_cost=0.5,
    )

    network.add("Load", "Grid_Load", bus="Grid", p_set=load_profile)
    network.add(
        "Generator",
        "Slack",
        bus="Grid",
        p_nom=1e6,
        p_min_pu=-1,
        p_max_pu=1,
        marginal_cost=1e6,
    )

    for src in ["Wind", "Solar", "Hydro"]:
        network.add(
            "Line",
            f"{src}_to_Grid",
            bus0=src,
            bus1="Grid",
            s_nom=1575,
            x=0.01,
            r=0.01,
            capital_cost=0,
        )

    # Mirror notebook's carrier assignment (even though it triggers warnings in some PyPSA versions)
    for comp in ["buses", "generators", "loads", "lines"]:
        getattr(network, comp)["carrier"] = "electricity"

    return network


def run_simulation(
    scenario_id: str,
    start_date: str,
    end_date: str,
    wind_speed_df: pd.DataFrame,
    real_prices: pd.DataFrame,
    solver_name: str = "glpk",
    output_dir: str = OUTPUT_DIR,
) -> pd.DataFrame:
    timestamps = pd.date_range(start=start_date, end=end_date, freq=RESOLUTION)

    wind_speed = wind_speed_df.reindex(timestamps, method="nearest")["wind_speed"].values
    wind_pu = realistic_wind_power_curve(wind_speed)

    # --- Realistic Solar Profile (verbatim) ---
    solar_profile = []
    for ts in timestamps:
        hour = ts.hour + ts.minute / 60
        daylight = np.exp(-0.5 * ((hour - 13) / 3) ** 2) if 6 <= hour <= 20 else 0
        seasonal = np.clip(np.cos((ts.dayofyear - 172) * 2 * np.pi / 365), 0.2, 1.0)
        value = daylight * seasonal * np.random.normal(1.0, 0.05)
        solar_profile.append(max(0, min(1, value)))
    solar_profile = np.array(solar_profile)

    hydro_profile = 0.5 + 0.3 * np.sin(np.linspace(0, 5 * np.pi, len(timestamps))) + np.random.uniform(-0.2, 0.2)
    hydro_profile += np.random.normal(0, 0.05, len(timestamps))
    hydro_profile = np.clip(hydro_profile, 0.1, 1.0)

    load_profile = 3000 + 1000 * np.sin(np.linspace(0, len(timestamps) * 2 * np.pi / (24 * 6), len(timestamps)))
    load_profile += np.random.normal(0, 200, len(timestamps))
    load_profile = np.clip(load_profile, 1000, 6000)

    carbon_tax = np.random.choice([0, 20, 50])
    wind_capacity = 100 * 15

    network = build_network(timestamps, wind_pu, solar_profile, hydro_profile, load_profile, carbon_tax, wind_capacity)
    network.optimize(solver_name=solver_name)

    prices = real_prices.reindex(timestamps, method="nearest")["market_price"].values

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "wind": network.generators_t.p["Wind"].values,
            "solar": network.generators_t.p["Solar"].values,
            "hydro": network.generators_t.p["Hydro"].values,
            "load": load_profile,
            "price": prices,
            "scenario": scenario_id,
        }
    )

    df["revenue"] = (df["wind"] + df["solar"] + df["hydro"]) * df["price"]

    # --- Battery Simulation (verbatim) ---
    battery_capacity_mwh = 10.0
    battery_power_limit_mw = 5.0
    battery_efficiency = 0.9
    battery_energy = 0.0
    battery_energy_series = []
    avg_price = df["price"].mean()

    for _, row in df.iterrows():
        surplus = max(0.0, row["wind"] + row["solar"] + row["hydro"] - row["load"])
        can_charge = min(battery_power_limit_mw, surplus)
        charge_energy = can_charge * battery_efficiency

        if row["price"] < avg_price:
            battery_energy += charge_energy
        else:
            discharge = min(battery_power_limit_mw, battery_energy)
            battery_energy -= discharge

        battery_energy = max(0.0, min(battery_capacity_mwh, battery_energy))
        battery_energy_series.append(battery_energy)

    df["battery_energy"] = battery_energy_series

    df["npv"] = df["revenue"].sum() - (wind_capacity * 1000 + 600000 + 1200000)
    df["risk"] = df["revenue"].std() / df["revenue"].mean()
    df["profit_label"] = (df["npv"] > 0).astype(int)

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/{scenario_id}.csv", index=False)
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--wind_csv", default="windspeed.csv", help="Notebook expects `windspeed.csv` with columns: timestamp,<value>")
    parser.add_argument("--price_area", default="DK1")
    parser.add_argument("--solver", default="glpk")
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Optional. If provided, sets numpy RNG seed for reproducibility. Default matches notebook (no explicit seed).",
    )
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # === Load Wind Speed CSV (verbatim) ===
    wind_speed_df = pd.read_csv(args.wind_csv, parse_dates=["timestamp"])
    wind_speed_df = wind_speed_df.set_index("timestamp").sort_index()
    wind_speed_df = wind_speed_df.resample(RESOLUTION).interpolate()
    wind_speed_df = wind_speed_df.rename(columns={wind_speed_df.columns[0]: "wind_speed"})

    print("📈 Fetching real market price data (2015–2024)...")
    real_prices = fetch_elspot_prices("2015-01-01T00:00", "2024-12-31T23:59", price_area=args.price_area)

    all_scenarios = []
    for i, s in enumerate(SCENARIO_DATES):
        scenario_id = f"scenario_{i:03d}"
        print(f"Running {scenario_id} from {s['start']} to {s['end']}...")
        df = run_simulation(
            scenario_id=scenario_id,
            start_date=s["start"],
            end_date=s["end"],
            wind_speed_df=wind_speed_df,
            real_prices=real_prices,
            solver_name=args.solver,
            output_dir=args.output_dir,
        )
        all_scenarios.append(df)

    final_df = pd.concat(all_scenarios)
    final_df.to_csv(f"{args.output_dir}/dataset.csv", index=False)
    print("\n✅ All 20 scenarios saved with 10-minute resolution, realistic solar output, and battery simulation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

