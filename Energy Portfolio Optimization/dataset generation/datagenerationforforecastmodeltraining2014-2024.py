"""
Forecast-model training dataset generator (rolling window), covering 2014–2024.

High-level intent (as requested):
  - Keep the existing 20 MARL-training scenarios for 2015–2024 *untouched* (identical bytes)
  - Add 2 new half-year scenarios for 2014 (H1 and H2)
  - Produce forecast-training "episodes" as rolling 2-half-year windows:
      episode 0  = 2014H1 + 2014H2   (train forecasts -> used for MARL 2015H1)
      episode 1  = 2014H2 + 2015H1   (-> used for MARL 2015H2)
      episode 2  = 2015H1 + 2015H2   (-> used for MARL 2016H1)
      ...
      episode 19 = 2023H2 + 2024H1   (-> used for MARL 2024H2)
      episode 20 = 2024H1 + 2024H2   (forecasts for 2025 evaluation)

Outputs (default `forecast_model_training_dataset/`):
  - `halfyear_scenarios/`
      - `scenario_2014_h1.csv`, `scenario_2014_h2.csv`   (generated)
      - `scenario_000.csv` ... `scenario_019.csv`        (copied unchanged from MARL dir)
  - `episodes/`
      - `episode_000.csv` ... `episode_020.csv`          (concatenations of adjacent halves)

Notes:
  - The 2014 half-year scenarios are generated with the same PyPSA + synthetic profiles
    logic as the MARL-training generator script.
  - By default, RNG is NOT seeded (to mirror notebook-style behavior). If you want exact
    reproducibility, pass `--seed`.
"""

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests


RESOLUTION = "10min"


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
    if df.empty:
        raise RuntimeError(
            "Elspotprices API returned 0 records. Check connectivity, date range, or PriceArea."
        )
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
):
    # Lazy import so `--validate_only` works even if PyPSA isn't installed in the active env.
    import pypsa  # type: ignore

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


def run_simulation_halfyear(
    scenario_id: str,
    start_date: str,
    end_date: str,
    wind_speed_df: pd.DataFrame,
    real_prices: pd.DataFrame,
    solver_name: str,
) -> pd.DataFrame:
    timestamps = pd.date_range(start=start_date, end=end_date, freq=RESOLUTION)

    wind_speed = wind_speed_df.reindex(timestamps, method="nearest")["wind_speed"].values
    wind_pu = realistic_wind_power_curve(wind_speed)

    # --- Realistic Solar Profile (matches MARL generator / notebook logic) ---
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

    # --- Battery Simulation (matches MARL generator / notebook logic) ---
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

    return df


def load_wind_speed_df(wind_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(wind_csv, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df.resample(RESOLUTION).interpolate()
    # Match the MARL generator: rename the single value column to "wind_speed"
    df = df.rename(columns={df.columns[0]: "wind_speed"})
    return df


@dataclass(frozen=True)
class HalfYearSpec:
    label: str
    start: str
    end: str


def halfyear_label_for_segment_index(seg_idx: int) -> str:
    # seg_idx 0 = 2014H1, 1 = 2014H2, 2 = 2015H1, ..., 21 = 2024H2
    year = 2014 + (seg_idx // 2)
    half = "H1" if (seg_idx % 2 == 0) else "H2"
    return f"{year}{half}"


def build_halfyear_specs_2014_2024() -> list[HalfYearSpec]:
    specs: list[HalfYearSpec] = []
    for year in range(2014, 2025):
        specs.append(HalfYearSpec(label=f"{year}H1", start=f"{year}-01-01 00:00", end=f"{year}-06-30 23:50"))
        specs.append(HalfYearSpec(label=f"{year}H2", start=f"{year}-07-01 00:00", end=f"{year}-12-31 23:50"))
    # 2014H1..2024H2 => 22 segments
    return specs[:22]


def assert_halfyear_schema(df: pd.DataFrame, path_hint: str) -> None:
    required = {
        "timestamp",
        "wind",
        "solar",
        "hydro",
        "load",
        "price",
        "scenario",
        "revenue",
        "battery_energy",
        "npv",
        "risk",
        "profit_label",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path_hint} is missing columns: {missing}")


def concat_episode(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    out = pd.concat([a, b], ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values("timestamp").reset_index(drop=True)
    if out["timestamp"].duplicated().any():
        dup = out[out["timestamp"].duplicated(keep=False)]["timestamp"].iloc[0]
        raise ValueError(f"Episode concatenation produced duplicate timestamp: {dup}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--marl_scenarios_dir",
        default="training_dataset",
        help="Directory containing MARL half-year scenarios scenario_000.csv..scenario_019.csv (2015H1..2024H2).",
    )
    parser.add_argument(
        "--wind_csv",
        default=str(Path("dataset generation") / "windspeed_2014-2024.csv"),
        help="Wind speed CSV containing at least 2014 (timestamp + value column).",
    )
    parser.add_argument("--price_area", default="DK1")
    parser.add_argument("--solver", default="glpk")
    parser.add_argument("--output_dir", default="forecast_training_dataset")
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Optional. If provided, sets numpy RNG seed. Default is unseeded (notebook-style).",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Only validate mapping + input presence; do not generate/copy/write.",
    )
    args = parser.parse_args()

    marl_dir = Path(args.marl_scenarios_dir)
    wind_csv = Path(args.wind_csv)
    out_dir = Path(args.output_dir)
    out_halfyears = out_dir / "halfyear_scenarios"
    out_episodes = out_dir / "episodes"

    # Mapping preview (high-signal sanity checks)
    print("Forecast rolling window mapping (episode -> halfyears):")
    print("  episode_000 =", halfyear_label_for_segment_index(0), "+", halfyear_label_for_segment_index(1))
    print("  episode_001 =", halfyear_label_for_segment_index(1), "+", halfyear_label_for_segment_index(2))
    print("  episode_019 =", halfyear_label_for_segment_index(19), "+", halfyear_label_for_segment_index(20))
    print("  episode_020 =", halfyear_label_for_segment_index(20), "+", halfyear_label_for_segment_index(21))

    # Validate MARL scenarios exist
    marl_paths = [marl_dir / f"scenario_{i:03d}.csv" for i in range(20)]
    missing = [str(p) for p in marl_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing MARL scenario files. Expected scenario_000.csv..scenario_019.csv in "
            f"{str(marl_dir)}. Missing: {missing[:5]}{' ...' if len(missing) > 5 else ''}"
        )
    if not wind_csv.exists():
        raise FileNotFoundError(f"Wind CSV not found: {str(wind_csv)}")

    if args.validate_only:
        print("validate_only=True: inputs look good. Exiting without writing outputs.")
        return 0

    if args.seed is not None:
        np.random.seed(args.seed)

    os.makedirs(out_halfyears, exist_ok=True)
    os.makedirs(out_episodes, exist_ok=True)

    # Load exogenous time series sources (wind + real prices)
    wind_speed_df = load_wind_speed_df(wind_csv)
    print("Fetching real market price data (2014–2024)...")
    real_prices = fetch_elspot_prices("2014-01-01T00:00", "2024-12-31T23:59", price_area=args.price_area)

    # 1) Generate 2014 H1/H2 half-year scenarios
    specs = build_halfyear_specs_2014_2024()
    h1_2014 = specs[0]
    h2_2014 = specs[1]

    print(f"Generating {h1_2014.label} -> scenario_2014_h1.csv")
    df_2014_h1 = run_simulation_halfyear(
        scenario_id="scenario_2014_h1",
        start_date=h1_2014.start,
        end_date=h1_2014.end,
        wind_speed_df=wind_speed_df,
        real_prices=real_prices,
        solver_name=args.solver,
    )
    print(f"Generating {h2_2014.label} -> scenario_2014_h2.csv")
    df_2014_h2 = run_simulation_halfyear(
        scenario_id="scenario_2014_h2",
        start_date=h2_2014.start,
        end_date=h2_2014.end,
        wind_speed_df=wind_speed_df,
        real_prices=real_prices,
        solver_name=args.solver,
    )

    df_2014_h1.to_csv(out_halfyears / "scenario_2014_h1.csv", index=False)
    df_2014_h2.to_csv(out_halfyears / "scenario_2014_h2.csv", index=False)

    # 2) Copy 2015–2024 MARL half-year scenarios unchanged into this dataset
    for i, src in enumerate(marl_paths):
        dst = out_halfyears / f"scenario_{i:03d}.csv"
        shutil.copy2(src, dst)

    # 3) Build segments list: [2014H1, 2014H2, 2015H1..2024H2]
    segments: list[pd.DataFrame] = []

    seg0 = pd.read_csv(out_halfyears / "scenario_2014_h1.csv", parse_dates=["timestamp"])
    seg1 = pd.read_csv(out_halfyears / "scenario_2014_h2.csv", parse_dates=["timestamp"])
    assert_halfyear_schema(seg0, "scenario_2014_h1.csv")
    assert_halfyear_schema(seg1, "scenario_2014_h2.csv")
    segments.extend([seg0, seg1])

    for i in range(20):
        seg = pd.read_csv(out_halfyears / f"scenario_{i:03d}.csv", parse_dates=["timestamp"])
        assert_halfyear_schema(seg, f"scenario_{i:03d}.csv")
        segments.append(seg)

    if len(segments) != 22:
        raise RuntimeError(f"Expected 22 half-year segments (2014H1..2024H2), got {len(segments)}")

    # 4) Write rolling forecast episodes (0..20): concat segment[k] + segment[k+1]
    for ep in range(21):
        a = segments[ep]
        b = segments[ep + 1]
        ep_df = concat_episode(a, b)

        # Minimal validation: monotonic-ish, expected span
        expected_start = pd.to_datetime(specs[ep].start)
        expected_end = pd.to_datetime(specs[ep + 1].end)
        got_start = pd.to_datetime(ep_df["timestamp"].min())
        got_end = pd.to_datetime(ep_df["timestamp"].max())
        if got_start != expected_start or got_end != expected_end:
            raise ValueError(
                f"episode_{ep:03d} timestamp span mismatch. "
                f"Expected {expected_start}..{expected_end}, got {got_start}..{got_end}"
            )

        ep_path = out_episodes / f"episode_{ep:03d}.csv"
        ep_df.to_csv(ep_path, index=False)

    print("Done. Wrote halfyear_scenarios/ (2014H1/H2 + copied MARL 2015-2024) and episodes/episode_000..episode_020.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

