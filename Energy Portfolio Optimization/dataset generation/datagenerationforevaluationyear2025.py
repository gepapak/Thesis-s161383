"""
Data generation for evaluation year 2025 (10-minute resolution).

This script lives in `dataset generation/` and writes outputs inside this folder by default.

It is a cleaned, reproducible port of:
  `dataset generation/Dataset Generation 2025.ipynb`

Notes:
- Evaluation dataset target is the FULL unseen year 2025 by default, generated as TWO 6-month scenarios:
    - 2025-H1: 2025-01-01 00:00 -> 2025-06-30 23:50
    - 2025-H2: 2025-07-01 00:00 -> 2025-12-31 23:50
  These are concatenated into a single "full-year" CSV for evaluation (1 year unseen).
- Wind speed and Elspot prices are REAL data (wind from local file, price from Energi Data Service).
- Solar/hydro/load are generated in the same synthetic style as the training generator.

Scenario template selection (important)
--------------------------------------
The 2015–2024 generator created 20 scenarios saved under `training_dataset/scenario_000.csv` … `scenario_019.csv`
(one scenario per half-year; even indices are H1, odd indices are H2).

For evaluation year 2025 we want TWO half-year scenarios (H1 + H2) that are representative of the synthetic
components (solar / hydro / load), while wind + price remain real for 2025.

We select the "most representative" training scenario for H1 and H2 by computing a medoid (closest-to-center)
in a feature space of summary statistics of (solar, hydro, load):
  - mean, std, p05, p95 for each of solar/hydro/load
  - daily load amplitude mean/std

Defaults (computed from your current `training_dataset/`):
  - H1 template: scenario_012 (2021-H1)
  - H2 template: scenario_001 (2015-H2)

These templates are used ONLY for the synthetic series (solar/hydro/load) and are time-aligned to 2025-H1/H2
by timestep index (10-minute grid). Wind + price come from 2025 real data.

Inputs:
- Wind speed file (xlsx/csv) with columns:
    timestamp, wind_speed_100m
  Default: `dataset generation/wind_speed_2025.xlsx`

Outputs:
- CSV dataset with columns:
    timestamp, wind, solar, hydro, load, price, scenario, revenue, battery_energy, npv, risk, profit_label
  Default output path:
    `dataset generation/dataset2025/dataset_eval_2025_full.csv`
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    start: str
    end: str
    freq: str
    price_area: str
    wind_capacity_mw: float
    solar_capacity_mw: float
    hydro_capacity_mw: float
    carbon_tax: float
    scenario_name: str
    price_fill: Literal["interpolate_then_ffill", "ffill_only"]
    template_solar: np.ndarray | None
    template_hydro: np.ndarray | None
    template_load: np.ndarray | None


def realistic_wind_power_curve(speed_mps: np.ndarray) -> np.ndarray:
    """Return p.u. output in [0,1] from wind speed (m/s)."""
    cut_in, rated, cut_out = 3.0, 12.0, 25.0
    out = np.zeros_like(speed_mps, dtype=float)
    ramp = (speed_mps >= cut_in) & (speed_mps <= rated)
    out[ramp] = (speed_mps[ramp] - cut_in) / (rated - cut_in)
    out[(speed_mps > rated) & (speed_mps < cut_out)] = 1.0
    return np.clip(out, 0.0, 1.0)


def load_wind_speed(path: Path, *, freq: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Wind-speed file not found: {path}")

    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if "timestamp" not in df.columns:
        raise ValueError(f"{path}: expected a 'timestamp' column. cols={list(df.columns)}")

    if "wind_speed_100m" in df.columns:
        speed_col = "wind_speed_100m"
    else:
        candidates = [c for c in df.columns if c != "timestamp"]
        if not candidates:
            raise ValueError(f"{path}: no wind speed column found. cols={list(df.columns)}")
        speed_col = candidates[0]

    out = df[["timestamp", speed_col]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    out = out.rename(columns={speed_col: "wind_speed"})

    # normalize to requested resolution
    out["wind_speed"] = pd.to_numeric(out["wind_speed"], errors="coerce")
    out = out.resample(freq).asfreq()
    out["wind_speed"] = out["wind_speed"].interpolate(method="time").ffill().bfill()

    return out.reset_index()


def maybe_export_wind_csv(xlsx_path: Path, csv_path: Path) -> None:
    """
    Convenience helper: export wind_speed_2025.xlsx -> wind_speed_2025.csv
    (timestamp, wind_speed_100m).
    """
    df = pd.read_excel(xlsx_path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{xlsx_path}: missing timestamp column. cols={list(df.columns)}")
    if "wind_speed_100m" not in df.columns:
        cols = [c for c in df.columns if c != "timestamp"]
        if not cols:
            raise ValueError(f"{xlsx_path}: missing wind speed column. cols={list(df.columns)}")
        df = df.rename(columns={cols[0]: "wind_speed_100m"})
    df = df[["timestamp", "wind_speed_100m"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df.to_csv(csv_path, index=False)


def fetch_elspot_prices(
    *,
    start_iso: str,
    end_iso: str,
    price_area: str,
    freq: str,
    price_fill: Literal["interpolate_then_ffill", "ffill_only"],
) -> pd.DataFrame:
    url = "https://api.energidataservice.dk/dataset/Elspotprices"
    params = {
        "start": start_iso,
        "end": end_iso,
        "filter": json.dumps({"PriceArea": [price_area]}),
        "columns": "HourDK,SpotPriceDKK",
        "sort": "HourDK ASC",
        "timezone": "dk",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    records = resp.json().get("records", [])
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "price"])

    df["HourDK"] = pd.to_datetime(df["HourDK"])
    df = df.drop_duplicates("HourDK").set_index("HourDK").sort_index()
    df = df.rename(columns={"SpotPriceDKK": "price"})[["price"]]

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.resample(freq).asfreq()
    if price_fill == "interpolate_then_ffill":
        df["price"] = df["price"].interpolate(method="time").ffill().bfill()
    else:
        df["price"] = df["price"].ffill().bfill()

    return df.reset_index().rename(columns={"HourDK": "timestamp"})


def _require_template(arr: np.ndarray | None, name: str, n: int) -> np.ndarray:
    if arr is None:
        raise ValueError(f"Missing template for {name}.")
    if len(arr) != n:
        raise ValueError(f"Template length mismatch for {name}: got {len(arr)} expected {n}")
    return arr.astype(float, copy=False)


def simulate_battery_energy(df: pd.DataFrame) -> np.ndarray:
    battery_capacity_mwh = 10.0
    battery_power_limit_mw = 5.0
    battery_efficiency = 0.9
    battery_energy = 0.0
    out = []
    avg_price = float(df["price"].mean())

    for _, row in df.iterrows():
        surplus = max(0.0, float(row["wind"] + row["solar"] + row["hydro"] - row["load"]))
        can_charge = min(battery_power_limit_mw, surplus)
        charge_energy = can_charge * battery_efficiency

        if float(row["price"]) < avg_price:
            battery_energy += charge_energy
        else:
            discharge = min(battery_power_limit_mw, battery_energy)
            battery_energy -= discharge

        battery_energy = max(0.0, min(battery_capacity_mwh, battery_energy))
        out.append(battery_energy)

    return np.asarray(out, dtype=float)


def generate_dataset(cfg: Config, wind_speed_path: Path) -> pd.DataFrame:
    timestamps = pd.date_range(start=cfg.start, end=cfg.end, freq=cfg.freq)

    wind_df = load_wind_speed(wind_speed_path, freq=cfg.freq).set_index("timestamp")
    wind_speed = wind_df.reindex(timestamps, method="nearest")["wind_speed"].to_numpy(dtype=float)
    wind_pu = realistic_wind_power_curve(wind_speed)

    # Use template synthetic series (already in MW) for solar/hydro/load
    n = len(timestamps)
    solar = _require_template(cfg.template_solar, "solar", n)
    hydro = _require_template(cfg.template_hydro, "hydro", n)
    load = _require_template(cfg.template_load, "load", n)

    prices_df = fetch_elspot_prices(
        start_iso=pd.Timestamp(cfg.start).strftime("%Y-%m-%dT%H:%M"),
        end_iso=pd.Timestamp(cfg.end).strftime("%Y-%m-%dT%H:%M"),
        price_area=cfg.price_area,
        freq=cfg.freq,
        price_fill=cfg.price_fill,
    ).set_index("timestamp")
    price = prices_df.reindex(timestamps, method="nearest")["price"].to_numpy(dtype=float)

    wind = wind_pu * float(cfg.wind_capacity_mw)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "wind": wind,
            "solar": solar,
            "hydro": hydro,
            "load": load,
            "price": price,
            "scenario": cfg.scenario_name,
        }
    )

    df["revenue"] = (df["wind"] + df["solar"] + df["hydro"]) * df["price"]
    df["battery_energy"] = simulate_battery_energy(df)

    npv = float(df["revenue"].sum() - (cfg.wind_capacity_mw * 1000.0 + 600000.0 + 1200000.0))
    risk = float(df["revenue"].std() / max(df["revenue"].mean(), 1e-8))
    df["npv"] = npv
    df["risk"] = risk
    df["profit_label"] = int(npv > 0)

    return df


def _generate_half(
    *,
    wind_speed_path: Path,
    start: str,
    end: str,
    freq: str,
    price_area: str,
    scenario_name: str,
    wind_capacity_mw: float,
    solar_capacity_mw: float,
    hydro_capacity_mw: float,
    carbon_tax: float,
    price_fill: Literal["interpolate_then_ffill", "ffill_only"],
    template_solar: np.ndarray,
    template_hydro: np.ndarray,
    template_load: np.ndarray,
) -> pd.DataFrame:
    cfg = Config(
        start=start,
        end=end,
        freq=freq,
        price_area=price_area,
        wind_capacity_mw=wind_capacity_mw,
        solar_capacity_mw=solar_capacity_mw,
        hydro_capacity_mw=hydro_capacity_mw,
        carbon_tax=carbon_tax,
        scenario_name=scenario_name,
        price_fill=price_fill,
        template_solar=template_solar,
        template_hydro=template_hydro,
        template_load=template_load,
    )
    return generate_dataset(cfg, wind_speed_path)


def load_training_template(
    training_dir: Path, scenario_id: int, expected_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = training_dir / f"scenario_{scenario_id:03d}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Training scenario file not found: {p}")
    df = pd.read_csv(p, parse_dates=["timestamp"])
    if len(df) != expected_len:
        raise ValueError(f"{p}: unexpected row count {len(df)} (expected {expected_len})")
    for c in ("solar", "hydro", "load"):
        if c not in df.columns:
            raise ValueError(f"{p}: missing column '{c}'. cols={list(df.columns)}")
    return (
        df["solar"].to_numpy(dtype=float),
        df["hydro"].to_numpy(dtype=float),
        df["load"].to_numpy(dtype=float),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate evaluation dataset for year 2025 (10-minute).")
    ap.add_argument(
        "--wind_speed",
        type=str,
        default=str(BASE_DIR / "wind_speed_2025.csv"),
        help="Wind speed input file (xlsx/csv) with columns timestamp, wind_speed_100m (default: dataset generation/wind_speed_2025.csv)",
    )
    ap.add_argument(
        "--export_wind_csv",
        action="store_true",
        help="If set, export wind_speed_2025.xlsx -> wind_speed_2025.csv inside dataset generation/ before running.",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default=str(BASE_DIR / "dataset2025"),
        help="Output directory (default: dataset generation/dataset2025)",
    )
    ap.add_argument("--output_name", type=str, default="dataset_eval_2025_full.csv", help="Output CSV filename")
    ap.add_argument("--freq", type=str, default="10min", help="Resolution (default: 10min)")
    ap.add_argument("--price_area", type=str, default="DK1", help="Elspot PriceArea (default: DK1)")
    ap.add_argument("--wind_capacity_mw", type=float, default=1500.0, help="Wind capacity in MW (default: 1500)")
    ap.add_argument("--solar_capacity_mw", type=float, default=1000.0, help="Solar capacity in MW (default: 1000)")
    ap.add_argument("--hydro_capacity_mw", type=float, default=1000.0, help="Hydro capacity in MW (default: 1000)")
    ap.add_argument("--carbon_tax", type=float, default=0.0, help="Carbon tax placeholder (kept for parity; default: 0)")
    ap.add_argument(
        "--scenario",
        type=str,
        default="scenario_2025_OOS_REALPRICE_REALWIND",
        help="Scenario name string stored in output",
    )
    ap.add_argument(
        "--price_fill",
        type=str,
        default="interpolate_then_ffill",
        choices=["interpolate_then_ffill", "ffill_only"],
        help="How to fill missing hourly prices before 10-min resampling",
    )
    ap.add_argument("--training_dir", type=str, default=str(BASE_DIR.parent / "training_dataset"),
                    help="Directory containing training scenarios scenario_000..019.csv (default: training_dataset/)")
    ap.add_argument("--h1_template", type=int, default=12,
                    help="Training scenario id to use as synthetic template for 2025-H1 (default: 12)")
    ap.add_argument("--h2_template", type=int, default=1,
                    help="Training scenario id to use as synthetic template for 2025-H2 (default: 1)")
    ap.add_argument("--write_unseendata", action="store_true",
                    help="If set, also write the combined full-year CSV to evaluation_dataset/unseendata.csv")
    args = ap.parse_args()

    wind_path = Path(args.wind_speed)
    if args.export_wind_csv:
        # Only meaningful if the user points --wind_speed at an Excel file.
        if wind_path.suffix.lower() in (".xlsx", ".xls") and wind_path.exists():
            maybe_export_wind_csv(wind_path, BASE_DIR / "wind_speed_2025.csv")
        else:
            raise FileNotFoundError(
                "--export_wind_csv requires --wind_speed to point to an existing .xlsx file. "
                f"Got: {wind_path}"
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name

    # Generate H1 + H2 (one scenario per 6 months) and concatenate into a full-year evaluation CSV
    training_dir = Path(args.training_dir)

    # Load solar/hydro/load templates from training scenarios
    h1_len = 181 * 24 * 6  # 2025-H1: Jan 1 -> Jun 30, 10min
    h2_len = 184 * 24 * 6  # 2025-H2: Jul 1 -> Dec 31, 10min
    h1_solar, h1_hydro, h1_load = load_training_template(training_dir, int(args.h1_template), h1_len)
    h2_solar, h2_hydro, h2_load = load_training_template(training_dir, int(args.h2_template), h2_len)

    df_h1 = _generate_half(
        wind_speed_path=wind_path,
        start="2025-01-01 00:00",
        end="2025-06-30 23:50",
        freq=args.freq,
        price_area=args.price_area,
        scenario_name="scenario_2025_H1",
        wind_capacity_mw=float(args.wind_capacity_mw),
        solar_capacity_mw=float(args.solar_capacity_mw),
        hydro_capacity_mw=float(args.hydro_capacity_mw),
        carbon_tax=float(args.carbon_tax),
        price_fill=args.price_fill,
        template_solar=h1_solar,
        template_hydro=h1_hydro,
        template_load=h1_load,
    )
    df_h2 = _generate_half(
        wind_speed_path=wind_path,
        start="2025-07-01 00:00",
        end="2025-12-31 23:50",
        freq=args.freq,
        price_area=args.price_area,
        scenario_name="scenario_2025_H2",
        wind_capacity_mw=float(args.wind_capacity_mw),
        solar_capacity_mw=float(args.solar_capacity_mw),
        hydro_capacity_mw=float(args.hydro_capacity_mw),
        carbon_tax=float(args.carbon_tax),
        price_fill=args.price_fill,
        template_solar=h2_solar,
        template_hydro=h2_hydro,
        template_load=h2_load,
    )

    df = pd.concat([df_h1, df_h2], ignore_index=True)
    df.to_csv(out_path, index=False)

    ts0 = pd.to_datetime(df["timestamp"].iloc[0])
    ts1 = pd.to_datetime(df["timestamp"].iloc[-1])
    print(f"Wrote: {out_path} rows={len(df):,} start={ts0} end={ts1}")

    if args.write_unseendata:
        eval_dir = BASE_DIR.parent / "evaluation_dataset"
        eval_dir.mkdir(parents=True, exist_ok=True)
        unseen_path = eval_dir / "unseendata.csv"
        df.to_csv(unseen_path, index=False)
        print(f"Wrote: {unseen_path} rows={len(df):,}")


if __name__ == "__main__":
    main()

