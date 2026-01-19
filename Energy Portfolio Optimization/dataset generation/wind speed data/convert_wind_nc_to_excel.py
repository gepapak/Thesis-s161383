"""
Convert ERA5 wind (u100/v100) netCDF files to 10-minute wind speed time series and export to Excel.

Expected input schema (matches your API/*.nc files):
- data_vars: u100, v100
- time coord: valid_time (or time)
- optional dims: latitude/longitude (averaged if present)

Outputs:
- An Excel file with columns: timestamp, wind_speed_100m

Notes (how these .nc files were originally downloaded)
------------------------------------------------------
Your `API/*.nc` files appear to come from Copernicus CDS ERA5 single levels using 100m wind components.

Example CDS request (from the old notebook, kept here for reproducibility; DO NOT embed API keys in code):

    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "100m_u_component_of_wind",
            "100m_v_component_of_wind"
        ],
        "year": ["2016"],
        "month": ["01", ..., "12"],
        "day": ["01", ..., "31"],
        "time": ["00:00", ..., "23:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        # Copernicus area format is [N, W, S, E]
        "area": [55.6, 7.48, 55.5, 7.58],
    }

We verified that your 2014–2024 and 2025 (full year) files are a 1x1 point at lat=55.5, lon=7.48.

Full-year 10-minute grid behavior
---------------------------------
ERA5 files here are hourly (they end at Dec 31 23:00). For downstream pipelines it's often easier if each year
is on a *complete* 10-minute grid, i.e.:

  - start: Jan 1 00:00
  - end:   Dec 31 23:50

That is 52,560 rows for a non-leap year (365 * 24 * 6).

To achieve this, this script:
  - linearly interpolates within the year to 10-minute resolution
  - linearly extrapolates ONLY the final partial hour (Dec 31 23:10..23:50) using the slope from the last two
    hourly points (22:00 and 23:00). This avoids dropping the last 50 minutes of the year.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr


def _detect_time_coord(ds: xr.Dataset) -> str:
    if "valid_time" in ds.coords:
        return "valid_time"
    if "time" in ds.coords:
        return "time"
    raise ValueError(f"Could not find a time coordinate. coords={list(ds.coords)}")


def nc_to_10min_df(
    nc_path: Path,
    *,
    year: Optional[int] = None,
    freq: str = "10min",
) -> pd.DataFrame:
    ds = xr.open_dataset(nc_path)

    # Collapse spatial dims (your files are 1x1 but keep this robust)
    for dim in ("latitude", "longitude"):
        if dim in ds.dims:
            # only mean if the dim length > 1 (avoid xarray warnings / no-op)
            if int(ds.sizes.get(dim, 1)) > 1:
                ds = ds.mean(dim=[dim])

    time_col = _detect_time_coord(ds)

    required = {"u100", "v100"}
    missing = required.difference(set(ds.data_vars))
    if missing:
        raise ValueError(f"{nc_path}: missing required variables: {sorted(missing)}. Found={list(ds.data_vars)}")

    wind_speed = np.sqrt(ds["u100"] ** 2 + ds["v100"] ** 2)
    df = wind_speed.to_dataframe(name="wind_speed_100m").reset_index()

    # Normalize timestamp column name
    df = df.rename(columns={time_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if year is not None:
        df = df[df["timestamp"].dt.year == int(year)]

    # Resample/interpolate to target frequency
    df = df.sort_values("timestamp").set_index("timestamp")
    # Ensure numeric dtype before interpolation
    df = df.infer_objects(copy=False)
    df["wind_speed_100m"] = pd.to_numeric(df["wind_speed_100m"], errors="coerce")

    # First, create the regular 10-minute series over the available index range
    df_10 = df.resample(freq).asfreq()
    df_10["wind_speed_100m"] = df_10["wind_speed_100m"].interpolate(method="time")

    # If a year was requested, force a full-year 10-minute grid (through Dec 31 23:50)
    if year is not None:
        y = int(year)
        start = pd.Timestamp(y, 1, 1, 0, 0, 0)
        end = pd.Timestamp(y, 12, 31, 23, 50, 0)
        full_index = pd.date_range(start=start, end=end, freq=freq)

        df_10 = df_10.reindex(full_index)
        df_10["wind_speed_100m"] = df_10["wind_speed_100m"].interpolate(method="time")

        # Linear extrapolation for any remaining edge NaNs (notably Dec 31 23:10..23:50)
        s = df_10["wind_speed_100m"]
        if s.isna().any():
            notna = s.dropna()
            if len(notna) >= 2:
                # Extrapolate tail
                last_idx = notna.index[-1]
                if s.loc[last_idx:].isna().any():
                    t1, t2 = notna.index[-2], notna.index[-1]
                    y1, y2 = float(notna.iloc[-2]), float(notna.iloc[-1])
                    dt = (t2.value - t1.value)
                    slope = 0.0 if dt == 0 else (y2 - y1) / dt
                    tail = s.loc[last_idx:].copy()
                    for ts in tail.index[tail.isna()]:
                        s.loc[ts] = y2 + slope * (ts.value - t2.value)

                # Extrapolate head (unlikely needed, but keep symmetric)
                first_idx = notna.index[0]
                if s.loc[:first_idx].isna().any():
                    t1, t2 = notna.index[0], notna.index[1]
                    y1, y2 = float(notna.iloc[0]), float(notna.iloc[1])
                    dt = (t2.value - t1.value)
                    slope = 0.0 if dt == 0 else (y2 - y1) / dt
                    head = s.loc[:first_idx].copy()
                    for ts in head.index[head.isna()]:
                        s.loc[ts] = y1 + slope * (ts.value - t1.value)

                df_10["wind_speed_100m"] = s

    df = df_10.reset_index().rename(columns={"index": "timestamp"})

    # Keep only the requested columns
    return df[["timestamp", "wind_speed_100m"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input .nc file (e.g., API/2014.nc)")
    ap.add_argument("--output", required=True, help="Path to output .xlsx file (e.g., API/wind_speed_100m_10min_2014.xlsx)")
    ap.add_argument("--year", type=int, default=None, help="Optional year filter (e.g., 2014). Useful if input spans multiple years.")
    ap.add_argument("--freq", type=str, default="10min", help="Resample frequency (default: 10min)")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = nc_to_10min_df(in_path, year=args.year, freq=args.freq)

    # Excel export
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="wind_speed_100m")

    # Keep console output ASCII-safe on Windows (cp1252)
    print(f"Wrote: {out_path} ({len(df):,} rows)")


if __name__ == "__main__":
    main()

