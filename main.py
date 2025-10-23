"""
main.py

Run simulations for one or more locations and one or more demand categories.
Outputs one Excel workbook per location.

This version:
- Uses flat USD/kWh tariffs by city via tariff.get_city_usd_rate
- Removes any CSV/TOU tariff loading
- Renames all currency outputs to *USD
"""
import os
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import os
from glob import glob
from typing import Dict, List


import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from config import (
    ROOT_DIR, SOLAR_DATA_DIR, USER_DATA_DIR, MODULE_NAME, NUM_PANELS,
    SELECTED_PROFILES, SELECTED_LOCATIONS, TILT, AZIMUTH,RUN_SWEEP
)


from simulate import simulate_household_efficient
from tariff import get_city_usd_rate



# -----------------------------
# Location → PV system params
# -----------------------------
_CITY_PARAMS: Dict[str, Dict] = {
    # Southern Hemisphere → face NORTH (azimuth=0°), Northern → face SOUTH (180°)
    "CapeTown":     {"latitude": -33.9249, "longitude": 18.4241, "timezone": "Africa/Johannesburg", "tilt": 30, "azimuth": 0},
    "Johannesburg": {"latitude": -26.2041, "longitude": 28.0473, "timezone": "Africa/Johannesburg", "tilt": 30, "azimuth": 0},
    "Lusaka":       {"latitude": -15.3875, "longitude": 28.3228, "timezone": "Africa/Lusaka",        "tilt": 30, "azimuth": 0},
    "Luanda":       {"latitude":  -8.8390, "longitude": 13.2894, "timezone": "Africa/Luanda",        "tilt": 30, "azimuth": 0},
    "Kinshasa":     {"latitude":  -4.4419, "longitude": 15.2663, "timezone": "Africa/Kinshasa",      "tilt": 30, "azimuth": 0},
    "Nairobi":      {"latitude":  -1.2921, "longitude": 36.8219, "timezone": "Africa/Nairobi",       "tilt":  30, "azimuth": 0},
    "Lagos":        {"latitude":   6.5244, "longitude":  3.3792, "timezone": "Africa/Lagos",         "tilt": 30, "azimuth": 0},
}


def get_system_params(city: str) -> dict:
    if city not in _CITY_PARAMS:
        raise KeyError(f"No PV system params for city: {city}")
    base = dict(_CITY_PARAMS[city])  # lat/lon/tz from the table
    # OVERRIDE angle with config knobs:
    base.update({
        "tilt": int(TILT),
        "azimuth": int(AZIMUTH),
        "num_panels": NUM_PANELS,
    })
    return base


# === Tilt × Azimuth sweep for one or more cities ===
def run_angle_sweep(
    cities=None,
    tilts=range(0, 61, 5),        # 0..60° step 5
    azimuths=range(0, 361, 15),   # 0..360° step 15
    n_jobs=-1
):
    if cities is None:
        cities = [c for c in SELECTED_LOCATIONS]

    for city in cities:
        print(f"\n====== Angle sweep: {city} | tilts {min(tilts)}..{max(tilts)}, az {min(azimuths)}..{max(azimuths)} ======")

        # Tariff
        try:
            flat_rate_usd = get_city_usd_rate(city)
        except KeyError as e:
            print(f"  ✖ {e}")
            continue

        # Irradiance → local naive
        irr_path = get_irradiance_path(city)
        if not os.path.exists(irr_path):
            print(f"  ✖ Irradiance file not found: {irr_path}")
            continue

        irr = load_irradiance_all(irr_path)
        local_tz = _get_city_timezone(city)   # e.g. "Africa/Johannesburg"
        irr = _to_local_naive(irr, local_tz)  # convert UTC→local, then drop tz

        print(
            f"  Irradiance sanity check for {city}:",
            "tz:", irr.index.tz,
            "| ghi>0:", int((irr['ghi'] > 0).sum()),
            "| ghi max:", float(irr['ghi'].max()),
            "| dni max:", float(irr['dni'].max())
        )

        all_rows = []
        for tilt in tilts:

            for az in azimuths:
                sys = get_system_params(city)
                sys["tilt"] = int(tilt)
                sys["azimuth"] = int(az)

                for cat in SELECTED_PROFILES:
                    files = list_profile_files(cat)
                    if not files:
                        continue

                    res_list = Parallel(n_jobs=n_jobs, backend="loky")(
                        delayed(simulate_household_efficient)(pf, irr, flat_rate_usd, sys, city=city)
                        for pf in files
                    )
                    ok = [r for r in res_list if r.get('success')]
                    for r in ok:
                        row = {
                            'city': city,
                            'tilt': tilt,
                            'azimuth': az,
                            'category': cat,
                            'profile': r['profile'],
                            **r['kpis'],
                            'cost_USD': r['cost_USD'],
                        }
                        # helpers
                        if 'annual_grid_kwh' in row and 'annual_solar_kwh' in row:
                            row['total_heating_kwh'] = row['annual_grid_kwh'] + row['annual_solar_kwh']
                        if 'annual_demand_kwh' in row and 'total_heating_kwh' in row:
                            row['energy_gap_kwh'] = row['total_heating_kwh'] - row['annual_demand_kwh']
                        all_rows.append(row)

        if not all_rows:
            print("  ⚠ No results from angle sweep.")
            continue

        df = pd.DataFrame(all_rows)

        # Rounding for readability
        round_map = {
            'cost_USD': 2, 'annual_solar_savings_USD': 2, 'cost_without_solar_USD': 2,
            'annual_demand_kwh': 1, 'annual_grid_kwh': 1, 'annual_solar_kwh': 1,
            'total_heating_kwh': 1, 'energy_gap_kwh': 1,
            'avg_temp': 1, 'cold_draw_pct': 1, 'savings_percentage': 1
        }
        for k, dp in round_map.items():
            if k in df.columns:
                df[k] = df[k].round(dp)

        # Summary (mean across profiles/categories) — keep ALL numeric KPIs
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        grp = df.groupby(['city','tilt','azimuth'], as_index=False)[num_cols].mean(numeric_only=True)
        summary = grp.sort_values(['tilt','azimuth'])

        # "Best" by solar and by savings
        best_solar = summary.loc[summary.groupby('city')['annual_solar_kwh'].idxmax(), ['city','tilt','azimuth','annual_solar_kwh']]
        if 'annual_solar_savings_USD' in summary.columns:
            best_savings = summary.loc[summary.groupby('city')['annual_solar_savings_USD'].idxmax(),
                                       ['city','tilt','azimuth','annual_solar_savings_USD']]
        else:
            best_savings = pd.DataFrame(columns=['city','tilt','azimuth','annual_solar_savings_USD'])

        # Save Excel
        out_xlsx = os.path.join(ROOT_DIR, f"results_{city}_angle_sweep.xlsx")
        with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="by_profile", index=False)
            summary.to_excel(writer, sheet_name="summary_by_angles", index=False)
            best_solar.to_excel(writer, sheet_name="best_by_solar_kwh", index=False)
            if not best_savings.empty:
                best_savings.to_excel(writer, sheet_name="best_by_savings_usd", index=False)
        print(f"  ✅ Angle sweep saved: {out_xlsx}")

        # Heatmaps: mean annual_solar_kwh and mean annual_solar_savings_USD vs (tilt, azimuth)
        try:
            s = summary.copy()

            def plot_heat(metric, fname_stub, cmap='viridis'):
                if metric not in s.columns:
                    print(f"  ⚠ Metric '{metric}' not in summary; skipping heatmap.")
                    return
                piv = s.pivot(index='tilt', columns='azimuth', values=metric).sort_index().sort_index(axis=1)
                fig, ax = plt.subplots(figsize=(8, 5))
                im = ax.imshow(
                    piv.values, origin='lower', aspect='auto', cmap=cmap,
                    extent=[piv.columns.min(), piv.columns.max(), piv.index.min(), piv.index.max()]
                )
                ax.set_xlabel('Azimuth (°)')
                ax.set_ylabel('Tilt (°)')
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label(metric)
                ax.set_title(f"{city}: mean {metric} vs tilt×azimuth")
                ax.grid(False)
                out_png = os.path.join(ROOT_DIR, f"{fname_stub}_{city}.png")
                fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
                print(f"  📈 Saved heatmap: {out_png}")

            plot_heat('annual_solar_kwh',        'angle_solar_kwh',   cmap='viridis')
            plot_heat('annual_solar_savings_USD','angle_savings_usd', cmap='magma')
            plot_heat('pv_generated_kwh', 'angle_pv_generated_kwh', cmap='plasma')

        except Exception as e:
            print(f"  ⚠ Plotting failed: {e}")
            
            
def get_irradiance_path(city: str) -> str:
    """
    Strict resolver:
      <SOLAR_DATA_DIR>/<city>/<city>.csv
      <SOLAR_DATA_DIR>/<city>/<city>_meteo_5min.csv
    If those aren't found, pick any .csv inside the <city> subfolder (shortest path first).
    """
    base = os.path.join(SOLAR_DATA_DIR, city)

    candidates = [
        os.path.join(base, f"{city}.csv"),
        os.path.join(base, f"{city}_meteo_5min.csv"),
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    # Fallback: any CSV in the city subfolder
    hits = glob(os.path.join(base, "*.csv"))
    if hits:
        hits.sort(key=lambda x: (len(x), x.lower()))
        return hits[0]

    raise FileNotFoundError(
        f"Irradiance file not found for city '{city}'.\n"
        f"Expected one of:\n  - {candidates[0]}\n  - {candidates[1]}\n"
        f"No CSVs found in folder: {base}"
    )

def _coerce_and_select_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # Prefer actual/modelled columns; if missing, fall back to clearsky_*
    for col, fallback in (("dni", "clearsky_dni"),
                          ("dhi", "clearsky_dhi"),
                          ("ghi", "clearsky_ghi")):
        if col in df.columns:
            out[col] = df[col].astype(float)
        elif fallback in df.columns:
            out[col] = df[fallback].astype(float)
        else:
            raise ValueError(f"Missing both '{col}' and '{fallback}' in irradiance file.")

    # Ambient temperature: dewpoint_temp as proxy if no air temp is available
    if "temp_air" in df.columns:
        out["temp_air"] = df["temp_air"].astype(float)
    elif "dewpoint_temp" in df.columns:
        out["temp_air"] = df["dewpoint_temp"].astype(float)
    else:
        out["temp_air"] = 20.0  # sensible default

    # Wind speed: prefer 10 m, else 100 m, else default
    if "wind_speed_10m" in df.columns:
        out["wind_speed"] = df["wind_speed_10m"].astype(float)
    elif "wind_speed_100m" in df.columns:
        out["wind_speed"] = df["wind_speed_100m"].astype(float)
    else:
        out["wind_speed"] = 1.0

    return out[["ghi","dni","dhi","temp_air","wind_speed"]]

def load_irradiance_all(path_or_dir: str) -> pd.DataFrame:
    """
    Load irradiance files that have a 'period_end' timestamp and the columns shown above.
    Accepts either a directory of monthly CSVs or a single CSV file.
    Returns a UTC-aware DatetimeIndex at 5-min resolution with columns:
    ghi/dni/dhi/temp_air/wind_speed (irradiance in W/m², temp in °C, wind in m/s).
    """
    # Gather files
    if os.path.isdir(path_or_dir):
        files = sorted(glob(os.path.join(path_or_dir, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSVs found in directory: {path_or_dir}")
    else:
        if not os.path.exists(path_or_dir):
            raise FileNotFoundError(path_or_dir)
        files = [path_or_dir]

    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        if "period_end" not in df.columns:
            raise RuntimeError(f"'{f}': missing 'period_end' column.")
        ts = pd.to_datetime(df["period_end"], utc=True, errors="coerce")
        df = df.loc[ts.notna()].copy()
        df.index = ts[ts.notna()]

        df = _coerce_and_select_cols(df)
        frames.append(df)

    irr = pd.concat(frames, axis=0).sort_index()
    irr = irr[~irr.index.duplicated(keep="first")]

    # Uniform 5-min grid; interpolate irradiance, carry met variables
    full_idx = pd.date_range(irr.index.min().floor("5min"),
                             irr.index.max().ceil("5min"),
                             freq="5min", tz="UTC")
    irr = irr.reindex(full_idx)
    irr[["ghi","dni","dhi"]] = irr[["ghi","dni","dhi"]].interpolate("time", limit_direction="both")
    irr[["temp_air","wind_speed"]] = irr[["temp_air","wind_speed"]].ffill().bfill()

    return irr



def list_profile_files(category: str) -> List[str]:
    """Return sorted list of CSV files for the given demand category."""
    p = os.path.join(USER_DATA_DIR, category)
    files = sorted(glob(os.path.join(p, "*.csv")))
    if not files:
        print(f"  ⚠ No profiles found in {p}")
    return files

def _get_city_timezone(city: str) -> str:
    if city not in _CITY_PARAMS or "timezone" not in _CITY_PARAMS[city]:
        raise KeyError(f"No timezone configured for city: {city}")
    return _CITY_PARAMS[city]["timezone"]


def _to_local_naive(irr: pd.DataFrame, tz: str) -> pd.DataFrame:
    """
    Convert tz-aware UTC index → local timezone, then drop tz (naive).
    If index is naive for some reason, assume it's UTC first.
    """
    idx = irr.index
    if idx.tz is None:
        irr = irr.tz_localize("UTC")
    irr = irr.tz_convert(tz).tz_localize(None)
    return irr

def main():
    selected_locations = [c for c in SELECTED_LOCATIONS]
    if not selected_locations:
        print("No locations selected. Set SELECTED_LOCATIONS in config.py")
        return
    
    if RUN_SWEEP:
        # Run the sweep and STOP — no normal run afterwards
        run_angle_sweep(
            cities=selected_locations,
            tilts=range(0, 61, 5),       # 0..60° step 5
            azimuths=range(0, 361, 15),  # 0..360° step 15
            n_jobs=-1
        )
        return

    for city in selected_locations:
        print(f"\n====== Running location: {city} ======")

        # Flat USD/kWh for this city (raises if unknown)
        try:
            flat_rate_usd = get_city_usd_rate(city)
        except KeyError as e:
            print(f"  ✖ {e}")
            continue

        # Irradiance
        irr_path = get_irradiance_path(city)
        if not os.path.exists(irr_path):
            print(f"  ✖ Irradiance file not found: {irr_path}")
            continue

        irr = load_irradiance_all(irr_path)      # pass the FOLDER with all months, or a full-year CSV
        local_tz = _get_city_timezone(city)
        irr = _to_local_naive(irr, local_tz)     # your existing helper: UTC→local tz, then naive

        print("IRR RANGE:", irr.index.min(), "→", irr.index.max(), f"(steps={len(irr)})")
        if len(irr) < 100_000:
            print("⚠ Detected partial year (~105,120 steps expected for full year @5-min). "
                "Point irr_path at the directory containing all months.")

        # PV system parameters
        try:
            system_params = get_system_params(city)
        except Exception as e:
            print(f"  ✖ {e}")
            continue
      
        all_category_results = {}

        for cat in SELECTED_PROFILES:
            print(f"  → Category: {cat}")
            profile_files = list_profile_files(cat)
            if not profile_files:
                continue

            res_list = Parallel(n_jobs=-1, backend="loky")(
                delayed(simulate_household_efficient)(pf, irr, flat_rate_usd, system_params, city=city)
                for pf in profile_files
            )

            ok = [r for r in res_list if r.get('success')]
            if not ok:
                print(f"  ⚠ No successful runs for category {cat}")
                continue

            # Build a DataFrame with ALL KPIs + id columns
            df_cat = pd.DataFrame([
                {**r['kpis'], 'cost_USD': r['cost_USD'], 'profile': r['profile'], 'category': cat}
                for r in ok
            ])
            
            # Order columns (only those that exist)
            desired_cols = [
                'profile', 'category',
                'annual_demand_kwh', 'annual_grid_kwh', 'annual_solar_kwh', 'solar_fraction',
                'solar_used_when_needed_kwh', 'grid_used_when_needed_kwh',
                'solar_heating_energy_fraction', 'solar_heating_event_fraction',
                'cold_draw_pct', 'avg_temp',
                'cost_USD', 'annual_solar_savings_USD', 'cost_without_solar_USD', 'savings_percentage',
            ]
            cols = [c for c in desired_cols if c in df_cat.columns] + [c for c in df_cat.columns if c not in desired_cols]
            df_cat = df_cat[cols]
            df_cat['total_heating_kwh'] = df_cat['annual_grid_kwh'] + df_cat['annual_solar_kwh']
            df_cat['energy_gap_kwh'] = df_cat['total_heating_kwh'] - df_cat['annual_demand_kwh']
            # Round values (money 2dp; energy/temp 1dp; leave fractions raw, pct columns are already 0–100)
            round_map = {
                'cost_USD': 2, 'annual_solar_savings_USD': 2, 'cost_without_solar_USD': 2,
                'annual_demand_kwh': 1, 'annual_grid_kwh': 1, 'annual_solar_kwh': 1,
                'solar_used_when_needed_kwh': 1, 'grid_used_when_needed_kwh': 1,
                'avg_temp': 1, 'cold_draw_pct': 1, 'savings_percentage': 1,
            }
            for k, dp in round_map.items():
                if k in df_cat.columns:
                    df_cat[k] = df_cat[k].round(dp)

            all_category_results[cat] = df_cat

        # Save workbook per city with formats
        if all_category_results:
            out_xlsx = os.path.join(ROOT_DIR, f"results_{city}.xlsx")
            with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
                for cat, df_cat in all_category_results.items():
                    df_cat.to_excel(writer, sheet_name=cat, index=False)
                    ws = writer.sheets[cat]
                    wb = writer.book

                    # Formats
                    fmt_usd = wb.add_format({'num_format': '$#,##0.00'})
                    fmt_kwh = wb.add_format({'num_format': '0.0'})
                    fmt_pct = wb.add_format({'num_format': '0.0%'})
                    fmt_deg = wb.add_format({'num_format': '0.0'})

                    # Helper to set a column's format by name (if present)
                    def set_col_fmt(col_name, fmt):
                        if col_name in df_cat.columns:
                            idx = df_cat.columns.get_loc(col_name)
                            ws.set_column(idx, idx, None, fmt)

                    # Apply formats
                    for c in ('cost_USD', 'annual_solar_savings_USD', 'cost_without_solar_USD'):
                        set_col_fmt(c, fmt_usd)
                    for c in ('annual_demand_kwh', 'annual_grid_kwh', 'annual_solar_kwh',
                            'solar_used_when_needed_kwh', 'grid_used_when_needed_kwh'):
                        set_col_fmt(c, fmt_kwh)
                    # Fractions (0–1) shown as percent
                    for c in ('solar_fraction', 'solar_heating_energy_fraction', 'solar_heating_event_fraction'):
                        set_col_fmt(c, fmt_pct)
                    # Percent already 0–100; show as number with 1dp
                    set_col_fmt('cold_draw_pct', fmt_deg)
                    set_col_fmt('savings_percentage', fmt_deg)
                    # Temperature
                    set_col_fmt('avg_temp', fmt_deg)

            print(f"  ✅ Saved: {out_xlsx}")
                        # -------------------------
            # Monthly-hourly PV summary for entire city (12x24)
            # -------------------------
            try:
                print(f"  ⏳ Building monthly-hourly PV summary for {city}...")

                # Load irradiance again to get timestamps
                irr = load_irradiance_all(get_irradiance_path(city))
                irr = _to_local_naive(irr, _get_city_timezone(city))

                # Approximate PV output using GHI scaling
                # (if you prefer, you can pull pv_kw arrays directly from simulate.py returns)
                df_pv = pd.DataFrame({'ghi': irr['ghi']}, index=irr.index)
                df_pv['pv_kw'] = df_pv['ghi'] / df_pv['ghi'].max() * NUM_PANELS * 0.42  # rough 420 W/panel
                df_pv['pv_kwh'] = df_pv['pv_kw'] * (5 / 60)  # convert 5-min kW → kWh

                # Group by month × hour
                pv_pivot = (
                    df_pv.groupby([df_pv.index.month, df_pv.index.hour])['pv_kwh']
                    .mean()
                    .unstack(level=1)
                )

                # Label months and hours
                pv_pivot.index = [
                    'Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec'
                ]
                pv_pivot.columns = [f'{h:02d}:00' for h in range(24)]

                # Save once per city
                out_summary = os.path.join(ROOT_DIR, f"pv_summary_{city}_12x24.xlsx")
                pv_pivot.to_excel(out_summary, sheet_name='MonthlyHourlyPV')
                print(f"  ✅ Monthly-hourly PV summary saved: {out_summary}")

            except Exception as e:
                print(f"  ⚠ Could not build monthly-hourly PV summary for {city}: {e}")

        else:
            print("  ⚠ No successful runs to save.")
    


if __name__ == "__main__":
    main()
