"""
demand.py

Load and process hot-water draw profiles. Aggregates 1-minute profiles to 5-minute steps.
"""
import pandas as pd
import numpy as np

class DemandProfile:
    """
    Convert volume profiles (1-min) to 5-min energy draws.
    """
    def __init__(self, profile_path: str, tank_setpoint: float, temp_in: float):
        # 1) Auto-detect separator; coerce numerics
        df = pd.read_csv(profile_path, sep=None, engine="python")
        # columns like Summer_Water_Consumption, Autumn_Water_Consumption, ...
        season_cols = [c for c in df.columns if c.lower().endswith("water_consumption")]
        if not season_cols:
            raise ValueError(
                f"No '*Water_Consumption' columns found in {profile_path}. "
                f"Got columns: {list(df.columns)[:8]}..."
            )

        for c in season_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        # Sum all seasons to one per-minute volume stream (litres/min)
        vol_1min = df[season_cols].sum(axis=1)

        # Build a 1-minute index starting 2024-01-01 (length = file length)
        idx_1min = pd.date_range("2024-01-01", periods=len(vol_1min), freq="min")
        tmp = pd.DataFrame({"volume_l": vol_1min.values}, index=idx_1min)

        # 2) Aggregate to 5-minute with RIGHT labels to match irradiance 'period_end'
        self.df = tmp.resample("5min", label="right", closed="right").sum()

        # Keep for debugging
        self.total_litres = float(self.df["volume_l"].sum())

        self.tank_setpoint = float(tank_setpoint)
        self.temp_in = float(temp_in)

    def get_draw_energy(self) -> pd.Series:
        # E = m c ΔT; 1 L ≈ 1 kg
        mass_kg = self.df["volume_l"]
        c = 4184.0
        dT = self.tank_setpoint - self.temp_in
        energy_J = mass_kg * c * dT
        return energy_J / (3600.0 * 1000.0)  # kWh
