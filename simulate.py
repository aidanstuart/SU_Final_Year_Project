# simulate.py

import os
import gc
from typing import Dict
import numpy as np
import pandas as pd
import pvlib
from tank import StratifiedTank
from demand import DemandProfile
from pv_module import PVModule
from utils import CHUNK_SIZE
from config import (
    TANK_PARAMS, SIM_PARAMS, MODULE_NAME,
    PERMANENT_LOAD_TEST, DEMAND_REPEAT_PER_SEASON, REVERSE_SEASONS_FOR,DIAG_PRINTS,
)
# --- Seasonal demand helpers (build full-year from 1-month-per-season CSV) ---

# Southern Hemisphere mapping (default for e.g. CapeTown, Luanda, etc.)
_SEASON_BY_MONTH_SOUTH = {
    12: "Summer", 1: "Summer", 2: "Summer",
     3: "Autumn", 4: "Autumn", 5: "Autumn",
     6: "Winter", 7: "Winter", 8: "Winter",
     9: "Spring",10: "Spring",11: "Spring",
}

# Northern Hemisphere mapping (used for cities in REVERSE_SEASONS_FOR, e.g., Lagos)
_SEASON_BY_MONTH_NORTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
     3: "Spring", 4: "Spring", 5: "Spring",
     6: "Summer", 7: "Summer", 8: "Summer",
     9: "Autumn",10: "Autumn",11: "Autumn",
}

def _read_seasonal_blocks(profile_path: str) -> dict[str, pd.Series]:
    """
    Read one CSV with columns like:
      <Season>_Timestamps, <Season>_Water_Consumption, <Season>_Ambient_Temperature, <Season>_Power
    for each of: Summer, Autumn, Winter, Spring.

    Returns { "Summer": Series[L/min @ 1min], ... } indexed by 1-minute timestamps.
    Clips negatives to zero. Missing minutes become 0 L/min.
    """
    df = pd.read_csv(profile_path, low_memory=False)
    out: dict[str, pd.Series] = {}

    for season in ("Summer", "Autumn", "Winter", "Spring"):
        ts_col = f"{season}_Timestamps"
        wc_col = f"{season}_Water_Consumption"
        if ts_col not in df.columns or wc_col not in df.columns:
            raise ValueError(f"{profile_path}: missing columns for season '{season}'")

        ts = pd.to_datetime(df[ts_col], errors="coerce")
        vals = pd.to_numeric(df[wc_col], errors="coerce").fillna(0.0).clip(lower=0.0)

        s = pd.Series(vals.to_numpy(float), index=ts)
        s = s.loc[s.index.notna()].sort_index()

        # ensure strict 1-min cadence; sum duplicates; empty minutes → 0 (not ffill)
        s = s.resample("1min").sum().fillna(0.0)
        out[season] = s

    return out

def _seasonal_energy_and_litres(
    season_L_per_min: pd.Series, setpoint_C: float, cold_C: float
) -> tuple[pd.Series, pd.Series]:
    """
    Convert 1-minute L/min to two aligned 5-min series:
    - litres_per_step (L/5min)  = sum over each 5-minute bin
    - energy_kwh (kWh/5min)     = litres * 4184 * dT / 3.6e6
    """
    dT = float(setpoint_C) - float(cold_C)
    L_5 = season_L_per_min.resample("5min", label="right", closed="right").sum()
    kWh = L_5 * 4184.0 * dT / 3_600_000.0
    return kWh, L_5

def _stitch_full_year_from_seasons(
    energy_5min: dict[str, pd.Series],
    litres_5min: dict[str, pd.Series],
    irr_index_local_5min: pd.DatetimeIndex,
    city: str,
    repeats_per_season: int,
) -> tuple[pd.Series, float]:
    """
    Build a full-year 5-min demand series aligned to irr_index_local_5min.
    - Chooses season per calendar month (South default; North if city in REVERSE_SEASONS_FOR)
    - Cycles through each season's provided month block in order, repeating exactly 3×
      to cover the three months of that season.
    Returns:
      (energy_kWh_series_aligned, total_litres_sum_over_year)
    """

    season_by_month = _SEASON_BY_MONTH_NORTH if city in REVERSE_SEASONS_FOR else _SEASON_BY_MONTH_SOUTH

    # Pre-tile each season’s arrays (energy and litres)
    tiled_energy: dict[str, np.ndarray] = {}
    tiled_litres: dict[str, np.ndarray] = {}
    for season in ("Summer", "Autumn", "Winter", "Spring"):
        base_e = energy_5min[season].to_numpy(float)
        base_L = litres_5min[season].to_numpy(float)
        if base_e.size == 0:
            base_e = np.zeros(1, float)
            base_L = np.zeros(1, float)
        tiled_energy[season] = np.tile(base_e, max(1, repeats_per_season))
        tiled_litres[season] = np.tile(base_L, max(1, repeats_per_season))

    # indices grouped by (year, month) on the irradiance (already local-naive, 5-min)
    month_slots: dict[tuple[int, int], np.ndarray] = {}
    for i, ts in enumerate(irr_index_local_5min):
        month_slots.setdefault((ts.year, ts.month), []).append(i)

    # rolling cursors per season (so each month of the season picks up where the previous left off)
    curs_e = {s: 0 for s in tiled_energy}
    curs_L = {s: 0 for s in tiled_litres}

    out_energy = np.zeros(len(irr_index_local_5min), float)
    total_L = 0.0

    for (y, m), idxs in month_slots.items():
        season = season_by_month[m]
        seq_e = tiled_energy[season]
        seq_L = tiled_litres[season]
        cur_e = curs_e[season]
        cur_L = curs_L[season]
        need = len(idxs)

        # energy slice
        if cur_e + need <= seq_e.size:
            take_e = seq_e[cur_e:cur_e+need];  cur_e += need
        else:
            r = (cur_e + need) - seq_e.size
            take_e = np.concatenate([seq_e[cur_e:], seq_e[:r]]);  cur_e = r

        # litres slice (same length)
        if cur_L + need <= seq_L.size:
            take_L = seq_L[cur_L:cur_L+need];  cur_L += need
        else:
            r = (cur_L + need) - seq_L.size
            take_L = np.concatenate([seq_L[cur_L:], seq_L[:r]]);  cur_L = r

        out_energy[np.array(idxs, dtype=int)] = take_e
        total_L += float(take_L.sum())

        curs_e[season] = cur_e
        curs_L[season] = cur_L

    return pd.Series(out_energy, index=irr_index_local_5min), float(total_L)


def simulate_household_efficient(
    profile_path: str,
    irr_df: pd.DataFrame,
    flat_rate_usd_per_kwh: float,
    system_params: Dict,
    city: str,
) -> Dict:
    """
    Memory-efficient simulation of a single household with PV-first dispatch.

    Dispatch:
      - Conventional thermostat with hysteresis unless PERMANENT_LOAD_TEST=True.
      - When heater ON: try to supply element_rating_kw; PV first, grid fills remainder.
      - Tank is stepped EVERY timestep (even when heater OFF), so losses and draws apply.

    Returns:
      dict(profile, kpis, cost_USD, success, [error])
    """
    try:
        household_id = os.path.basename(profile_path)

        # -------------------------
        # 1) Build full-year demand from seasonal CSV (3 months per season)
        # -------------------------
        # Helpers (scoped to this function)
        def _read_seasonal_blocks(csv_path: str) -> dict[str, pd.Series]:
            df = pd.read_csv(csv_path, low_memory=False)
            out = {}
            for season in ("Summer", "Autumn", "Winter", "Spring"):
                ts_col = f"{season}_Timestamps"
                wc_col = f"{season}_Water_Consumption"
                if ts_col not in df.columns or wc_col not in df.columns:
                    raise ValueError(f"{csv_path}: missing columns for {season}")
                ts   = pd.to_datetime(df[ts_col], errors="coerce")
                vals = pd.to_numeric(df[wc_col], errors="coerce").fillna(0.0).clip(lower=0.0)
                s = pd.Series(vals.to_numpy(float), index=ts)
                s = s.loc[s.index.notna()].sort_index()
                # Strict 1-min cadence, sum duplicates, no forward-fill of demand
                s = s.resample("1min").sum().fillna(0.0)
                out[season] = s
            return out

        def _to_5min_energy_and_litres(season_Lpm: pd.Series, setpoint_C: float, cold_C: float) -> tuple[pd.Series, pd.Series]:
            dT = float(setpoint_C) - float(cold_C)
            L_5 = season_Lpm.resample("5min", label="right", closed="right").sum()
            kWh = L_5 * 4184.0 * dT / 3_600_000.0
            return kWh, L_5

        # Season mappings
        SEASON_BY_MONTH_SOUTH = {
            12: "Summer", 1: "Summer", 2: "Summer",
             3: "Autumn", 4: "Autumn", 5: "Autumn",
             6: "Winter", 7: "Winter", 8: "Winter",
             9: "Spring",10: "Spring",11: "Spring",
        }
        SEASON_BY_MONTH_NORTH = {
            12: "Winter", 1: "Winter", 2: "Winter",
             3: "Spring", 4: "Spring", 5: "Spring",
             6: "Summer", 7: "Summer", 8: "Summer",
             9: "Autumn",10: "Autumn",11: "Autumn",
        }

        if irr_df is None or len(irr_df) == 0:
            return _fail(household_id, "Empty irradiance DataFrame")

        # Irradiance index (already local-naive 5-min from main.py)
        idx = irr_df.index

        # Read seasonal litre/min series at 1-min, convert each to 5-min kWh and litres
        setpoint = float(TANK_PARAMS['setpoint'])
        inlet_temp = float(SIM_PARAMS["cold_event_temperature"])
        seasonal_Lpm = _read_seasonal_blocks(profile_path)
        seasonal_kwh: dict[str, pd.Series] = {}
        seasonal_L5:  dict[str, pd.Series] = {}
        for season, ser in seasonal_Lpm.items():
            k5, L5 = _to_5min_energy_and_litres(ser, setpoint, inlet_temp)
            seasonal_kwh[season] = k5
            seasonal_L5[season]  = L5

        # Choose season mapping (reverse for Lagos)
        city_is_north = any(city == c for c in REVERSE_SEASONS_FOR)
        season_by_month = SEASON_BY_MONTH_NORTH if city_is_north else SEASON_BY_MONTH_SOUTH

        # Pre-tile each season's arrays exactly DEMAND_REPEAT_PER_SEASON times
        import numpy as _np
        reps = int(DEMAND_REPEAT_PER_SEASON)
        tiled_energy: dict[str, _np.ndarray] = {}
        tiled_litres: dict[str, _np.ndarray] = {}
        for season in ("Summer", "Autumn", "Winter", "Spring"):
            base_e = seasonal_kwh[season].reindex(seasonal_kwh[season].index, fill_value=0.0).to_numpy(float)
            base_L = seasonal_L5[season].reindex(seasonal_L5[season].index,   fill_value=0.0).to_numpy(float)
            if base_e.size == 0:
                base_e = _np.zeros(1, float); base_L = _np.zeros(1, float)
            tiled_energy[season] = _np.tile(base_e, max(1, reps))
            tiled_litres[season] = _np.tile(base_L, max(1, reps))

        # Group irradiance positions by calendar (year, month)
        month_slots: dict[tuple[int, int], list[int]] = {}
        for i, ts in enumerate(idx):
            month_slots.setdefault((ts.year, ts.month), []).append(i)

        # Roll through each season's tiled array month-by-month
        curs_e = {s: 0 for s in tiled_energy}
        curs_L = {s: 0 for s in tiled_litres}
        demand_energy = _np.zeros(len(idx), float)
        # (we don't return litres, so no need to store per-step litres; totals are not printed)
        for (y, m), pos in month_slots.items():
            season = season_by_month[m]
            seq_e = tiled_energy[season]; cur_e = curs_e[season]
            need = len(pos)
            if cur_e + need <= seq_e.size:
                take_e = seq_e[cur_e:cur_e+need]; cur_e += need
            else:
                r = (cur_e + need) - seq_e.size
                take_e = _np.concatenate([seq_e[cur_e:], seq_e[:r]]); cur_e = r
            demand_energy[_np.array(pos, int)] = take_e
            curs_e[season] = cur_e

        draw_kwh = pd.Series(demand_energy, index=idx, dtype=float)
        demand_kwh_sum = float(draw_kwh.sum())

        # -------------------------
        # 2) PV system init (CEC), ensure met columns exist
        # -------------------------
        try:
            cec = pvlib.pvsystem.retrieve_sam('CECMod')
            if MODULE_NAME not in cec:
                return _fail(household_id, f"CEC module not found: {MODULE_NAME}")
            module_params = cec[MODULE_NAME]
            pv = PVModule(module_params, system_params)
        except Exception as e:
            return _fail(household_id, f"Error initializing PV module: {e}")

        # Ensure required meteo columns exist on the whole irr_df
        for col, default in (('temp_air', 20.0), ('wind_speed', 1.0)):
            if col not in irr_df.columns:
                irr_df[col] = default
        for col in ('dni', 'ghi', 'dhi'):
            if col not in irr_df.columns:
                return _fail(household_id, f"Missing irradiance column: {col}")

        irr_aligned = irr_df  # already aligned to idx

        # PV DC power (kW) for all steps
        pv_kw = pv.get_power(irr_aligned).clip(lower=0.0).astype(float).to_numpy()
        
        # Diagnostics counters (optional prints at the end)
        on_steps_total = 0


        # -------------------------
        # 3) Tank init & constants
        # -------------------------
        tank = StratifiedTank(**TANK_PARAMS)
        tank.initialize(setpoint)
        if not hasattr(tank, 'top_temp'):    tank.top_temp = setpoint
        if not hasattr(tank, 'bottom_temp'): tank.bottom_temp = setpoint

        dt_s = int(TANK_PARAMS['dt_s'])
        dt_h = dt_s / 3600.0
        p_target = float(TANK_PARAMS['element_rating_kw'])
        deadband = float(TANK_PARAMS.get('deadband', 3.0))
        on_thr   = setpoint - deadband
        off_thr  = setpoint
        ambient_override = TANK_PARAMS.get('ambient_override_C', None)

        # Persistent heater state
        heater_on = False

        # -------------------------
        # 4) Accumulators
        # -------------------------
        total_cost_usd = 0.0
        total_solar_savings_usd = 0.0
        grid_kwh_sum = 0.0
        pv_used_kwh_sum = 0.0
        heating_event_count = 0
        solar_event_count   = 0
        solar_used_when_needed_kwh = 0.0
        grid_used_when_needed_kwh  = 0.0
        cold_draws  = 0
        total_points = 0
        temp_sum     = 0.0
        pv_generated_kwh_sum = 0.0

        # -------------------------
        # 5) Single-pass simulation loop
        # -------------------------
        # Pull once for speed
        draw_vals = draw_kwh.to_numpy(float)
        if 'temp_air' in irr_aligned.columns:
            amb_vals = irr_aligned['temp_air'].to_numpy(float)
        else:
            amb_vals = _np.full(len(idx), 20.0, float)

        step_heater_kw  = np.zeros(len(idx), float)   # element kW each step
        step_heater_kwh = np.zeros(len(idx), float)   # element kWh each step (dt_h integrated)

        for i, ts in enumerate(idx):
            p_pv_kw   = float(pv_kw[i])
            demand_kwh = float(draw_vals[i])
            t_amb = float(ambient_override) if ambient_override is not None else float(amb_vals[i])

            # Control
            if PERMANENT_LOAD_TEST:
                heater_on = True
            else:
                if not heater_on:
                    if (getattr(tank, 'bottom_temp', None) is not None and tank.bottom_temp <= on_thr) or (tank.top_temp <= on_thr):
                        heater_on = True
                if heater_on and tank.top_temp >= off_thr:
                    heater_on = False

            # Dispatch
            if heater_on:
                p_pv_used = float(min(p_pv_kw, p_target))
                p_grid    = float(max(0.0, p_target - p_pv_used))
                heating_event_count += 1
                on_steps_total += 1
                if p_pv_used > 0:
                    solar_event_count += 1
            else:
                p_pv_used = 0.0
                p_grid    = 0.0

            # Step tank
            top_T, bot_T = tank.step(p_grid + p_pv_used, demand_kwh, t_amb)
            tank.top_temp = float(top_T)
            tank.bottom_temp = float(bot_T)

            # Energy & money
            step_grid_kwh    = p_grid * dt_h
            step_pv_used_kwh = p_pv_used * dt_h
            step_heater_kw[i]  = p_grid + p_pv_used
            step_heater_kwh[i] = step_grid_kwh + step_pv_used_kwh
            grid_kwh_sum    += step_grid_kwh
            pv_used_kwh_sum += step_pv_used_kwh
            total_cost_usd          += step_grid_kwh * flat_rate_usd_per_kwh
            total_solar_savings_usd += step_pv_used_kwh * flat_rate_usd_per_kwh
            solar_used_when_needed_kwh += step_pv_used_kwh
            grid_used_when_needed_kwh  += step_grid_kwh
            pv_generated_kwh_sum += p_pv_kw * dt_h

            # Comfort
            if demand_kwh > 0 and tank.top_temp < float(TANK_PARAMS['min_usage_temperature']):
                cold_draws += 1
            temp_sum += tank.top_temp
            total_points += 1

        if DIAG_PRINTS:
            # Basic spans & sizes
            sim_steps_total = len(idx)
            dt_h = float(TANK_PARAMS['dt_s']) / 3600.0
            sim_hours_total = sim_steps_total * dt_h
            sim_range_min, sim_range_max = idx[0], idx[-1]

            # PV energy available regardless of tank (kWh)
            # pv_kw is your full-year numpy array of kW
            pv_kwh_possible_total = float((pv_kw * dt_h).sum())

            # Demand sanity: estimate total litres from energy using ΔT
            setpoint = float(TANK_PARAMS['setpoint'])
            inlet_temp = float(SIM_PARAMS['cold_event_temperature'])
            dT = max(1e-6, setpoint - inlet_temp)
            total_litres_est = float(demand_kwh_sum * 3_600_000.0 / (4184.0 * dT))
            sim_days = max(1, (sim_range_max.normalize() - sim_range_min.normalize()).days + 1)

            print(f"[{household_id}] SIM RANGE: {sim_range_min} → {sim_range_max}")
            print(f"[{household_id}] SIM STEPS={sim_steps_total}, HOURS={sim_hours_total:.1f}")
            print(f"[{household_id}] Heater ON steps: {on_steps_total}/{sim_steps_total} → hours_on={on_steps_total * dt_h:.1f}")
            print(f"[{household_id}] PV energy available (kWh): {pv_kwh_possible_total:.1f}")
            print(f"[{household_id}] PV used (kWh): {pv_used_kwh_sum:.1f} | Grid (kWh): {grid_kwh_sum:.1f} | Total heating (kWh): {pv_used_kwh_sum + grid_kwh_sum:.1f}")
            print(f"[{household_id}] Demand (kWh): {demand_kwh_sum:.1f} | Total litres (est): {total_litres_est:,.0f} L (≈ {total_litres_est/sim_days:.1f} L/day)")

            if PERMANENT_LOAD_TEST:
                expected_kwh = float(p_target * sim_hours_total)
                actual_kwh   = float(pv_used_kwh_sum + grid_kwh_sum)
                diff_pct = (abs(actual_kwh - expected_kwh) / expected_kwh * 100.0) if expected_kwh > 0 else 0.0
                print(f"[{household_id}] PERM-LOAD: expected {expected_kwh:.1f} kWh, actual {actual_kwh:.1f} kWh (Δ={diff_pct:.2f}%)")

        # -------------------------
        # 6) KPIs
        # -------------------------
        total_heating_kwh = pv_used_kwh_sum + grid_kwh_sum
        solar_heating_energy_fraction = (pv_used_kwh_sum / total_heating_kwh) if total_heating_kwh > 0 else 0.0
        solar_heating_event_fraction  = (solar_event_count / heating_event_count) if heating_event_count > 0 else 0.0

        cost_without_solar_usd = total_cost_usd + total_solar_savings_usd
        savings_pct = (total_solar_savings_usd / cost_without_solar_usd * 100.0) if cost_without_solar_usd > 0 else 0.0

        kpis = {
            'annual_grid_kwh': grid_kwh_sum,
            'annual_solar_kwh': pv_used_kwh_sum,
            'annual_demand_kwh': demand_kwh_sum,
            'solar_fraction': (pv_used_kwh_sum / total_heating_kwh) if total_heating_kwh > 0 else 0.0,
            'solar_heating_energy_fraction': solar_heating_energy_fraction,
            'solar_heating_event_fraction':  solar_heating_event_fraction,
            'cold_draw_pct': (cold_draws / total_points * 100.0) if total_points > 0 else 0.0,
            'avg_temp': (temp_sum / total_points) if total_points > 0 else np.nan,
            'solar_used_when_needed_kwh': solar_used_when_needed_kwh,
            'grid_used_when_needed_kwh':  grid_used_when_needed_kwh,
            'annual_solar_savings_USD': total_solar_savings_usd,
            'cost_without_solar_USD':   cost_without_solar_usd,
            'savings_percentage':       savings_pct,
            'pv_generated_kwh': pv_generated_kwh_sum,
        }

        return {
            'profile': household_id,
            'kpis': kpis,
            'cost_USD': total_cost_usd,
            'timeseries': {
            'heater_kw':      pd.Series(step_heater_kw,  index=idx),
            'heater_kwh_5min':pd.Series(step_heater_kwh, index=idx),
            'draw_kwh_5min':  draw_kwh, 
            'water_L_5min':   draw_kwh * 3_600_000.0 / (4184.0 * max(1e-6, setpoint - inlet_temp)),
        },
            'success': True
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return _fail(os.path.basename(profile_path), str(e))

                    
def _fail(household: str, msg: str):
    return {
        'profile': household,
        'kpis': {},
        'cost_USD': 0.0,
        'success': False,
        'error': msg,
    }
