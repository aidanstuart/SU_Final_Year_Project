# config.py
import os

# -----------------------------
# Paths (Windows-friendly)
# -----------------------------
ROOT_DIR = r"C:\Users\Aidan Stuart\OneDrive\Documents\2025\Skripsie\Code\ewh_pv_simulation"
SOLAR_DATA_DIR = os.path.join(ROOT_DIR, "solar_data")
USER_DATA_DIR  = os.path.join(ROOT_DIR, "user_data", "Classified_Profiles")

# -----------------------------
# Simulation knobs
# -----------------------------
SELECTED_PROFILES   = ['Light', 'Medium', 'Heavy']           # choose any of: 'Light', 'Medium', 'Heavy'
SELECTED_LOCATIONS  = ["CapeTown","Johannesburg"]          # choose any subset of LOCATIONS
NUM_PANELS          = 4                     # panels per string (single inverter)
TILT                = 30                    # deg
AZIMUTH             = 30                   # deg (0 = North in PVLib)

# -----------------------------
# Locations & site metadata
# -----------------------------
LOCATIONS = ["CapeTown","Johannesburg","Lusaka","Luanda","Kinshasa","Nairobi","Lagos"]

LOCATION_PARAMS = {
    "CapeTown":     {"latitude": -33.9249, "longitude":  18.4241, "timezone": "Africa/Johannesburg"},
    "Johannesburg": {"latitude": -26.2041, "longitude":  28.0473, "timezone": "Africa/Johannesburg"},
    "Lusaka":       {"latitude": -15.3875, "longitude":  28.3228, "timezone": "Africa/Lusaka"},
    "Luanda":       {"latitude":  -8.8390, "longitude":  13.2894, "timezone": "Africa/Luanda"},
    "Kinshasa":     {"latitude":  -4.4419, "longitude":  15.2663, "timezone": "Africa/Kinshasa"},
    "Nairobi":      {"latitude":  -1.2921, "longitude":  36.8219, "timezone": "Africa/Nairobi"},
    "Lagos":        {"latitude":   6.5244, "longitude":   3.3792, "timezone": "Africa/Lagos"},
}

def get_irradiance_path(city: str) -> str:
    """
    Build absolute path to the city's irradiance CSV:
    <ROOT>/solar_data/<City>/<City>.csv
    """
    return os.path.join(SOLAR_DATA_DIR, city, f"{city}.csv")

def get_system_params(city: str) -> dict:
    """Return a per-location PV system parameter dictionary."""
    if city not in LOCATION_PARAMS:
        raise ValueError(f"Unknown city '{city}'. Valid: {LOCATIONS}")
    loc = LOCATION_PARAMS[city]
    return {
        "tilt": TILT,
        "azimuth": AZIMUTH,
        "inverter": {"pdc0": 3000, "eta_inv_nom": 0.96},
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "timezone": loc["timezone"],
        "racking_model": "open_rack_glass_polymer",
        "num_panels": NUM_PANELS,
    }

# -----------------------------
# Tank & demand parameters
# -----------------------------
TANK_PARAMS = {
    "setpoint": 60.0,             # °C
    "deadband": 3.0,              # °C
    "volume_l": 150,              # L
    "c": 4184,                    # J/(kg·K)
    "rho": 1000,                  # kg/m³
    "R_th": 2.5,                  # K/W
    "element_rating_kw": 3.0,     # kW
    "dt_s": 300,                  # 5-min step
    "min_usage_temperature": 50,  # °C
}

SIM_PARAMS = {
    "cold_event_temperature": 15.0,  # °C mains water temp
    "min_draw_l_per_event": 2.0,     # L
}

# Diagnostics
PERMANENT_LOAD_TEST = False   # set True to force element ON every step
DIAG_PRINTS = False            # set False to silence the prints

#---------RUNSWEEP-------
RUN_SWEEP = False  # set to False/True to skip/run

# Demand synthesis from seasonal month blocks
DEMAND_REPEAT_PER_SEASON = 3      # always 3 months per season
REVERSE_SEASONS_FOR = {"Lagos"}   # cities that use Northern Hemisphere seasons

# -----------------------------
# PV module choice
# -----------------------------
MODULE_NAME = "United_Renewable_Energy_Co_Ltd_D7K420H8A"



