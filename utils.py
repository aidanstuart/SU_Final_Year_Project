# utils.py

CHUNK_SIZE = 1000

def get_hourly_rate(ts, flat_rate_usd: float) -> float:
    """
    Flat-rate version: ignore time-of-day, just return the city USD/kWh.
    Keeping the same signature so call sites don't need to change.
    """
    return float(flat_rate_usd)
