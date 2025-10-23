# tariff.py

TARIFFS_USD_PER_KWH = {
    # exact city keys as your SELECTED_LOCATIONS uses
    "CapeTown": 0.186,
    "Johannesburg": 0.186,
    "Lusaka": 0.023,
    "Luanda": 0.015,
    "Kinshasa": 0.064,
    "Nairobi": 0.22,
    "Lagos": 0.036,
}

def get_city_usd_rate(city: str) -> float:
    """
    Return the flat USD/kWh tariff for the exact city key.
    Raises a clear error if the city is unknown.
    """
    try:
        return float(TARIFFS_USD_PER_KWH[city])
    except KeyError:
        raise KeyError(f"Tariff data does not exist for city: {city}")
