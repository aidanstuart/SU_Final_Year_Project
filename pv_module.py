import warnings
import numpy as np
import pandas as pd
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

class PVModule:
    """
    Rooftop PV wrapper using pvlib. Returns DC p_mp power in kW.
    """
    def __init__(self, module_params: dict, system_params: dict):
        self.module = module_params

        # --- Unpack required system params (no defaults) ---
        tilt = system_params['tilt']
        azimuth = system_params['azimuth']
        lat = system_params['latitude']
        lon = system_params['longitude']
        tz = system_params['timezone']
        racking_model = system_params.get('racking_model', 'open_rack_glass_glass')

        n_series = int(system_params['num_panels'])
        if n_series < 1:
            raise ValueError("num_panels must be >= 1")
        strings = 1  # all panels in series, one string

        # --- Temperature model params (SAPM) ---
        if racking_model not in TEMPERATURE_MODEL_PARAMETERS['sapm']:
            print(f"Warning: Invalid racking model '{racking_model}', using 'open_rack_glass_glass'.")
            racking_model = 'open_rack_glass_glass'
        temp_params = TEMPERATURE_MODEL_PARAMETERS['sapm'][racking_model]

        # --- Compute pdc0 for PVWatts AC (neutral inverter ~ no loss) ---
        def _first_present(d, keys):
            for k in keys:
                if k in d and pd.notnull(d[k]):
                    return float(d[k])
            return None

        vmp = _first_present(self.module, ('V_mp_ref', 'Vmp', 'V_mp'))
        imp = _first_present(self.module, ('I_mp_ref', 'Imp', 'I_mp'))
        if vmp is None or imp is None:
            raise ValueError("CEC module params missing Vmp/Imp (need V_mp_ref/I_mp_ref or Vmp/Imp).")

        pmp_module_w = vmp * imp                      # per-module STC Pmp [W]
        pdc0 = pmp_module_w * n_series * strings      # array STC DC power [W]
        inverter_params = {
            'pdc0': float(pdc0),
            'eta_inv_nom': 1.0,
            'eta_inv_ref': 1.0,
        }
        print(f"Initializing PVSystem with tilt={tilt}°, azimuth={azimuth}°, location=({lat}, {lon}), pdc0={pdc0:.1f} W")

        # --- Build PVSystem ---
        pv_sys = PVSystem(
            module_parameters=self.module,
            inverter_parameters=inverter_params,
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            modules_per_string=n_series,
            strings_per_inverter=strings,
            temperature_model_parameters=temp_params,
        )

        # --- Location & ModelChain ---
        self.location = Location(latitude=lat, longitude=lon, tz=tz)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.mc = ModelChain(
                pv_sys,
                self.location,
                aoi_model='no_loss',
                spectral_model='no_loss',
                temperature_model='sapm',
                dc_model='cec',
                ac_model='pvwatts',   # pvlib accepts this (your version rejects 'dc')
            )
        print("✅ ModelChain initialized successfully")

    def get_power(self, meteo: pd.DataFrame) -> pd.Series:
        """
        Compute DC power (p_mp) from meteorological inputs. Returns kW aligned to meteo.index.
        """
        import numpy as np
        import pandas as pd

        try:
            required = ['dni', 'ghi', 'dhi', 'temp_air', 'wind_speed']
            missing = [c for c in required if c not in meteo.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            orig_idx = meteo.index  # keep original to align outputs

            weather = meteo[required].copy()
            for c in required:
                if weather[c].dtype == 'object':
                    weather[c] = pd.to_numeric(weather[c], errors='coerce')
                weather[c] = weather[c].replace([np.inf, -np.inf], 0).fillna(0)

            # ✅ Ensure pvlib sees tz-aware times in the site’s timezone
            site_tz = self.location.tz
            idx = weather.index
            if idx.tz is None:
                weather.index = idx.tz_localize(site_tz)
            else:
                weather.index = idx.tz_convert(site_tz)

            # Run pvlib
            self.mc.run_model(weather)

            dc = getattr(self.mc.results, 'dc', None)
            if dc is None:
                return pd.Series(0.0, index=orig_idx)

            if isinstance(dc, pd.DataFrame) and 'p_mp' in dc:
                pmp = dc['p_mp']
            else:
                pmp = getattr(dc, 'p_mp', None)
                if pmp is None:
                    return pd.Series(0.0, index=orig_idx)

            # 🔁 Align tz with original index
            if pmp.index.tz is not None and orig_idx.tz is None:
                pmp.index = pmp.index.tz_localize(None)
            elif pmp.index.tz is None and orig_idx.tz is not None:
                pmp.index = pmp.index.tz_localize(orig_idx.tz)

            out = (pmp / 1000.0).reindex(orig_idx)  # W → kW

            return out.clip(lower=0.0).fillna(0.0)

        except Exception as e:
            print(f"❌ Error computing PV power: {e}")
            import traceback; traceback.print_exc()
            return pd.Series(0.0, index=meteo.index)
