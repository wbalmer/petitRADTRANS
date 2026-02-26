from .gradient import dtdp_temperature_profile
from .guillot import (
    temperature_profile_function_guillot,
    temperature_profile_function_guillot_dayside,
    temperature_profile_function_guillot_global,
    temperature_profile_function_guillot_global_ret,
    temperature_profile_function_guillot_metallic,
    temperature_profile_function_guillot_modif
    )
from .isothermal import temperature_profile_function_isothermal
from .madhusudhan import madhusudhan_seager_temperature_profile
from .molliere import spline_eddington_adiabatic_temperature_profile
from .power_law import power_law_temperature_profile
from .spline import linear_spline_temperature_profile, cubic_spline_temperature_profile