from typing import Any, Mapping

_ALIAS_GROUPS: list[tuple[str, frozenset[str]]] = [
    ("system_distance", frozenset({"distance_to_system", "D_pl", "d_pl", "distance"})),
    ("planet_mass", frozenset({"mass", "m_pl", "M_pl"})),
    ("planet_radius", frozenset({"radius", "R_pl", "r_pl"})),
    ("f_sed", frozenset({"cloud_f_sed", "fsed"})),
    ("log_g", frozenset({"log_surface_gravity"})),
    ("M/H", frozenset({"Fe/H", "[Fe/H]", "metallicity"})),
    ("C/O", frozenset({"c_o_ratio", "C_O_ratio", "co_ratio"})),
    ("log_pcloud", frozenset({"log_Pcloud"})),
    ("log_pquench", frozenset({"log_Pquench", "log_quench_pressure", "log_carbon_quench_pressure"})),
    ("sigma_lnorm", frozenset({"log_normal_distribution_std"})),
]

# Flat lookup built once at import time: alias -> canonical name
PARAMETER_ALIAS_MAP: dict[str, str] = {
    alias: canonical
    for canonical, aliases in _ALIAS_GROUPS
    for alias in aliases
}


def normalize_parameter_name(name: str) -> str:
    """Return the canonical parameter name for a given name or alias."""
    return PARAMETER_ALIAS_MAP.get(name, name)


def normalize_parameter_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Rename alias keys to their canonical names. Last value wins on collision."""
    return {normalize_parameter_name(k): v for k, v in mapping.items()}
