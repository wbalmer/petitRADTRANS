"""Interface between molmass and petitRADTRANS."""

from molmass import Formula

import petitRADTRANS.nat_cst as nc


def getMM(species):
    """
    Get the molecular mass of a given species.

    This function uses the molmass package to
    calculate the mass number for the standard
    isotope of an input species. If all_iso
    is part of the input, it will return the
    mean molar mass.

    Args:
        species : string
            The chemical formula of the compound. ie C2H2 or H2O
    Returns:
        The molar mass of the compound in atomic mass units.
    """
    if species == 'e-':
        return nc.e_molar_mass
    elif species == 'H-':
        return Formula('H').mass + nc.e_molar_mass

    name = species.split("_")[0]
    name = name.split(',')[0]
    f = Formula(name)

    if "all_iso" in species:
        return f.mass

    return f.isotope.massnumber
