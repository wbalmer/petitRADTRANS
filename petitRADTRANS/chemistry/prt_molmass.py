"""Interface between molmass and petitRADTRANS."""

import re

from molmass import Formula

import petitRADTRANS.physical_constants as cst


def get_species_molar_mass(species):
    """
    Get the molecular mass of a given species.

    This function uses the molmass package to calculate the mass number for the standard isotope of an input species.
    If all_iso is part of the input, it will return the mean molar mass.

    Args:
        species : string
            The chemical formula of the compound. ie C2H2 or H2O
    Returns:
        The molar mass of the compound in atomic mass units.
    """
    if species == 'e-':
        return cst.e_molar_mass

    if "(s)" in species or "(l)" in species:
        return 0  # ignore cloud species

    if "__" in species:
        species = species.split("__", 1)[0]  # remove resolving power or opacity source information

    if len(re.findall(r'^[A-Z][a-z]?(\d{1,3})?[+|-]$', species)) == 1:  # positive or negative ion
        # Get the number of electrons lost or gained
        ionisation = re.findall(r'^.{1,2}(\d{1,3})[+|-]$', species)

        if len(ionisation) == 0:
            ionisation_str = ''
            ionisation = 1
        else:
            ionisation_str = ionisation[0]
            ionisation = int(ionisation_str)

        # Get the corresponding molar mass
        if '-' in species:
            return (Formula(species.rsplit(f'{ionisation_str}-', 1)[0]).isotope.massnumber
                    + ionisation * cst.e_molar_mass)
        elif '+' in species:
            return (Formula(species.rsplit(f'{ionisation_str}+', 1)[0]).isotope.massnumber
                    - ionisation * cst.e_molar_mass)
    elif species[-1] in ['-', '+']:
        raise ValueError(f"invalid species formula '{species}', either a symbol used is unknown, or the ion formula "
                         f"does not respects the pattern '<element_symbol><ionisation_number><+|->' "
                         f"(e.g., 'Ca2+', 'H-')")
    elif '-' in species:
        isotopes = species.split('-')

        for i, isotope in enumerate(isotopes):
            isotopes[i] = f"[{isotope}]"

        species = "".join(isotopes)

    name = species.split("_")[0]
    name = name.split(',')[0]
    f = Formula(name)

    if "all_iso" in species:
        return f.mass

    return f.isotope.massnumber
