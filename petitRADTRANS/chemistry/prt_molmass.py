"""Interface between molmass and petitRADTRANS."""

from molmass import Formula

from petitRADTRANS._input_data_loader import get_species_isotopologue_name, split_species_all_info
from petitRADTRANS.physical_constants import e_molar_mass


def get_molmass_name(species):
def get_molmass_name(species: str):
    """Convert a pRT species' name into a molmass-compatible name."""
    species = get_species_isotopologue_name(species, join=True)

    name, natural_abundance, charge, _, _, _ = split_species_all_info(species)

    # Rearrange isotopes in molmass format
    isotopes = name.split('-')

    if natural_abundance != '':
        return [''.join(isotopes) + charge]

    isotopes.append(charge)

    return isotopes


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
        return e_molar_mass

    if "-NatAbund" in species:
        natural_abundance = True
    else:
        natural_abundance = False

    names = get_molmass_name(species)

    molar_mass = None

    if natural_abundance and len(names) == 1:
        return Formula(names[0]).mass
    elif natural_abundance and len(names) != 1:
        raise ValueError(f"species '{species}' has the natural abundance flag, "
                         f"but several isotopes were detected: ({names})")
    elif not natural_abundance:
        molar_mass = 0.0
        charge = None

        # molmass cannot manage formulae containing both '[]'-separated isotopes and charges,
        if names[-1] == '':
            charge = 0.0
        elif names[-1][-1] in ['+', '-']:
            charge = names[-1].replace('_', '')  # get rid of the leading '_'
            charge = charge[-1] + charge[:-1]  # put sign in front of digits

            if len(charge) == 1:
                charge = charge + '1'  # when no charge number is given, assume that it is 1

            charge = float(charge)

        for name in names[:-1]:
            molar_mass += Formula(name).isotope.massnumber

        # Change charge sign ('+' means *less* e-, thanks B. Franklin! https://xkcd.com/567/)
        molar_mass += -charge * e_molar_mass

    return molar_mass
