"""Shared opacity registry helpers.

This module centralizes legacy base-name registries and default conversion settings
that are still consumed by file conversion, documentation tooling, and deprecated
compatibility shims.
"""

import numpy as np

from petitRADTRANS.utils import LockedDict


def _get_base_cia_names():
    return LockedDict.build_and_lock(
        {
            "H2--H2": "H2--H2-NatAbund__BoRi.R831_0.6-250mu",
            "H2--He": "H2--He-NatAbund__BoRi.DeltaWavenumber2_0.5-500mu",
            "H2O--H2O": "H2O--H2O-NatAbund",
            "H2O--N2": "H2O--N2-NatAbund",
            "N2--H2": "N2--H2-NatAbund.DeltaWavenumber1_5.3-909mu",
            "N2--He": "N2--He-NatAbund.DeltaWavenumber1_10-909mu",
            "N2--N2": "N2--N2-NatAbund.DeltaWavelength1e-6_2-100mu",
            "O2--O2": "O2--O2-NatAbund.DeltaWavelength1e-6_0.34-8.7mu",
            "N2--O2": "N2--O2-NatAbund.DeltaWavelength1e-6_0.72-5.4mu",
            "CO2--CO2": "CO2--CO2-NatAbund.DeltaWavelength1e-6_3-100mu",
        }
    )


def _get_base_cloud_names():
    return LockedDict.build_and_lock(
        {
            "Al2O3(s)_crystalline": "Al2-O3-NatAbund(s)_crystalline_000.R39_0.1-250mu",
            "Fe(s)_amorphous": "Fe-NatAbund(s)_amorphous.R39_0.1-250mu",
            "Fe(s)_crystalline": "Fe-NatAbund(s)_crystalline_000.R39_0.1-250mu",
            "H2O(s)_crystalline": "H2-O-NatAbund(s)_crystalline_000.R39_0.1-250mu",
            "H2O(l)": "H2-O-NatAbund(l).R39_0.1-250mu",
            "H2OSO4(l)": "H2-O-S-O4-NatAbund(l).R39_0.1-250mu",
            "KCl(s)_crystalline": "K-Cl-NatAbund(s)_crystalline_000.R39_0.1-250mu",
            "Mg2SiO4(s)_amorphous": "Mg2-Si-O4-NatAbund(s)_amorphous.R39_0.1-250mu",
            "Mg2SiO4(s)_crystalline": "Mg2-Si-O4-NatAbund(s)_crystalline_000.R39_0.1-250mu",
            "Mg05Fe05SiO3(s)_amorphous": "Mg05-Fe05-Si-O3-NatAbund(s)_amorphous.R39_0.1-250mu",
            "MgAl2O4(s)_crystalline": "Mg-Al2-O4-NatAbund(s)_amorphous.R39_0.1-250mu",
            "MgFeSiO4(s)_amorphous": "Mg-Fe-Si-O4-NatAbund(s)_amorphous.R39_0.2-250mu",
            "MgSiO3(s)_amorphous": "Mg-Si-O3-NatAbund(s)_amorphous.R39_0.1-250mu",
            "MgSiO3(s)_crystalline": "Mg-Si-O3-NatAbund(s)_crystalline_000.R39_0.1-250mu",
            "Na2S(s)_crystalline": "Na2-S-NatAbund(s)_crystalline_000.R39_0.1-250mu",
            "SiC(s)_crystalline": "Si-C-NatAbund(s)_crystalline_000.R39_0.1-250mu",
        }
    )


def _get_base_correlated_k_names():
    """Only used for file conversion from pRT2 to pRT3."""
    return LockedDict.build_and_lock(
        {
            "Al": "27Al__Kurucz.R1000_0.1-250mu",
            "Al+": "27Al_p__Kurucz.R1000_0.1-250mu",
            "AlH": "27Al-1H__AlHambra.R1000_0.3-50mu",
            "AlO": "27Al-16O__ATP.R1000_0.3-50mu",
            "C2H2": "12C2-1H2__aCeTY.R1000_0.3-50mu",
            "C2H4": "12C2-1H4__MaYTY.R1000_0.3-50mu",
            "Ca": "40Ca__Kurucz.R1000_0.1-250mu",
            "Ca+": "40Ca_p__Kurucz.R1000_0.1-250mu",
            "CaH": "40Ca-1H__MoLLIST.R1000_0.3-50mu",
            "CH4": "12C-1H4__YT34to10.R1000_0.3-50mu",
            "CO": "12C-16O__HITEMP.R1000_0.1-250mu",
            "13CO": "13C-16O__HITEMP.R1000_0.1-250mu",
            "CO_all_iso": "C-O-NatAbund__HITEMP.R1000_0.1-250mu",
            "CO2": "12C-16O2__UCL-4000.R1000_0.3-50mu",
            "CrH": "52Cr-1H__MoLLIST.R1000_0.3-50mu",
            "Fe": "56Fe__Kurucz.R1000_0.1-250mu",
            "Fe+": "56Fe_p__Kurucz.R1000_0.1-250mu",
            "FeH": "56Fe-1H__MoLLIST.R1000_0.3-50mu",
            "H2O": "1H2-16O__HITEMP.R1000_0.1-250mu",
            "H2S": "1H2-32S__AYT2.R1000_0.3-50mu",
            "HCN": "1H-12C-14N__Harris.R1000_0.3-50mu",
            "K": "39K__Allard.R1000_0.1-250mu",
            "Li": "7Li__Kurucz.R1000_0.1-250mu",
            "Mg": "24Mg__Kurucz.R1000_0.1-250mu",
            "Mg+": "24Mg_p__Kurucz.R1000_0.1-250mu",
            "MgH": "24Mg-1H__MoLLIST.R1000_0.3-50mu",
            "MgO": "24Mg-16O__LiTY.R1000_0.3-50mu",
            "Na": "23Na__Allard.R1000_0.1-250mu",
            "NaH": "23Na-1H__Rivlin.R1000_0.3-50mu",
            "NH3": "14N-1H3__CoYuTe.R1000_0.3-50mu",
            "O": "16O__Kurucz.R1000_0.1-250mu",
            "O+": "16O_p__Kurucz.R1000_0.1-250mu",
            "O2": "16O2__HITRAN.R1000_0.3-50mu",
            "O3": "16O3__HITRAN.R1000_0.1-250mu",
            "OH": "16O-1H__HITEMP.R1000_0.3-50mu",
            "PH3": "31P-1H3__SAlTY.R1000_0.3-50mu",
            "SH": "32S-1H__GYT.R1000_0.3-50mu",
            "Si": "28Si__Kurucz.R1000_0.1-250mu",
            "Si+": "28Si_p__Kurucz.R1000_0.1-250mu",
            "SiO": "28Si-16O__EBJT.R1000_0.3-50mu",
            "SiO2": "28Si-16O2__OYT3.R1000_0.3-50mu",
            "Ti": "48Ti__Kurucz.R1000_0.1-250mu",
            "Ti+": "48Ti_p__Kurucz.R1000_0.1-250mu",
            "48TiO": "48Ti-16O__Plez.R1000_0.1-250mu",
            "TiO_all_iso": "Ti-O-NatAbund__Plez.R1000_0.1-250mu",
            "V": "51V__Kurucz.R1000_0.1-250mu",
            "V+": "51V_p__Kurucz.R1000_0.1-250mu",
            "VO": "51V-16O__VOMYT.R1000_0.3-50mu",
        }
    )


def _get_base_line_by_line_names():
    """Only used for file conversion from pRT2 to pRT3."""
    return LockedDict.build_and_lock(
        {
            "Al": "27Al__Kurucz.R1e6_0.3-28mu",
            "B": "11B__Kurucz.R1e6_0.3-28mu",
            "Be": "9Be__Kurucz.R1e6_0.3-28mu",
            "C2H2": "12C2-1H2__HITRAN.R1e6_0.3-28mu",
            "Ca": "40Ca__Kurucz.R1e6_0.3-28mu",
            "Ca+": "40Ca_p__Kurucz.R1e6_0.3-28mu",
            "CaH": "40Ca-1H__MoLLIST.R1e6_0.3-28mu",
            "CH3D": "12C-1H3-2H__HITRAN.R1e6_0.3-28mu",
            "CH4": "12C-1H4__Hargreaves.R1e6_0.3-28mu",
            "CO2": "12C-16O2__HITEMP.R1e6_0.3-28mu",
            "C-17O": "12C-17O__HITRAN.R1e6_0.3-28mu",
            "C-18O": "12C-18O__HITRAN.R1e6_0.3-28mu",
            "13CO": "13C-16O__HITRAN.R1e6_0.3-28mu",
            "13C-17O": "13C-17O__HITRAN.R1e6_0.3-28mu",
            "13C-18O": "13C-18O__HITRAN.R1e6_0.3-28mu",
            "CO_all_iso": "C-O-NatAbund__HITEMP.R1e6_0.3-28mu",
            "CO": "12C-16O__HITEMP.R1e6_0.3-28mu",
            "Cr": "52Cr__Kurucz.R1e6_0.3-28mu",
            "Fe": "56Fe__Kurucz.R1e6_0.3-28mu",
            "Fe+": "56Fe_p__Kurucz.R1e6_0.3-28mu",
            "FeH": "56Fe-1H__MoLLIST.R1e6_0.3-28mu",
            "HD": "1H-2H__HITRAN.R1e6_0.3-28mu",
            "H2": "1H2__HITRAN.R1e6_0.3-28mu",
            "HDO": "1H-2H-16O__HITEMP.R1e6_0.3-28mu",
            "H2-17O": "1H2-17O__HITEMP.R1e6_0.3-28mu",
            "HD-17O": "1H-2H-17O__HITEMP.R1e6_0.3-28mu",
            "H2-18O": "1H2-18O__HITEMP.R1e6_0.3-28mu",
            "HD-18O": "1H-2H-18O__HITEMP.R1e6_0.3-28mu",
            "H2O": "1H2-16O__HITEMP.R1e6_0.3-28mu",
            "H2S_main_iso": "1H2-32S__HITRAN.R1e6_0.3-28mu",
            "HCN_main_iso": "1H-12C-14N__Harris.R1e6_0.3-28mu",
            "K": "39K__Allard.R1e6_0.3-28mu",
            "Li": "7Li__Kurucz.R1e6_0.3-28mu",
            "Mg": "24Mg__Kurucz.R1e6_0.3-28mu",
            "Mg+": "24Mg_p__Kurucz.R1e6_0.3-28mu",
            "N": "14N__Kurucz.R1e6_0.3-28mu",
            "Na": "23Na__Allard.R1e6_0.3-28mu",
            "NH3": "14N-1H3__BYTe.R1e6_0.3-28mu",
            "O3": "16O3__HITRAN.R1e6_0.3-28mu",
            "OH": "16O-1H__HITEMP.R1e6_0.3-28mu",
            "PH3": "31P-1H3__SAlTY.R1e6_0.3-28mu",
            "Si": "28Si__Kurucz.R1e6_0.3-28mu",
            "SiO": "28Si-16O__EBJT.R1e6_0.3-28mu",
            "Ti": "48Ti__Kurucz.R1e6_0.3-28mu",
            "46TiO": "46Ti-16O__Plez.R1e6_0.3-28mu",
            "47TiO": "47Ti-16O__Plez.R1e6_0.3-28mu",
            "TiO": "48Ti-16O__Plez.R1e6_0.3-28mu",
            "49TiO": "49Ti-16O__Plez.R1e6_0.3-28mu",
            "50TiO": "50Ti-16O__Plez.R1e6_0.3-28mu",
            "TiO_all_iso": "Ti-O-NatAbund__Plez.R1e6_0.3-28mu",
            "V": "51V__Kurucz.R1e6_0.3-28mu",
            "V+": "51V_p__Kurucz.R1e6_0.3-28mu",
            "VO": "51V-16O__Plez.R1e6_0.3-28mu",
            "Y": "89Y__Kurucz.R1e6_0.3-28mu",
        }
    )


def _get_default_rebinning_wavelength_range():
    return np.array([0.1, 251.0])
