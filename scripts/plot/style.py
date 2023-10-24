"""

"""
import matplotlib.pyplot as plt


TINY_FIGURE_FONT_SIZE = 40  # 0.5 text width 16/9
SMALL_FIGURE_FONT_SIZE = 22  # 0.25 text width
MEDIUM_FIGURE_FONT_SIZE = 16  # 0.5 text width
LARGE_FIGURE_FONT_SIZE = 22  # 1.0 text width

large_figsize = [19.20, 10.80]  # 1920 x 1080 for 100 dpi (default)

species_color = {
    'CH4': 'C7',
    'CO': 'C3',
    'CO2': 'C5',
    'FeH': 'C4',
    'H2O': 'C0',
    'H2S': 'olive',
    'HCN': 'darkblue',
    'K': 'C8',
    'Na': 'gold',
    'NH3': 'C9',
    'PH3': 'C1',
    'TiO': 'C2',
    'VO': 'darkgreen',
}

other_gases_color = {
    'Al': 'C7',
    'Ar': 'violet',
    'AsH3': 'm',
    'Ca': 'peru',
    'Co': 'aliceblue',
    'Cr': 'skyblue',
    'Cu': 'tan',
    'Fe': 'C4',
    'GeH4': 'olivedrab',
    'H': 'dimgray',
    'H2': 'k',
    'HCl': 'palegreen',
    'HF': 'y',
    'He': 'r',
    'KCl': 'darkolivegreen',
    'Kr': 'lightgrey',
    'Li': 'c',
    'Mg': 'darkorange',
    'Mn': 'olive',
    'N2': 'b',
    'NaCl': 'yellowgreen',
    'Ne': 'brown',
    'Ni': 'lightcoral',
    'P': 'wheat',
    'P2': 'navajowhite',
    'PH2': 'papayawhip',
    'PO': 'sandybrown',
    'SiH4': 'plum',
    'SiO': 'darkred',
    'Ti': 'lime',
    'TiO2': 'mediumseagreen',
    'V': 'forestgreen',
    'VO2': 'seagreen',
    'Xe': 'dodgerblue',
    'Zn': 'salmon'
}

cloud_color = {
    # condensation profiles
    'NH3': 'C9',
    'NH4SH': 'C1',
    'H2O': 'C0',
    'NH4Cl': 'C6',
    'H3PO4': 'wheat',
    'ZnS': 'C3',
    'KCl': 'C8',
    'Na2S': 'gold',
    'MnS': 'olive',
    'Cr': 'skyblue',
    'Cr2O3': 'deepskyblue',
    'MgSiO3': 'darkorange',
    'Mg2SiO4': 'C5',
    'SiO2': 'darkred',
    'TiN': 'lime',
    'VO': 'forestgreen',
    'Fe': 'C4',
    'CaTiO3': 'peru',
    'Al2O3': 'C7',
}

wavenumber_units = r'cm$^{-1}$'
wavelength_units = r'cm'
spectral_radiosity_units = r'erg$\cdot$s$^{-1}\cdot$cm${-2}$/cm$^{-1}$'


def update_figure_font_size(font_size):
    """
    Update the figure font size in a nice way.
    :param font_size: new font size
    """
    plt.rc('font', size=font_size)  # controls default text sizes
    plt.rc('axes', titlesize=font_size)  # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)  # fontsize of the x and y labels
    plt.rc('axes.formatter', use_mathtext=True)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('xtick', direction='in')  # fontsize of the tick labels
    plt.rc('xtick.major', width=font_size / 10 * 0.8, size=font_size / 10 * 3.5)  # fontsize of the tick labels
    plt.rc('xtick.minor', width=font_size / 10 * 0.6, size=font_size / 10 * 2)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('ytick', direction='in')  # fontsize of the tick labels
    plt.rc('ytick.major', width=font_size / 10 * 0.8, size=font_size / 10 * 3.5)  # fontsize of the tick labels
    plt.rc('ytick.minor', width=font_size / 10 * 0.6, size=font_size / 10 * 2)  # fontsize of the tick labels
    plt.rc('legend', fontsize=font_size)  # legend fontsize
    plt.rc('figure', titlesize=font_size)  # fontsize of the figure title
