import copy
import glob
import os
import sys
import warnings

import h5py
import numpy as np
from scipy.interpolate import interp1d

from petitRADTRANS.config import petitradtrans_config
from petitRADTRANS import prt_molmass
from petitRADTRANS import nat_cst as nc
from petitRADTRANS import phoenix
from petitRADTRANS.fort_input import fort_input as fi
from petitRADTRANS.fort_rebin import fort_rebin as fr
from petitRADTRANS.fort_spec import fort_spec as fs


class Radtrans:
    def __init__(
            self,
            pressures=None,
            line_species=None,
            gas_continuum_contributors=None,
            rayleigh_species=None,
            cloud_species=None,
            line_opacity_mode='c-k',
            lbl_opacity_sampling=1,
            scattering_in_emission=False,
            wavelengths_boundaries=None,
            test_ck_shuffle_comp=False,
            anisotropic_cloud_scattering='auto',
            hack_cloud_photospheric_optical_depths=None,
            use_detailed_line_absorber_names=True,
            path_input_data=petitradtrans_config['Paths']['prt_input_data_path']
    ):
        r"""Object for calculating spectra using a given set of opacities.

        Args:
            line_species (Optional):
                list of strings, denoting which line absorber species to include.
            rayleigh_species (Optional):
                list of strings, denoting which Rayleigh scattering species to include.
            cloud_species (Optional):
                list of strings, denoting which cloud opacity species to include.
            gas_continuum_contributors (Optional):
                list of strings, denoting which continuum absorber species to include.
            wavelengths_boundaries (Optional):
                list containing left and right border of wavelength region to be considered, in micron. If nothing else
                is specified, it will be equal to ``[0.05, 300]``, hence using the full petitRADTRANS wavelength range
                (0.11 to 250 microns for ``'c-k'`` mode, 0.3 to 30 microns for the ``'lbl'`` mode). The larger the
                range the longer the computation time.
            line_opacity_mode (Optional[string]):
                if equal to ``'c-k'``: use low-resolution mode, at :math:`\\lambda/\\Delta \\lambda = 1000`, with the
                correlated-k assumption. if equal to ``'lbl'``: use high-resolution mode, at
                :math:`\\lambda/\\Delta \\lambda = 10^6`, with a line-by-line treatment.
            scattering_in_emission (Optional[bool]):
                Will be ``False`` by default.
                If ``True`` scattering will be included in the emission spectral calculations. Note that this increases
                the runtime of pRT!
            lbl_opacity_sampling (Optional[int]):
                Will be ``None`` by default. If integer positive value, and if ``mode == 'lbl'`` is ``True``, then this
                will only consider every lbl_opacity_sampling-nth point of the high-resolution opacities.
                This may be desired in the case where medium-resolution spectra are required with a
                :math:`\\lambda/\\Delta \\lambda > 1000`, but much smaller than :math:`10^6`, which is the resolution
                of the ``lbl`` mode. In this case it may make sense to carry out the calculations with
                lbl_opacity_sampling = 10, for example, and then re-binning to the final desired resolution: this may
                save time! The user should verify whether this leads to solutions which are identical to the re-binned
                results of the fiducial :math:`10^6` resolution. If not, this parameter must not be used.
            use_detailed_line_absorber_names (Optional[bool]):
                False by default. If True, the keywords of the mass fraction dictionary handed to
                get_spectral_radiosities() and get_transit_radii() must match the line absorber names exactly,
                including line list and resolution flags.
                For example, if "H2O_ExoMol_R_10" is loaded, the mass fraction keyword has to be "H2O_ExoMol_R_10",
                instead of the nominal "H2O".
        """
        # TODO add wavelengths generator
        if pressures is None:
            warnings.warn("pressure was not set, initializing one layer at 1 bar")
            pressures = np.array([1.0])  # bar

        if line_species is None:
            self.line_species = []
        else:
            self.line_species = line_species

        if gas_continuum_contributors is None:
            self.gas_continuum_contributors = []
        else:
            self.gas_continuum_contributors = gas_continuum_contributors

        if rayleigh_species is None:
            self.rayleigh_species = []
        else:
            self.rayleigh_species = rayleigh_species

        if cloud_species is None:
            self.cloud_species = []
        else:
            self.cloud_species = cloud_species

        if line_opacity_mode not in ['c-k', 'lbl']:
            raise ValueError(f"opacity mode must be 'c-k'|'lbl', but was '{line_opacity_mode}'")

        self.line_opacity_mode = line_opacity_mode
        self.lbl_opacity_sampling = lbl_opacity_sampling
        self.scattering_in_emission = scattering_in_emission

        if wavelengths_boundaries is None:
            self.wavelengths_boundaries = np.array([0.05, 300.])  # um
        else:
            self.wavelengths_boundaries = wavelengths_boundaries

        self.test_ck_shuffle_comp = test_ck_shuffle_comp  # TODO find better name

        self.anisotropic_cloud_scattering = anisotropic_cloud_scattering
        self.hack_cloud_photospheric_optical_depths = hack_cloud_photospheric_optical_depths  # TODO find better name

        self.use_detailed_line_absorber_names = use_detailed_line_absorber_names
        self.path_input_data = path_input_data

        # Initialize line parameters
        if len(line_species) > 0:  # TODO init Radtrans even if there is no opacity
            self.frequencies, self.frequencies_bin_edges, self.g_size, i_start_opacities \
                = self._init_line_opacities_parameters()
        else:
            self.g_size = None
            self.frequencies = None
            self.frequencies_bin_edges = None
            i_start_opacities = None

        # Initialize pressure-dependent parameters
        # TODO is radius_hse (planetary radius at hydrostatic equilibrium) useful?
        # TODO line_opacities actually stores the final, combined opacities -> choose a better name
        self.pressures, \
            self.continuum_opacities, self.continuum_opacities_scattering, \
            self.continuum_opacities_scattering_emission, \
            self.contribution_emission, self.contribution_transmission, \
            self.radius_hydrostatic_equilibrium, \
            self.mean_molar_masses, \
            self.line_species_mass_fractions, self.cloud_species_mass_fractions, self.r_g = \
            self._init_pressure_dependent_parameters(pressures=pressures)

        # Some necessary definitions, also prepare arrays for fluxes, transmission radius...
        self.flux = np.zeros(self.frequencies.size, dtype='d', order='F')
        self.transit_radii = np.zeros(self.frequencies.size, dtype='d', order='F')

        # Initialize information attributes (will change when calculating spectra)
        # TODO put attributes that must be updated to calculate spectra into these dicts below
        self.emission_parameters = {}
        self.transmission_parameters = {}

        # Initialize emission information attributes
        self.gravity = None
        self.emission_geometry = None
        self.stellar_intensity = None
        self.mu_star = None
        self.skip_radiative_transfer_step = False

        # Read in the angle (mu) grid for the emission spectral calculations.
        mu_points = np.genfromtxt(os.path.join(self.path_input_data, 'opa_input_files', 'mu_points.dat'))
        self.mu = mu_points[:, 0]
        self.w_gauss_mu = mu_points[:, 1]

        # Initialize spectral information attributes
        self.scattering_in_transmission = False

        #  Default surface albedo and emissivity -- will be used only if the surface scattering is turned on.
        self.reflectance = 0 * np.ones_like(self.frequencies)  # TODO never updated?
        self.emissivity = 1 * np.ones_like(self.frequencies)

        # Initialize cloud and haze information attributes
        self.opaque_layers_top_pressure = None
        self.haze_factor = None
        self.gray_opacity = None
        self.power_law_opacity_350nm = None
        self.power_law_opacity_coefficient = None
        self.cloud_f_sed = None
        self.hack_cloud_wavelengths = None

        # Initialize derived variables  TODO check if some of these can be made private variables instead of attributes
        self.cloud_opacities = None  # TODO only as information?
        self.photon_destruction_probabilities = None
        self.opacities_rosseland = None
        self.optical_depths_rosseland = None

        # Initialize special variables
        self.hack_cloud_total_scattering_anisotropic = None
        self.hack_cloud_total_abs = None
        self.photon_radius = None

        # Initialize line opacities variables
        self.has_custom_line_opacities_tp_grid = {}
        self.line_opacities_temperature_profile_grid = {}
        self.line_opacities_temperature_grid_size = {}
        self.line_opacities_pressure_grid_size = {}
        self.line_opacities_grid = {}
        self.g_gauss = np.array(np.ones(1), dtype='d', order='F')
        self.weights_gauss = np.array(np.ones(1), dtype='d', order='F')

        # Initialize cloud opacities variables
        self.cloud_species_mode = None
        self.cloud_particles_densities = None
        self.cloud_species_absorption_opacities = None
        self.cloud_species_scattering_opacities = None
        self.cloud_particles_asymmetry_parameters = None
        self.cloud_wavelengths = None
        self.cloud_particle_radius_bins = None
        self.cloud_particles_radii = None

        # Load all opacities
        self.cia_species = {}

        self.load_all_opacities(i_start_opacities)

    def _clouds_have_effect(self, mass_mixing_ratios):
        """Check if the clouds have any effect, i.e. if the cloud species MMR is greater than 0.

        Args:
            mass_mixing_ratios: atmospheric mass mixing ratios
        """
        add_cloud_opacity = False

        if len(self.cloud_species) > 0:
            for i_spec in range(len(self.cloud_species)):
                if np.any(mass_mixing_ratios[self.cloud_species[i_spec]] > 0):
                    add_cloud_opacity = True  # add cloud opacity only if there are actually clouds

                    break

        return add_cloud_opacity

    def _init_line_opacities_parameters(self):
        """Initialize parameters useful for loading line opacities.
        This includes the frequency grid used for spectral calculations

        Returns:
            frequencies:
                (Hz) frequencies (center of bin) of the line opacities, also use for spectral calculations, of size N
            frequencies_bin_edges:
                (Hz) edges of the frequencies bins, of size N+1
            n_g:
                for correlated-k only, number of points used to sample the g-space (1 in the case lbl is used)
            start_index:
                index where to start reading .dat line-by-line opacity files, not used if all lbl files are in HDF5
        """
        if self.line_opacity_mode == 'c-k':  # correlated-k
            if self.scattering_in_emission and not self.test_ck_shuffle_comp:
                print("Emission scattering is enabled: enforcing test_ck_shuffle_comp = True")

                self.test_ck_shuffle_comp = True

            # Get dimensions of molecular opacity arrays for a given P-T point, they define the resolution.
            # Use the first entry of self.line_species for this, if given.
            opacities_dir = os.path.join(self.path_input_data, 'opacities', 'lines', 'corr_k', self.line_species[0])
            hdf5_files = glob.glob(opacities_dir + '/*.h5')  # check if first species is hdf5

            if hdf5_files:
                with h5py.File(hdf5_files[0], 'r') as f:
                    n_g = len(f['samples'][:])
                    frequencies_bin_edges = nc.c * f['bin_edges'][:][::-1]
            else:
                # Use classical pRT format
                # In the long run: move to hdf5 fully?
                # But: people calculate their own k-tables with my code sometimes now.
                # TODO make a code to convert Paul's k-tables into HDF5 (if it doesn't exists), and get rid of this
                size_frequencies, n_g = fi.get_freq_len(self.path_input_data, self.line_species[0])

                # Read the frequency range of the opacity data
                frequencies, frequencies_bin_edges = fi.get_freq(
                    self.path_input_data, self.line_species[0], size_frequencies
                )

            # Extend the wavelength range if user requests larger range than what first line opa species contains
            wavelengths = nc.c / frequencies_bin_edges * 1e4  # Hz to um

            if wavelengths[-1] < self.wavelengths_boundaries[1]:
                delta_log_wavelength = np.diff(np.log10(wavelengths))[-1]
                add_high = 1e1 ** np.arange(
                    np.log10(wavelengths[-1]),
                    np.log10(self.wavelengths_boundaries[-1]) + delta_log_wavelength,
                    delta_log_wavelength
                )[1:]
                wavelengths = np.concatenate((wavelengths, add_high))

            if wavelengths[0] > self.wavelengths_boundaries[0]:
                delta_log_wavelength = np.diff(np.log10(wavelengths))[0]
                add_low = 1e1 ** (-np.arange(
                    -np.log10(wavelengths[0]),
                    -np.log10(self.wavelengths_boundaries[0]) + delta_log_wavelength,
                    delta_log_wavelength
                )[1:][::-1])
                wavelengths = np.concatenate((add_low, wavelengths))

            frequencies_bin_edges = nc.c / (wavelengths * 1e-4)  # um to Hz
            frequencies = (frequencies_bin_edges[1:] + frequencies_bin_edges[:-1]) * 0.5

            # Cut the wavelength range if user requests smaller range than what first line opa species contains
            indices_within_boundaries = np.nonzero(np.logical_and(
                np.greater(nc.c / frequencies, self.wavelengths_boundaries[0] * 1e-4),
                np.less(nc.c / frequencies, self.wavelengths_boundaries[1] * 1e-4)
            ))[0]

            frequencies = np.array(frequencies[indices_within_boundaries], dtype='d', order='F')

            # Get the corresponding frequencies bin edges, +2 is to catch the upper bin edge
            frequencies_bin_edges = np.array(
                frequencies_bin_edges[indices_within_boundaries[0]:indices_within_boundaries[-1]+2],
                dtype='d',
                order='F'
            )

            start_index = -1
        elif self.line_opacity_mode == 'lbl':  # line-by-line
            size_frequencies = None
            start_index = None
            wavelengths_dat_file = None
            load_from_dat = False

            # Seek if there is a .dat file to load the wavelength grid from, otherwise a HDF5 file will be used
            for species in self.line_species:
                wavelengths_dat_file = os.path.join(
                    self.path_input_data, 'opacities', 'lines', 'line_by_line', species, 'wlen.dat'
                )

                if os.path.isfile(wavelengths_dat_file):
                    # Get dimensions of opacity arrays for a given P-T point
                    print(f"Loading file '{wavelengths_dat_file}'...")
                    size_frequencies, start_index, _ = fi.get_arr_len_array_bords(
                        self.wavelengths_boundaries[0] * 1e-4,  # um to cm
                        self.wavelengths_boundaries[1] * 1e-4,  # um to cm
                        wavelengths_dat_file
                    )

                    load_from_dat = True

                    break

            if not load_from_dat:  # load the wavelength grid from a HDF5 file
                # Load the wavelength grid
                opacities_file = os.path.join(
                    self.path_input_data, 'opacities', 'lines', 'line_by_line',
                    self.line_species[0] + '.otable.petitRADTRANS.h5'
                )

                with h5py.File(opacities_file, 'r') as f:
                    wavelength_grid = 1 / f['wavenumbers'][:]  # cm-1 to cm

                wavelength_grid = wavelength_grid[::-1]
                wavelength_min = self.wavelengths_boundaries[0] * 1e-4  # um to cm
                wavelength_max = self.wavelengths_boundaries[1] * 1e-4  # um to cm

                # Check if the requested wavelengths boundaries are within the file boundaries
                bad_boundaries = False

                if wavelength_min < wavelength_grid[0]:
                    bad_boundaries = True

                if wavelength_max > wavelength_grid[-1]:
                    bad_boundaries = True

                if bad_boundaries:
                    raise ValueError(f"Requested wavelength interval "
                                     f"({self.wavelengths_boundaries[0]}--{self.wavelengths_boundaries[1]}) "
                                     f"is out of opacities table wavelength grid "
                                     f"({1e4 * wavelength_grid[0]}--{1e4 * wavelength_grid[-1]})")

                # Get the freq. corresponding to the requested boundaries, with the request fully within the selection
                selection = np.nonzero(np.logical_and(
                    np.greater_equal(wavelength_grid, wavelength_min),
                    np.less_equal(wavelength_grid, wavelength_max)
                ))[0]
                selection = np.array([selection[0], selection[-1]])

                if wavelength_grid[selection[0]] > wavelength_min:
                    selection[0] -= 1

                if wavelength_grid[selection[-1]] < wavelength_max:
                    selection[-1] += 1

                if self.lbl_opacity_sampling > 1:
                    # Ensure that down-sampled wavelength upper bound >= requested wavelength upper bound
                    selection[-1] += self.lbl_opacity_sampling - 1

                frequencies = nc.c / wavelength_grid[selection[0]:selection[-1] + 1]  # cm to s-1
            else:
                if self.lbl_opacity_sampling > 1:
                    size_frequencies += self.lbl_opacity_sampling - 1

                frequencies = nc.c / fi.read_wlen(start_index, size_frequencies, wavelengths_dat_file)

            n_g = 1

            # Down-sample frequency grid in lbl mode if requested
            if self.lbl_opacity_sampling > 1:
                frequencies = frequencies[::self.lbl_opacity_sampling]

            frequencies_bin_edges = np.array(nc.c / self.calculate_bins_edges(nc.c / frequencies), dtype='d', order='F')
        else:
            raise ValueError(f"line opacity mode must be 'c-k' or 'lbl', but was '{self.line_opacity_mode}'")

        return frequencies, frequencies_bin_edges, n_g, start_index

    def _init_pressure_dependent_parameters(self, pressures):
        """Initialize opacity arrays at atmospheric structure dimensions, and set the atmospheric pressure array.

        Args:
            pressures:
                (bar) 1-d numpy array, sorted in increasing order, representing the atmospheric pressure.
                Will be converted to cgs internally.
        """
        pressures = pressures * 1e6  # bar to cgs
        n_layers = pressures.shape[0]
        continuum_opacities = np.zeros((self.frequencies.size, n_layers), dtype='d', order='F')
        continuum_opacities_scattering = np.zeros((self.frequencies.size, n_layers), dtype='d', order='F')
        continuum_opacities_scattering_emission = None
        contribution_emission = None
        contribution_transmission = None
        radius_hydrostatic_equilibrium = np.zeros(n_layers, dtype='d', order='F')

        mean_molar_masses = np.zeros(n_layers)

        if len(self.line_species) > 0:
            line_species_mass_fractions = np.zeros((n_layers, len(self.line_species)), dtype='d', order='F')
        else:
            # If there are no specified line species then we need at
            # least an array to contain the continuum opacities
            # I'll (mis)use the line_struc_kappas array for that
            line_species_mass_fractions = np.zeros((n_layers, 1), dtype='d', order='F')

        if len(self.cloud_species) > 0:
            cloud_species_mass_fractions = np.zeros((n_layers, len(self.cloud_species)), dtype='d', order='F')
            r_g = np.zeros((n_layers, len(self.cloud_species)), dtype='d', order='F')
        else:
            cloud_species_mass_fractions = None
            r_g = None

        return (pressures, continuum_opacities, continuum_opacities_scattering, continuum_opacities_scattering_emission,
                contribution_emission, contribution_transmission, radius_hydrostatic_equilibrium, mean_molar_masses,
                line_species_mass_fractions, cloud_species_mass_fractions, r_g)

    @staticmethod
    def _sort_pt_grid(pressure_temperature_grid_file):
        # Read the Ps and Ts
        pressure_temperature_grid = np.genfromtxt(pressure_temperature_grid_file)

        # Read the file names
        with open(pressure_temperature_grid_file, 'r') as f:
            lines = f.readlines()

        n_lines = len(lines)

        # Prepare the array to contain the pressures, temperatures, indices in the unsorted list.
        # Also prepare the list of unsorted names
        sorted_grid = np.ones((n_lines, 3))
        names = []

        # Fill the array and name list
        for i in range(n_lines):
            columns = lines[i].split(' ')

            sorted_grid[i, 0] = pressure_temperature_grid[i, 0]
            sorted_grid[i, 1] = pressure_temperature_grid[i, 1]
            sorted_grid[i, 2] = i

            if columns[-1][-1] == '\n':
                names.append(columns[-1][:-1])
            else:
                names.append(columns[-1])

        # Sort the array by temperature
        sorted_indices = np.argsort(sorted_grid[:, 1])
        sorted_grid = sorted_grid[sorted_indices, :]

        # Sort the array entries with constant temperatures by pressure
        n_pressures = 0

        for i in range(n_lines):
            if np.abs(sorted_grid[i, 1] - sorted_grid[0, 1]) > 1e-10:
                break

            n_pressures = n_pressures + 1

        n_temperatures = int(n_lines / n_pressures)

        for i in range(n_temperatures):
            sorted_grid_ = sorted_grid[i * n_pressures:(i + 1) * n_pressures, :]
            sorted_indices = np.argsort(sorted_grid_[:, 0])
            sorted_grid_ = sorted_grid_[sorted_indices, :]
            sorted_grid[i * n_pressures:(i + 1) * n_pressures, :] = sorted_grid_

        names_sorted = []

        for i in range(n_lines):
            names_sorted.append(names[int(sorted_grid[i, 2] + 0.01)])

        # Convert from bar to cgs
        sorted_grid[:, 0] = sorted_grid[:, 0] * 1e6

        return [sorted_grid[:, :-1][:, ::-1], names_sorted, n_temperatures, n_pressures]

    def add_rayleigh_scattering_opacities(self, temperatures, mass_fractions):
        """Add Rayleigh scattering opacities to scattering continuum opacities.

        Args:
            temperatures: temperatures in each atmospheric layer
            mass_fractions: dictionary of the Rayleigh scattering species mass fractions
        """
        wavelengths_angstroem = np.array(nc.c / self.frequencies * 1e8, dtype='d', order='F')

        for spec in self.rayleigh_species:
            haze_factor = 1.0

            if self.haze_factor is not None:
                haze_factor = self.haze_factor

            add_term = haze_factor * fs.add_rayleigh(
                spec,
                mass_fractions[spec],
                wavelengths_angstroem,
                self.mean_molar_masses,
                temperatures,
                self.pressures
            )

            self.continuum_opacities_scattering += add_term

    @staticmethod
    def calculate_bins_edges(middle_bin_points):
        """Calculate bin edges for middle bin points.

        Args:
            middle_bin_points: array of size N containing the middle bin points to calculate the bin edges of

        Returns:
            array of size N+1 containing the bins edges
        """
        bin_edges = [middle_bin_points[0] - (middle_bin_points[1] - middle_bin_points[0]) / 2]

        for i in range(int(len(middle_bin_points)) - 1):
            bin_edges.append(middle_bin_points[i] + (middle_bin_points[i + 1] - middle_bin_points[i]) / 2)

        bin_edges.append(
            middle_bin_points[len(middle_bin_points) - 1]
            + (middle_bin_points[len(middle_bin_points) - 1] - middle_bin_points[len(middle_bin_points) - 2]) / 2
        )

        return np.array(bin_edges)

    @staticmethod
    def calculate_h_minus_free_free_xsec(wavelengths, temperatures, electron_partial_pressure):
        """Calculate the H- free-free cross-section in units of cm^2 per H per e- pressure (in cgs).
        Source: "The Observation and Analysis of Stellar Photospheres" by David F. Gray, p. 156

        Args:
            wavelengths: (angstroem)
            temperatures:
            electron_partial_pressure:

        Returns:

        """

        index = (wavelengths >= 2600.) & (wavelengths <= 113900.)
        lamb_use = wavelengths[index]

        if temperatures >= 2500.:
            # Convert to Angstrom (from cgs)
            theta = 5040. / temperatures

            f0 = -2.2763 - 1.6850 * np.log10(lamb_use) \
                + 0.76661 * np.log10(lamb_use) ** 2. \
                - 0.053346 * np.log10(lamb_use) ** 3.
            f1 = 15.2827 - 9.2846 * np.log10(lamb_use) \
                + 1.99381 * np.log10(lamb_use) ** 2. \
                - 0.142631 * np.log10(lamb_use) ** 3.
            f2 = -197.789 + 190.266 * np.log10(lamb_use) - 67.9775 * np.log10(lamb_use) ** 2. \
                + 10.6913 * np.log10(lamb_use) ** 3. - 0.625151 * np.log10(lamb_use) ** 4.

            ret_val = np.zeros_like(wavelengths)
            ret_val[index] = 1e-26 * electron_partial_pressure * 1e1 ** (
                    f0 + f1 * np.log10(theta) + f2 * np.log10(theta) ** 2.)
            return ret_val

        else:

            return np.zeros_like(wavelengths)

    @staticmethod
    def calculate_h_minus_bound_free_xsec(border_lambda_angstroem):
        """
        Returns the H- bound-free cross-section in units of cm^2 \
        per H-, as defined on page 155 of
        "The Observation and Analysis of Stellar Photospheres"
        by David F. Gray
        """

        left = border_lambda_angstroem[:-1]
        right = border_lambda_angstroem[1:]
        diff = np.diff(border_lambda_angstroem)

        a = [
            1.99654,
            -1.18267e-5,
            2.64243e-6,
            -4.40524e-10,
            3.23992e-14,
            -1.39568e-18,
            2.78701e-23
        ]

        ret_val = np.zeros_like(border_lambda_angstroem[1:])

        index = right <= 1.64e4

        for i_a in range(len(a)):
            ret_val[index] += a[i_a] * (
                    right[index] ** (i_a + 1) - left[index] ** (i_a + 1)
            ) / (i_a + 1)

        index_bracket = (left < 1.64e4) & (right > 1.64e4)
        for i_a in range(len(a)):
            ret_val[index_bracket] += a[i_a] * (1.64e4 ** (i_a + 1) -
                                                left[index_bracket] ** (i_a + 1)) / (i_a + 1)

        index = (left + right) / 2. > 1.64e4
        ret_val[index] = 0.
        index = ret_val < 0.
        ret_val[index] = 0.

        return ret_val * 1e-18 / diff

    def get_cloud_opacities(self, temperatures, cloud_species_mass_fractions, mean_molar_masses, gravity,
                            cloud_particle_radius_distribution_std, f_sed=None, eddy_diffusion_coefficient=None,
                            radius=None, add_cloud_scattering_as_absorption=False,
                            cloud_particle_radius_distribution="lognormal", a_hans=None, b_hans=None,
                            get_cloud_contribution=False):
        """Calculate cloud opacities for a defined atmospheric structure.

        Args:
            temperatures:
            cloud_species_mass_fractions:
            mean_molar_masses:
            gravity:
            cloud_particle_radius_distribution_std:
            f_sed:
            eddy_diffusion_coefficient:
            radius:
            add_cloud_scattering_as_absorption:
            cloud_particle_radius_distribution:
            a_hans:
            b_hans:
            get_cloud_contribution:

        Returns:

        """
        # TODO this does much more than just calculating the cloud opacities
        rho = self.pressures / nc.kB / temperatures * mean_molar_masses * nc.amu

        if "hansen" in cloud_particle_radius_distribution.lower():
            if isinstance(b_hans, np.ndarray):
                if not b_hans.shape == (self.pressures.shape[0], len(self.cloud_species)):
                    raise ValueError(
                        "b_hans must be a float, a dictionary with arrays for each cloud species, "
                        f"or a numpy array with shape {(self.pressures.shape[0], len(self.cloud_species))}, "
                        f"but was of shape {np.shape(b_hans)}"
                    )
            elif isinstance(b_hans, dict):
                b_hans = np.array(list(b_hans.values()), dtype='d', order='F').T
            elif isinstance(b_hans, float):
                b_hans = np.array(
                    np.tile(b_hans * np.ones_like(self.pressures), (len(self.cloud_species), 1)),
                    dtype='d',
                    order='F'
                ).T
            else:
                raise ValueError(f"The Hansen distribution width (b_hans) must be an array, a dict, or a float, "
                                 f"but is of type '{type(b_hans)}' ({b_hans})")

        for i_spec, cloud_name in enumerate(self.cloud_species):
            self.cloud_species_mass_fractions[:, i_spec] = cloud_species_mass_fractions[cloud_name]

            if radius is not None:
                self.r_g[:, i_spec] = radius[cloud_name]
            elif a_hans is not None:
                self.r_g[:, i_spec] = a_hans[cloud_name]

        if radius is not None or a_hans is not None:
            if cloud_particle_radius_distribution == "lognormal":
                cloud_abs_opa_tot, cloud_scat_opa_tot, cloud_red_fac_anisotropic_tot = \
                    self.calculate_cloud_opacities(
                        rho=rho,
                        rho_p=self.cloud_particles_densities,
                        cloud_mass_fracs=self.cloud_species_mass_fractions,
                        r_g=self.r_g,
                        sigma_n=cloud_particle_radius_distribution_std,
                        cloud_rad_bins=self.cloud_particle_radius_bins,
                        cloud_radii=self.cloud_particles_radii,
                        cloud_specs_abs_opa=self.cloud_species_absorption_opacities,
                        cloud_specs_scat_opa=self.cloud_species_scattering_opacities,
                        cloud_aniso=self.cloud_particles_asymmetry_parameters
                    )
            else:
                cloud_abs_opa_tot, cloud_scat_opa_tot, cloud_red_fac_anisotropic_tot = \
                    fs.calc_hansen_opas(
                        rho,
                        self.cloud_particles_densities,
                        self.cloud_species_mass_fractions,
                        self.r_g,
                        b_hans,
                        self.cloud_particle_radius_bins,
                        self.cloud_particles_radii,
                        self.cloud_species_absorption_opacities,
                        self.cloud_species_scattering_opacities,
                        self.cloud_particles_asymmetry_parameters
                    )
        else:
            f_seds = np.zeros(len(self.cloud_species))

            for i_spec, cloud in enumerate(self.cloud_species):
                if isinstance(f_sed, dict):
                    f_seds[i_spec] = f_sed[cloud.split('_')[0]]
                elif not hasattr(f_sed, '__iter__'):
                    f_seds[i_spec] = f_sed

            if cloud_particle_radius_distribution == "lognormal":
                self.r_g = fs.get_rg_n(
                    gravity,
                    rho,
                    self.cloud_particles_densities,
                    temperatures,
                    mean_molar_masses,
                    f_seds,
                    cloud_particle_radius_distribution_std,
                    eddy_diffusion_coefficient
                )

                cloud_abs_opa_tot, cloud_scat_opa_tot, cloud_red_fac_anisotropic_tot = \
                    self.calculate_cloud_opacities(
                        rho,
                        self.cloud_particles_densities,
                        self.cloud_species_mass_fractions,
                        self.r_g,
                        cloud_particle_radius_distribution_std,
                        self.cloud_particle_radius_bins,
                        self.cloud_particles_radii,
                        self.cloud_species_absorption_opacities,
                        self.cloud_species_scattering_opacities,
                        self.cloud_particles_asymmetry_parameters
                    )
            else:
                self.r_g = fs.get_rg_n_hansen(
                    gravity,
                    rho,
                    self.cloud_particles_densities,
                    temperatures,
                    mean_molar_masses,
                    f_seds,
                    b_hans,
                    eddy_diffusion_coefficient
                )

                cloud_abs_opa_tot, cloud_scat_opa_tot, cloud_red_fac_anisotropic_tot = \
                    fs.calc_hansen_opas(
                        rho,
                        self.cloud_particles_densities,
                        self.cloud_species_mass_fractions,
                        self.r_g,
                        b_hans,
                        self.cloud_particle_radius_bins,
                        self.cloud_particles_radii,
                        self.cloud_species_absorption_opacities,
                        self.cloud_species_scattering_opacities,
                        self.cloud_particles_asymmetry_parameters
                    )

        # anisotropic = (1-g)
        cloud_abs, cloud_abs_plus_scat_anisotropic, anisotropic, cloud_abs_plus_scat_no_anisotropic = \
            fs.interp_integ_cloud_opas(
                cloud_abs_opa_tot,
                cloud_scat_opa_tot,
                cloud_red_fac_anisotropic_tot,
                self.cloud_wavelengths,
                self.frequencies_bin_edges
            )

        if self.anisotropic_cloud_scattering:
            self.continuum_opacities_scattering += cloud_abs_plus_scat_anisotropic - cloud_abs
        else:
            if self.scattering_in_emission and self.continuum_opacities_scattering_emission is not None:
                self.continuum_opacities_scattering_emission = copy.deepcopy(self.continuum_opacities_scattering)
                self.continuum_opacities_scattering_emission += cloud_abs_plus_scat_anisotropic - cloud_abs

            self.continuum_opacities_scattering += cloud_abs_plus_scat_no_anisotropic - cloud_abs

        if self.scattering_in_emission:
            if self.hack_cloud_photospheric_optical_depths is not None:
                self.hack_cloud_total_scattering_anisotropic = cloud_abs_plus_scat_anisotropic - cloud_abs
                self.hack_cloud_total_abs = cloud_abs

        if add_cloud_scattering_as_absorption:
            self.continuum_opacities += cloud_abs + 0.20 * (cloud_abs_plus_scat_no_anisotropic - cloud_abs)
        else:
            self.continuum_opacities += cloud_abs

        # This included scattering plus absorption
        if get_cloud_contribution:
            opacity_shape = (1, self.frequencies.size, 1, self.pressures.size)
            self.cloud_opacities = cloud_abs_plus_scat_anisotropic.reshape(opacity_shape)
        else:
            self.cloud_opacities = None

    def get_photon_radius(self, planet_radius, temperatures, mean_molar_masses, gravity, opacities):
        try:
            radius_hydrostatic_equilibrium = self.calculate_radius_hydrostatic_equilibrium(
                pressures=self.pressures * 1e-6,
                temperatures=temperatures,
                mean_molar_masses=mean_molar_masses,
                gravity=gravity,
                planet_radius=planet_radius,
                reference_pressure=self.pressures[-1] * 1e-6,
            )

            radius_interp = interp1d(self.pressures, radius_hydrostatic_equilibrium)

            self.photon_radius = np.zeros(self.frequencies.size)

            if self.line_opacity_mode == 'lbl' or self.test_ck_shuffle_comp:
                optical_depths = self.get_optical_depths(
                    gravity,
                    opacities=opacities,
                    cloud_wavelengths=self.hack_cloud_wavelengths
                )
                weights_gauss_reshape = self.weights_gauss.reshape(len(self.weights_gauss), 1)

                for i_freq in range(self.frequencies.size):
                    tau_p = np.sum(weights_gauss_reshape * optical_depths[:, i_freq, 0, :], axis=0)
                    pressures_tau_p = interp1d(tau_p, self.pressures)
                    self.photon_radius[i_freq] = radius_interp(pressures_tau_p(2. / 3.))
        except Exception:  # TODO find what is expected here
            self.photon_radius = -np.ones(self.frequencies.size)

    def get_flux(self, temp, mass_fractions, gravity, mmw, r_pl=None,
                 cloud_particle_radius_distribution_std=None,
                 fsed=None, kzz=None, radius=None,
                 contribution=False,
                 gray_opacity=None, p_cloud=None,
                 kappa_zero=None,
                 gamma_scat=None,
                 add_cloud_scat_as_abs=False,
                 t_star=None, r_star=None, orbit_semi_major_axis=None,
                 emission_geometry='dayside_ave', star_inclination_angle=0,
                 hack_cloud_photospheric_tau=None,
                 dist="lognormal", a_hans=None, b_hans=None,
                 stellar_intensity=None,
                 give_absorption_opacity=None,
                 give_scattering_opacity=None,
                 cloud_wavelengths=None,
                 get_photon_radius=False,
                 get_cloud_contribution=False
                 ):
        """ Method to calculate the atmosphere's emitted flux
        (emission spectrum).

            Args:
                temp:
                    the atmospheric temperature in K, at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                mass_fractions:
                    dictionary of mass fractions for all atmospheric absorbers.
                    Dictionary keys are the species names.
                    Every mass fraction array
                    has same length as pressure array.
                gravity (float):
                    Surface gravity in cgs. Vertically constant for emission
                    spectra.
                mmw:
                    the atmospheric mean molecular weight in amu,
                    at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                r_pl: planet radius at maximum pressure in cm. If specified, the planet's changing photospheric radius
                    as function of wavelength will be calculated and saved in the self.phot_radius attribute (in cm).
                cloud_particle_radius_distribution_std (Optional[float]):
                    width of the log-normal cloud particle size distribution
                fsed (Optional[float]):
                    cloud settling parameter
                kzz (Optional):
                    the atmospheric eddy diffusion coefficient in cgs
                    (i.e. :math:`\\rm cm^2/s`),
                    at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                radius (Optional):
                    dictionary of mean particle radii for all cloud species.
                    Dictionary keys are the cloud species names.
                    Every radius array has same length as pressure array.
                contribution (Optional[bool]):
                    If ``True`` the emission contribution function will be
                    calculated. Default is ``False``.
                gray_opacity (Optional[float]):
                    Gray opacity value, to be added to the opacity at all
                    pressures and wavelengths (units :math:`\\rm cm^2/g`)
                p_cloud (Optional[float]):
                    Pressure, in bar, where opaque cloud deck is added to the
                    absorption opacity.
                kappa_zero (Optional[float]):
                    Scattering opacity at 0.35 micron, in cgs units (cm^2/g).
                gamma_scat (Optional[float]):
                    Has to be given if kappa_zero is defined, this is the
                    wavelength powerlaw index of the parametrized scattering
                    opacity.
                add_cloud_scat_as_abs (Optional[bool]):
                    If ``True``, 20 % of the cloud scattering opacity will be
                    added to the absorption opacity, introduced to test for the
                    effect of neglecting scattering.  # TODO is it worth keeping?
                t_star (Optional[float]):
                    The temperature of the host star in K, used only if the
                    scattering is considered. If not specified, the direct
                    light contribution is not calculated.
                r_star (Optional[float]):
                    The radius of the star in cm. If specified,
                    used to scale the to scale the stellar flux,
                    otherwise it uses PHOENIX radius.
                orbit_semi_major_axis (Optional[float]):
                    The distance of the planet from the star. Used to scale
                    the stellar flux when the scattering of the direct light
                    is considered.
                emission_geometry (Optional[string]):
                    if equal to ``'dayside_ave'``: use the dayside average
                    geometry. if equal to ``'planetary_ave'``: use the
                    planetary average geometry. if equal to
                    ``'non-isotropic'``: use the non-isotropic
                    geometry.
                star_inclination_angle (Optional[float]):
                    Inclination angle of the direct light with respect to
                    the normal to the atmosphere. Used only in the
                    non-isotropic geometry scenario.
                hack_cloud_photospheric_tau (Optional[float]):
                    Median optical depth (across ``wavelengths_boundaries``) of the
                    clouds from the top of the atmosphere down to the gas-only
                    photosphere. This parameter can be used for enforcing the
                    presence of clouds in the photospheric region.
                dist (Optional[string]):
                    The cloud particle size distribution to use.
                    Can be either 'lognormal' (default) or 'hansen'.
                    If hansen, the b_hans parameters must be used.
                a_hans (Optional[dict]):
                    A dictionary of the 'a' parameter values for each
                    included cloud species and for each atmospheric layer,
                    formatted as the kzz argument. Equivalent to radius arg.
                    If a_hans is not included and dist is "hansen", then it will
                    be computed using Kzz and fsed (recommended).
                b_hans (Optional[dict]):
                    A dictionary of the 'b' parameter values for each
                    included cloud species and for each atmospheric layer,
                    formatted as the kzz argument. This is the width of the hansen
                    distribution normalized by the particle area (1/a_hans^2)
                give_absorption_opacity (Optional[function]):
                    A python function that takes wavelength arrays in microns and pressure arrays in bars
                    as input, and returns an absorption opacity matrix in units of cm^2/g, in the shape of
                    number of wavelength points x number of pressure points.
                    This opacity will then be added to the atmospheric absorption opacity.
                    This must not be used to add atomic / molecular line opacities in low-resolution mode (c-k),
                    because line opacities require a proper correlated-k treatment.
                    It may be used to add simple cloud absorption laws, for example, which
                    have opacities that vary only slowly with wavelength, such that the current
                    model resolution is sufficient to resolve any variations.
                stellar_intensity (Optional[array]):
                    The stellar intensity to use. If None, it will be calculated using a PHOENIX model.
                give_scattering_opacity (Optional[function]):
                    A python function that takes wavelength arrays in microns and pressure arrays in bars
                    as input, and returns an isotropic scattering opacity matrix in units of cm^2/g, in the shape of
                    number of wavelength points x number of pressure points.
                    This opacity will then be added to the atmospheric absorption opacity.
                    It may be used to add simple cloud absorption laws, for example, which
                    have opacities that vary only slowly with wavelength, such that the current
                    model resolution is sufficient to resolve any variations.
                cloud_wavelengths (Optional[Tuple[float, float]]):
                    Tuple with the wavelength range (in micron) that is used
                    for calculating the median optical depth of the clouds at
                    gas-only photosphere and then scaling the cloud optical
                    depth to the value of ``hack_cloud_photospheric_tau``. The
                    range of ``cloud_wavelengths`` should be encompassed by
                    ``wavelengths_boundaries``. The full wavelength range is used
                    when ``cloud_wavelengths=None``.
                get_photon_radius (Optional[bool]):
                    if True, the photon radius is calculated
                get_cloud_contribution (Optional[bool]):
                    if True, the cloud contribution is calculated
        """
        self.hack_cloud_photospheric_optical_depths = hack_cloud_photospheric_tau
        self.opaque_layers_top_pressure = p_cloud
        self.power_law_opacity_350nm = kappa_zero
        self.power_law_opacity_coefficient = gamma_scat
        self.gray_opacity = gray_opacity
        self.emission_geometry = emission_geometry
        self.mu_star = np.cos(np.deg2rad(star_inclination_angle))
        self.cloud_f_sed = fsed
        self.gravity = gravity
        self.hack_cloud_wavelengths = cloud_wavelengths

        if self.hack_cloud_wavelengths is not None and (
                self.hack_cloud_wavelengths[0] < 1e4 * nc.c / self.frequencies[0] or
                self.hack_cloud_wavelengths[1] > 1e4 * nc.c / self.frequencies[-1]):
            raise ValueError(
                f"cloud wavelength range must be within the interval "
                f"[{1e4 * nc.c / self.frequencies[0]}, {1e4 * nc.c / self.frequencies[-1]}], "
                f"but was {self.hack_cloud_wavelengths}"
            )

        if self.mu_star <= 0.:
            self.mu_star = 1e-8

        if stellar_intensity is None:
            if t_star is not None and orbit_semi_major_axis is not None:
                self.stellar_intensity = self.get_star_spectrum(t_star, orbit_semi_major_axis, self.frequencies, r_star)
            else:
                self.stellar_intensity = np.zeros_like(self.frequencies)
        else:
            self.stellar_intensity = stellar_intensity

        auto_anisotropic_cloud_scattering = False

        if self.anisotropic_cloud_scattering == 'auto':
            self.anisotropic_cloud_scattering = True
            auto_anisotropic_cloud_scattering = True
        elif not self.anisotropic_cloud_scattering:
            warnings.warn(f"anisotropic cloud scattering is recommended for emission spectra, "
                          f"but 'anisotropic_cloud_scattering' was set to {self.anisotropic_cloud_scattering}; "
                          f"set it to True or 'auto' to disable this warning")

        opacities = self.get_opacities(
            temperatures=temp,
            mass_fractions=mass_fractions,
            mean_molar_masses=mmw,
            gravity=gravity,
            cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
            fsed=fsed,
            kzz=kzz,
            radius=radius,
            add_cloud_scat_as_abs=add_cloud_scat_as_abs,
            dist=dist,
            a_hans=a_hans,
            b_hans=b_hans,
            get_cloud_contribution=get_cloud_contribution,
            give_absorption_opacity=give_absorption_opacity,
            give_scattering_opacity=give_scattering_opacity
        )

        if r_pl is not None and get_photon_radius:  # TODO what is the purpose of that?
            self.get_photon_radius(r_pl, temp, mmw, gravity, opacities)

        if auto_anisotropic_cloud_scattering:
            self.anisotropic_cloud_scattering = 'auto'

        if not self.skip_radiative_transfer_step:
            self._get_spectral_radiosities(
                temperatures=temp,
                opacities=opacities,
                contribution=contribution
            )

            if self._clouds_have_effect(mass_fractions) and get_cloud_contribution:
                self.get_cloud_optical_depths(gravity)

            if (self.line_opacity_mode == 'lbl' or self.test_ck_shuffle_comp) and len(self.line_species) > 1:
                if self.scattering_in_emission and self.opacities_rosseland is not None:
                    self.optical_depths_rosseland = fs.calc_tau_g_tot_ck(
                        gravity,
                        self.pressures,
                        self.opacities_rosseland.reshape(1, 1, 1, len(self.pressures))
                    ).reshape(len(self.pressures))
        else:
            warnings.warn("Cloud rescaling lead to nan opacities, skipping RT calculation!")

            self.flux = np.ones_like(self.frequencies) * np.nan
            self.contribution_emission = None
            self.skip_radiative_transfer_step = False

    def get_optical_depths(self, gravity, opacities, cloud_wavelengths=None):
        # Calculate optical depth for the total opacity.
        if self.line_opacity_mode == 'lbl' or self.test_ck_shuffle_comp:
            optical_depths = copy.deepcopy(opacities)
            cloud_scaling_factor = None

            if self.hack_cloud_photospheric_optical_depths is not None:
                if self.scattering_in_emission:
                    continuum_opacities_scattering_emission = copy.deepcopy(self.continuum_opacities_scattering)
                else:
                    continuum_opacities_scattering_emission = np.zeros(self.continuum_opacities_scattering.shape)

                # TODO is this block structure intended for the user or just here for the developer tests?
                block1 = True
                block2 = True
                block3 = True
                block4 = True

                ab = np.ones_like(self.line_species_mass_fractions)

                # BLOCK 1, subtract cloud, calc. tau for gas only
                if block1:
                    # Get continuum scattering opacity, without clouds:
                    continuum_opacities_scattering_emission -= self.hack_cloud_total_scattering_anisotropic

                    opacities = fi.mix_opas_ck(
                        ab,
                        opacities,
                        -self.hack_cloud_total_abs
                    )

                    # Calc. cloud-free optical depth
                    optical_depths[:, :, :1, :], self.photon_destruction_probabilities = \
                        fs.calc_tau_g_tot_ck_scat(
                            gravity,
                            self.pressures,
                            opacities[:, :, :1, :],
                            self.scattering_in_emission,
                            continuum_opacities_scattering_emission
                        )

                # BLOCK 2, calc optical depth of cloud only!
                total_tau_cloud = np.zeros_like(optical_depths)

                if block2:
                    # Reduce total (absorption) line opacity by continuum absorption opacity
                    # (those two were added in  before)
                    mock_line_cloud_continuum_only = np.zeros_like(opacities)

                    if not block1 and not block3 and not block4:
                        ab = np.ones_like(self.line_species_mass_fractions)

                    mock_line_cloud_continuum_only = \
                        fi.mix_opas_ck(ab, mock_line_cloud_continuum_only, self.hack_cloud_total_abs)

                    # Calc. optical depth of cloud only
                    total_tau_cloud[:, :, :1, :], photon_destruction_prob_cloud = \
                        fs.calc_tau_g_tot_ck_scat(
                            gravity,
                            self.pressures,
                            mock_line_cloud_continuum_only[:, :, :1, :],
                            self.scattering_in_emission,
                            self.hack_cloud_total_scattering_anisotropic
                        )

                    if (not block1 and not block3) and not block4:
                        print("Cloud only (for tests purposes...)!")
                        optical_depths[:, :, :1, :], self.photon_destruction_probabilities = \
                            total_tau_cloud[:, :, :1, :], photon_destruction_prob_cloud

                # BLOCK 3, calc. photospheric position of atmo without cloud,
                # determine cloud optical depth there, compare to
                # hack_cloud_photospheric_tau, calculate scaling ratio
                if block3:
                    median = True

                    if cloud_wavelengths is None:
                        # Use the full wavelength range for calculating the median
                        # optical depth of the clouds
                        wavelengths_select = np.ones(self.frequencies.shape[0], dtype=bool)

                    else:
                        # Use a smaller wavelength range for the median optical depth
                        # The units of cloud_wavelengths are converted from micron to cm
                        wavelengths_select = (nc.c / self.frequencies >= 1e-4 * cloud_wavelengths[0]) & \
                                      (nc.c / self.frequencies <= 1e-4 * cloud_wavelengths[1])

                    # Calculate the cloud-free optical depth per wavelength
                    w_gauss_photosphere = self.weights_gauss[..., np.newaxis, np.newaxis]
                    optical_depth = np.sum(w_gauss_photosphere * optical_depths[:, :, 0, :], axis=0)

                    if median:
                        optical_depth_integral = np.median(optical_depth[wavelengths_select, :], axis=0)
                    else:
                        optical_depth_integral = np.sum(
                            (optical_depth[1:, :] + optical_depth[:-1, :]) * np.diff(self.frequencies)[..., np.newaxis],
                            axis=0) / (self.frequencies[-1] - self.frequencies[0]) / 2.

                    optical_depth_cloud = np.sum(w_gauss_photosphere * total_tau_cloud[:, :, 0, :], axis=0)

                    if median:
                        optical_depth_cloud_integral = np.median(optical_depth_cloud[wavelengths_select, :], axis=0)
                    else:
                        optical_depth_cloud_integral = np.sum(
                            (optical_depth_cloud[1:, :] + optical_depth_cloud[:-1, :]) * np.diff(self.frequencies)[
                                ..., np.newaxis], axis=0) / \
                                                    (self.frequencies[-1] - self.frequencies[0]) / 2.

                    # Interpolate the pressure where the optical
                    # depth of cloud-free atmosphere is 1.0

                    press_bol_clear = interp1d(optical_depth_integral, self.pressures)

                    try:
                        p_phot_clear = press_bol_clear(1.)
                    except ValueError:
                        p_phot_clear = self.pressures[-1]

                    # Interpolate the optical depth of the
                    # cloud-only atmosphere at the pressure
                    # of the cloud-free photosphere
                    tau_bol_cloud = interp1d(self.pressures, optical_depth_cloud_integral)
                    tau_cloud_at_phot_clear = tau_bol_cloud(p_phot_clear)

                    # Apply cloud scaling
                    cloud_scaling_factor = self.hack_cloud_photospheric_optical_depths / tau_cloud_at_phot_clear

                    if len(self.cloud_f_sed) > 0:
                        max_rescaling = 1e100

                        for f in self.cloud_f_sed.keys():
                            mr = 2. * (self.cloud_f_sed[f] + 1.)
                            max_rescaling = min(max_rescaling, mr)

                        print(f"Scaling_physicality: {cloud_scaling_factor / max_rescaling}")

                # BLOCK 4, add scaled cloud back to opacities
                if block4:
                    # Get continuum scattering opacity, including clouds:
                    continuum_opacities_scattering_emission = \
                        (continuum_opacities_scattering_emission
                         + cloud_scaling_factor * self.hack_cloud_total_scattering_anisotropic)

                    opacities = \
                        fi.mix_opas_ck(ab, opacities, cloud_scaling_factor * self.hack_cloud_total_abs)

                    # Calc. total optical depth, including clouds
                    optical_depths[:, :, :1, :], self.photon_destruction_probabilities = \
                        fs.calc_tau_g_tot_ck_scat(
                            gravity,
                            self.pressures,
                            opacities[:, :, :1, :],
                            self.scattering_in_emission,
                            continuum_opacities_scattering_emission
                        )
            else:
                if self.scattering_in_emission:
                    continuum_opacities_scattering_ = self.continuum_opacities_scattering
                else:
                    continuum_opacities_scattering_ = np.zeros(self.continuum_opacities_scattering.shape)

                optical_depths[:, :, :1, :], self.photon_destruction_probabilities = \
                    fs.calc_tau_g_tot_ck_scat(
                        gravity,
                        self.pressures,
                        opacities[:, :, :1, :],
                        self.scattering_in_emission,
                        continuum_opacities_scattering_
                    )

            # To handle cases without any absorbers, where kappas are zero
            if len(self.line_species) == 0 \
                    and len(self.gas_continuum_contributors) == 0 \
                    and len(self.rayleigh_species) == 0 \
                    and len(self.cloud_species) == 0:
                print('No absorbers present, setting the photon'
                      ' destruction probability in the atmosphere to 1.')
                self.photon_destruction_probabilities[np.isnan(self.photon_destruction_probabilities)] = 1.

            # To handle cases when tau_cloud_at_Phot_clear = 0,
            # therefore cloud_scaling_factor = inf,
            # continuum_opacities_scattering_emission will contain nans and infs,
            # and photon_destruction_prob contains only nans
            if len(self.photon_destruction_probabilities[np.isnan(self.photon_destruction_probabilities)]) > 0.:
                print('Region of zero opacity detected, setting the photon'
                      ' destruction probability in this spectral range to 1.')
                self.photon_destruction_probabilities[np.isnan(self.photon_destruction_probabilities)] = 1.
                self.skip_radiative_transfer_step = True
        else:
            optical_depths = fs.calc_tau_g_tot_ck(
                gravity,
                self.pressures,
                opacities
            )

        return optical_depths

    @staticmethod
    def calculate_pressure_hydrostatic_equilibrium(mmw, gravity, r_pl, p0, temperature, radii, rk4=True):
        # TODO is it used?
        p0 = p0 * 1e6
        vs = 1. / radii
        r_pl_sq = r_pl ** 2

        def integrand(press):
            temp = temperature(press)
            mu = mmw(press / 1e6, temp)
            if not np.isscalar(mu):
                mu = mu[0]

            integral = mu * nc.amu * gravity * r_pl_sq / nc.kB / temp
            return integral

        pressures = [p0]
        dvs = np.diff(vs)
        press1 = p0
        chi1 = np.log(p0)
        for dv in dvs:
            k1 = integrand(press1)

            if rk4:
                chi2 = chi1 + 0.5 * dv * k1
                press2 = np.exp(chi2)
                k2 = integrand(press2)
                chi3 = chi1 + 0.5 * dv * k2
                press3 = np.exp(chi3)
                k3 = integrand(press3)
                chi4 = chi1 + dv * k3
                press4 = np.exp(chi4)
                k4 = integrand(press4)
                chi1 = chi1 + 1. / 6. * (k1 + 2 * k2 + 2 * k3 + k4) * dv
            else:
                chi1 = chi1 + dv * k1

            press1 = np.exp(chi1)
            pressures.append(press1)

        return np.array(pressures) / 1e6

    @staticmethod
    def calculate_radius_hydrostatic_equilibrium(pressures,
                                                 temperatures,
                                                 mean_molar_masses,
                                                 gravity,
                                                 reference_pressure,
                                                 planet_radius,
                                                 variable_gravity=True):
        pressures = pressures * 1e6  # bar to cgs
        reference_pressure = reference_pressure * 1e6  # bar to cgs

        rho = pressures * mean_molar_masses * nc.amu / nc.kB / temperatures
        radius = fs.calc_radius(
            pressures,
            gravity,
            rho,
            reference_pressure,
            planet_radius,
            variable_gravity
        )

        return radius

    def get_rosseland_planck_opacities(self, temperatures, mass_fractions, gravity, mmw, cloud_particle_radius_std=None,
                                       fsed=None, kzz=None, radius=None, gray_opacity=None, p_cloud=None,
                                       kappa_zero=None, gamma_scat=None, haze_factor=None, add_cloud_scat_as_abs=False,
                                       dist="lognormal", b_hans=None, a_hans=None):
        """ Method to calculate the atmosphere's Rosseland and Planck mean opacities.

            Args:
                temperatures:
                    the atmospheric temperature in K, at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                mass_fractions:
                    dictionary of mass fractions for all atmospheric absorbers.
                    Dictionary keys are the species names.
                    Every mass fraction array
                    has same length as pressure array.
                gravity (float):
                    Surface gravity in cgs. Vertically constant for emission
                    spectra.
                mmw:
                    the atmospheric mean molecular weight in amu,
                    at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                cloud_particle_radius_std (Optional[float]):
                    width of the log-normal cloud particle size distribution
                fsed (Optional[float]):
                    cloud settling parameter
                kzz (Optional):
                    the atmospheric eddy diffusion coefficient in cgs
                    (i.e. :math:`\\rm cm^2/s`),
                    at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                radius (Optional):
                    dictionary of mean particle radii for all cloud species.
                    Dictionary keys are the cloud species names.
                    Every radius array has same length as pressure array.
                gray_opacity (Optional[float]):
                    Gray opacity value, to be added to the opacity at all
                    pressures and wavelengths (units :math:`\\rm cm^2/g`)
                p_cloud (Optional[float]):
                    Pressure, in bar, where opaque cloud deck is added to the
                    absorption opacity.
                kappa_zero (Optional[float]):
                    Scattering opacity at 0.35 micron, in cgs units (cm^2/g).
                gamma_scat (Optional[float]):
                    Has to be given if kappa_zero is defined, this is the
                    wavelength powerlaw index of the parametrized scattering
                    opacity.
                haze_factor (Optional[float]):
                    Scalar factor, increasing the gas Rayleigh scattering
                    cross-section.
                add_cloud_scat_as_abs (Optional[bool]):
                    If ``True``, 20 % of the cloud scattering opacity will be
                    added to the absorption opacity, introduced to test for the
                    effect of neglecting scattering.
                dist (Optional[string]):
                    The cloud particle size distribution to use.
                    Can be either 'lognormal' (default) or 'hansen'.
                    If hansen, the b_hans parameters must be used.
                a_hans (Optional[dict]):
                    A dictionary of the 'a' parameter values for each
                    included cloud species and for each atmospheric layer,
                    formatted as the kzz argument. Equivalent to radius arg.
                    If a_hans is not included and dist is "hansen", then it will
                    be computed using Kzz and fsed (recommended).
                b_hans (Optional[dict]):
                    A dictionary of the 'b' parameter values for each
                    included cloud species and for each atmospheric layer,
                    formatted as the kzz argument. This is the width of the hansen
                    distribution normalized by the particle area (1/a_hans^2)
        """
        # TODO should be 2 separated functions
        if not self.scattering_in_emission:
            raise ValueError(
                "pRT must run in scattering_in_emission = True mode to calculate kappa_Rosseland and kappa_Planck'"
            )

        self.opaque_layers_top_pressure = p_cloud
        self.haze_factor = haze_factor
        self.power_law_opacity_350nm = kappa_zero
        self.power_law_opacity_coefficient = gamma_scat
        self.gray_opacity = gray_opacity

        auto_anisotropic_cloud_scattering = False

        if self.anisotropic_cloud_scattering == 'auto':
            self.anisotropic_cloud_scattering = True
            auto_anisotropic_cloud_scattering = True
        elif not self.anisotropic_cloud_scattering:
            warnings.warn(f"anisotropic cloud scattering is recommended for emission spectra, "
                          f"but 'anisotropic_cloud_scattering' was set to {self.anisotropic_cloud_scattering}; "
                          f"set it to 'auto' to disable this warning")

        opacities = self.get_opacities(
            temperatures=temperatures,
            mass_fractions=mass_fractions,
            mean_molar_masses=mmw,
            gravity=gravity,
            cloud_particle_radius_distribution_std=cloud_particle_radius_std,
            fsed=fsed,
            kzz=kzz,
            radius=radius,
            add_cloud_scat_as_abs=add_cloud_scat_as_abs,
            dist=dist,
            a_hans=a_hans,
            b_hans=b_hans
        )

        if auto_anisotropic_cloud_scattering:
            self.anisotropic_cloud_scattering = 'auto'

        self.opacities_rosseland = \
            fs.calc_kappa_rosseland(opacities[:, :, :1, :], temperatures,
                                    self.weights_gauss, self.frequencies_bin_edges,
                                    self.scattering_in_emission, self.continuum_opacities_scattering)

        opacities_planck = \
            fs.calc_kappa_planck(opacities[:, :, :1, :], temperatures,
                                 self.weights_gauss, self.frequencies_bin_edges,
                                 self.scattering_in_emission, self.continuum_opacities_scattering)

        return self.opacities_rosseland, opacities_planck

    def _get_spectral_radiosities(self, temperatures, opacities, contribution=False, get_kappa_rosseland=False):
        """Calculate the flux.
        """
        optical_depths = self.get_optical_depths(
            gravity=self.gravity,
            opacities=opacities,
            cloud_wavelengths=self.hack_cloud_wavelengths
        )

        if contribution:
            self.contribution_emission = np.zeros(
                (np.size(temperatures), self.frequencies.size), dtype='d', order='F'
            )

        if self.scattering_in_emission:
            # TODO investigate bug with scattering and low VMR near surface
            # print(np.shape(self.total_tau[:, :, 0, :]))
            # with open('tau.txt', 'w') as f:
            #     for i in range(np.shape(self.total_tau[:, :, 0, :])[0]):
            #         f.write('')
            #         for j in range(np.shape(self.total_tau[:, :, 0, :])[1]):
            #             line = self.total_tau[i, j, 0, :]
            #
            #             line = ' '.join(str(x) for x in line)
            #
            #             f.write(line)

            # raise ValueError('!')
            # Only use 0 index for species because for lbl or test_ck_shuffle_comp = True
            # everything has been moved into the 0th index
            if contribution:
                self.flux, self.contribution_emission = fs.feautrier_rad_trans(
                    self.frequencies_bin_edges,
                    optical_depths[:, :, 0, :],
                    temperatures,
                    self.mu,
                    self.w_gauss_mu,
                    self.weights_gauss,
                    self.photon_destruction_probabilities,
                    contribution,
                    self.reflectance,
                    self.emissivity,
                    self.stellar_intensity,
                    self.emission_geometry,
                    self.mu_star
                )
            else:
                self.flux, _ = fs.feautrier_rad_trans(
                    self.frequencies_bin_edges,
                    optical_depths[:, :, 0, :],
                    temperatures,
                    self.mu,
                    self.w_gauss_mu,
                    self.weights_gauss,
                    self.photon_destruction_probabilities,
                    contribution,
                    self.reflectance,
                    self.emissivity,
                    self.stellar_intensity,
                    self.emission_geometry,
                    self.mu_star
                )

            if get_kappa_rosseland:
                if self.scattering_in_emission:
                    self.opacities_rosseland = \
                        fs.calc_kappa_rosseland(
                            opacities[:, :, 0, :],
                            temperatures,
                            self.weights_gauss,
                            self.frequencies_bin_edges,
                            self.scattering_in_emission,
                            self.continuum_opacities_scattering
                        )
                else:
                    self.opacities_rosseland = \
                        fs.calc_kappa_rosseland(
                            opacities[:, :, 0, :],
                            temperatures,
                            self.weights_gauss,
                            self.frequencies_bin_edges,
                            self.scattering_in_emission,
                            np.zeros(self.continuum_opacities_scattering.shape)
                        )
        else:
            if (self.line_opacity_mode == 'lbl' or self.test_ck_shuffle_comp) and len(self.line_species) > 1:
                if contribution:
                    self.flux, self.contribution_emission = fs.flux_ck(
                        self.frequencies,
                        optical_depths[:, :, :1, :],
                        temperatures,
                        self.mu,
                        self.w_gauss_mu,
                        self.weights_gauss,
                        contribution
                    )
                else:
                    self.flux, _ = fs.flux_ck(
                        self.frequencies,
                        optical_depths[:, :, :1, :],
                        temperatures,
                        self.mu,
                        self.w_gauss_mu,
                        self.weights_gauss,
                        contribution
                    )
            else:
                if contribution:
                    self.flux, self.contribution_emission = fs.flux_ck(
                        self.frequencies,
                        optical_depths,
                        temperatures,
                        self.mu,
                        self.w_gauss_mu,
                        self.weights_gauss,
                        contribution
                    )
                else:
                    self.flux, _ = fs.flux_ck(
                        self.frequencies,
                        optical_depths,
                        temperatures,
                        self.mu,
                        self.w_gauss_mu,
                        self.weights_gauss,
                        contribution
                    )

    def get_cloud_optical_depths(self, surface_gravity):
        """Calculate the optical depth of the clouds as function of
        frequency and pressure. The array with the optical depths is set to the
        ``tau_cloud`` attribute. The optical depth is calculated from the top of
        the atmosphere (i.e. the smallest pressure). Therefore, below the cloud
        base, the optical depth is constant and equal to the value at the cloud
        base.

            Args:
                surface_gravity (float):
                    Surface gravity in cgs. Vertically constant for emission
                    spectra.
        """
        self.cloud_opacities = fs.calc_tau_g_tot_ck(surface_gravity, self.pressures, self.cloud_opacities)

    def _get_transit_radii(self, temperatures, p0_bar, r_pl, gravity, mmw, opacities, contribution, variable_gravity):
        if contribution:
            self.contribution_transmission = np.zeros(
                (np.size(self.pressures), self.frequencies.size), dtype='d', order='F'
            )

        # Calculate the transmission spectrum
        if (self.line_opacity_mode == 'lbl' or self.test_ck_shuffle_comp) and len(self.line_species) > 1:
            self.transit_radii, self.radius_hydrostatic_equilibrium = self.calculate_transit_radii(
                opacities=opacities,
                continuum_opacities_scattering=self.continuum_opacities_scattering,
                pressures=self.pressures * 1e-6,  # cgs to bar
                temperatures=temperatures,
                weights_gauss=self.weights_gauss,
                mean_molar_masses=mmw,
                gravity=gravity,
                reference_pressure=p0_bar,
                planet_radius=r_pl,
                variable_gravity=variable_gravity,
                line_by_line=True
            )

            # TODO: contribution function calculation with python-only implementation
            if contribution:
                self.transit_radii, self.radius_hydrostatic_equilibrium = fs.calc_transm_spec(
                    opacities[:, :, :1, :],
                    temperatures,
                    self.pressures,
                    gravity,
                    mmw,
                    p0_bar,
                    r_pl,
                    self.weights_gauss,
                    self.scattering_in_transmission,
                    self.continuum_opacities_scattering,
                    variable_gravity
                )

                self.contribution_transmission, self.radius_hydrostatic_equilibrium = fs.calc_transm_spec_contr(
                    opacities[:, :, :1, :],
                    temperatures,
                    self.pressures,
                    gravity,
                    mmw,
                    p0_bar,
                    r_pl,
                    self.weights_gauss,
                    self.transit_radii ** 2,
                    self.scattering_in_transmission,
                    self.continuum_opacities_scattering,
                    variable_gravity
                )
        else:
            self.transit_radii, self.radius_hydrostatic_equilibrium = self.calculate_transit_radii(
                opacities=opacities,
                continuum_opacities_scattering=self.continuum_opacities_scattering,
                pressures=self.pressures * 1e-6,  # cgs to bar
                temperatures=temperatures,
                weights_gauss=self.weights_gauss,
                mean_molar_masses=mmw,
                gravity=gravity,
                reference_pressure=p0_bar,
                planet_radius=r_pl,
                variable_gravity=variable_gravity,
                line_by_line=False
            )

            # TODO: contribution function calculation with python-only implementation
            if contribution:
                self.transit_radii, self.radius_hydrostatic_equilibrium = fs.calc_transm_spec(
                    opacities,
                    temperatures,
                    self.pressures,
                    gravity,
                    mmw,
                    p0_bar,
                    r_pl,
                    self.weights_gauss,
                    self.scattering_in_transmission,
                    self.continuum_opacities_scattering,
                    variable_gravity
                )

                self.contribution_transmission, self.radius_hydrostatic_equilibrium = fs.calc_transm_spec_contr(
                    opacities,
                    temperatures,
                    self.pressures,
                    gravity,
                    mmw,
                    p0_bar,
                    r_pl,
                    self.weights_gauss,
                    self.transit_radii ** 2.,
                    self.scattering_in_transmission,
                    self.continuum_opacities_scattering,
                    variable_gravity
                )

    def get_transit_radii(self, temp, mass_fractions, gravity, mmw, p0_bar, r_pl,
                          cloud_particle_radius_distribution_std=None,
                          fsed=None, kzz=None, radius=None,
                          p_cloud=None,
                          kappa_zero=None,
                          gamma_scat=None,
                          contribution=False, haze_factor=None,
                          gray_opacity=None, variable_gravity=True,
                          dist="lognormal", b_hans=None, a_hans=None,
                          get_cloud_contribution=False,
                          give_absorption_opacity=None,
                          give_scattering_opacity=None):
        """ Method to calculate the atmosphere's transmission radius
        (for the transmission spectrum).

            Args:
                temp:
                    the atmospheric temperature in K, at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                mass_fractions:
                    dictionary of mass fractions for all atmospheric absorbers.
                    Dictionary keys are the species names.
                    Every mass fraction array
                    has same length as pressure array.
                gravity (float):
                    Surface gravity in cgs at reference radius and pressure.
                mmw:
                    the atmospheric mean molecular weight in amu,
                    at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                p0_bar (float):
                    Reference pressure P0 in bar where R(P=P0) = R_pl,
                    where R_pl is the reference radius (parameter of this
                    method), and g(P=P0) = gravity, where gravity is the
                    reference gravity (parameter of this method)
                r_pl (float):
                    Reference radius R_pl, in cm.
                cloud_particle_radius_distribution_std (Optional[float]):
                    width of the log-normal cloud particle size distribution
                fsed (Optional[float]):
                    cloud settling parameter
                kzz (Optional):
                    the atmospheric eddy diffusion coefficient in cgs
                    (i.e. :math:`\\rm cm^2/s`),
                    at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                radius (Optional):
                    dictionary of mean particle radii for all cloud species.
                    Dictionary keys are the cloud species names.
                    Every radius array has same length as pressure array.
                contribution (Optional[bool]):
                    If ``True`` the transmission and emission
                    contribution function will be
                    calculated. Default is ``False``.
                gray_opacity (Optional[float]):
                    Gray opacity value, to be added to the opacity at all
                    pressures and wavelengths (units :math:`\\rm cm^2/g`)
                p_cloud (Optional[float]):
                    Pressure, in bar, where opaque cloud deck is added to the
                    absorption opacity.
                kappa_zero (Optional[float]):
                    Scattering opacity at 0.35 micron, in cgs units (cm^2/g).
                gamma_scat (Optional[float]):
                    Has to be given if kappa_zero is defined, this is the
                    wavelength powerlaw index of the parametrized scattering
                    opacity.
                haze_factor (Optional[float]):
                    Scalar factor, increasing the gas Rayleigh scattering
                    cross-section.
                variable_gravity (Optional[bool]):
                    Standard is ``True``. If ``False`` the gravity will be
                    constant as a function of pressure, during the transmission
                    radius calculation.
                dist (Optional[string]):
                    The cloud particle size distribution to use.
                    Can be either 'lognormal' (default) or 'hansen'.
                    If hansen, the b_hans parameters must be used.
                a_hans (Optional[dict]):
                    A dictionary of the 'a' parameter values for each
                    included cloud species and for each atmospheric layer,
                    formatted as the kzz argument. Equivalent to radius arg.
                    If a_hans is not included and dist is "hansen", then it will
                    be computed using Kzz and fsed (recommended).
                b_hans (Optional[dict]):
                    A dictionary of the 'b' parameter values for each
                    included cloud species and for each atmospheric layer,
                    formatted as the kzz argument. This is the width of the hansen
                    distribution normalized by the particle area (1/a_hans^2)
                get_cloud_contribution (Optional[bool]):
                    if True, the cloud contribution is calculated
                give_absorption_opacity (Optional[function]):
                    A python function that takes wavelength arrays in microns and pressure arrays in bars
                    as input, and returns an absorption opacity matrix in units of cm^2/g, in the shape of
                    number of wavelength points x number of pressure points.
                    This opacity will then be added to the atmospheric absorption opacity.
                    This must not be used to add atomic / molecular line opacities in low-resolution mode (c-k),
                    because line opacities require a proper correlated-k treatment.
                    It may be used to add simple cloud absorption laws, for example, which
                    have opacities that vary only slowly with wavelength, such that the current
                    model resolution is sufficient to resolve any variations.
                give_scattering_opacity (Optional[function]):
                    A python function that takes wavelength arrays in microns and pressure arrays in bars
                    as input, and returns an isotropic scattering opacity matrix in units of cm^2/g, in the shape of
                    number of wavelength points x number of pressure points.
                    This opacity will then be added to the atmospheric absorption opacity.
                    It may be used to add simple cloud absorption laws, for example, which
                    have opacities that vary only slowly with wavelength, such that the current
                    model resolution is sufficient to resolve any variations.
        """
        self.hack_cloud_photospheric_optical_depths = None
        self.opaque_layers_top_pressure = p_cloud
        self.gray_opacity = gray_opacity
        self.haze_factor = haze_factor
        self.power_law_opacity_350nm = kappa_zero
        self.power_law_opacity_coefficient = gamma_scat

        auto_anisotropic_cloud_scattering = False

        if self.anisotropic_cloud_scattering == 'auto':
            self.anisotropic_cloud_scattering = False
        elif self.anisotropic_cloud_scattering:
            warnings.warn(f"anisotropic cloud scattering is not recommended for transmission spectra, "
                          f"but 'anisotropic_cloud_scattering' was set to {self.anisotropic_cloud_scattering}; "
                          f"set it to False or 'auto' to disable this warning")

        opacities = self.get_opacities(
            temperatures=temp,
            mass_fractions=mass_fractions,
            mean_molar_masses=mmw,
            gravity=gravity,
            cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
            fsed=fsed,
            kzz=kzz,
            radius=radius,
            dist=dist,
            a_hans=a_hans,
            b_hans=b_hans,
            get_cloud_contribution=get_cloud_contribution,
            give_absorption_opacity=give_absorption_opacity,
            give_scattering_opacity=give_scattering_opacity
        )

        if auto_anisotropic_cloud_scattering:
            self.anisotropic_cloud_scattering = 'auto'

        self._get_transit_radii(
            temperatures=temp,
            p0_bar=p0_bar,
            r_pl=r_pl,
            gravity=gravity,
            mmw=mmw,
            opacities=opacities,
            contribution=contribution,
            variable_gravity=variable_gravity
        )

    @staticmethod
    def calculate_cloud_opacities(
            rho,  # (M,)
            rho_p,  # (N,)
            cloud_mass_fracs,  # (M, N)
            r_g,  # (M, N)
            sigma_n,
            cloud_rad_bins,  # (P + 1,)
            cloud_radii,  # (P,)
            cloud_specs_abs_opa,  # (P, Q, N)
            cloud_specs_scat_opa,  # (P, Q, N)
            cloud_aniso,  # (P, Q, N)
    ):
        r"""
        This function reimplements calc_cloud_opas from fort_spec.f90. For some reason
        it runs faster in python than in fortran, so we'll use this from now on.
        This function integrates the cloud opacity through the different layers of
        the atmosphere to get the total optical depth, scattering and anisotropic fraction.

        author: Francois Rozet
        """
        # TODO why outside Radtrans?
        n = (  # (M, N)
                3.0
                * cloud_mass_fracs
                * rho[:, None]
                / (4.0 * np.pi * rho_p * (r_g ** 3))
                * np.exp(-4.5 * np.log(sigma_n) ** 2)
        )

        diff = np.log(cloud_radii[:, None, None]) - np.log(r_g)
        dndr = (  # (P, M, N)
                n
                / (cloud_radii[:, None, None] * np.sqrt(2.0 * np.pi) * np.log(sigma_n))
                * np.exp(-diff ** 2 / (2.0 * np.log(sigma_n) ** 2))
        )

        integrand_scale = (  # (P, M, N)
                (4.0 * np.pi / 3.0)
                * cloud_radii[:, None, None] ** 3
                * rho_p
                * dndr
        )

        integrand_abs = integrand_scale[:, None] * cloud_specs_abs_opa[:, :, None]
        integrand_scat = integrand_scale[:, None] * cloud_specs_scat_opa[:, :, None]
        integrand_aniso = integrand_scat * (1.0 - cloud_aniso[:, :, None])

        widths = np.diff(cloud_rad_bins)[:, None, None, None]  # (P, 1, 1, 1)

        cloud_abs_opa = np.sum(integrand_abs * widths, axis=(0, 3))  # (Q, M)
        cloud_scat_opa = np.sum(integrand_scat * widths, axis=(0, 3))  # (Q, M)
        cloud_red_fac_aniso = np.sum(integrand_aniso * widths, axis=(0, 3))  # (Q, M)

        cloud_red_fac_aniso = np.true_divide(
            cloud_red_fac_aniso,
            cloud_scat_opa,
            out=np.zeros_like(cloud_scat_opa),
            where=cloud_scat_opa > 1e-200,
        )

        cloud_abs_opa = cloud_abs_opa / rho
        cloud_scat_opa = cloud_scat_opa / rho

        return cloud_abs_opa, cloud_scat_opa, cloud_red_fac_aniso

    def interpolate_cia(self, temperatures, key, combined_mas_fractions):
        """Interpolate CIA cross-sections onto the Radtrans (wavelength, temperature) grid and convert it into
        opacities.

        Args:
            key: collision (e.g. H2-He)
            combined_mas_fractions: combined mass fractions of the colliding species
                e.g., for H2-He and an atmosphere with H2 and He MMR of respectively 0.74 and 0.24,
                combined_mas_fractions = 0.74 * 0.24
                combined_mas_fractions is divided by the combined weight (e.g. for H2 and He, 2 * 4 AMU^2), so there is
                no units issue.

        Returns:
            A (wavelength, temperature) array containing the CIA opacities.
        """
        '''
        Dev note: this function is one of the costliest when calculating a spectrum.
        Interpolating on wavelengths during instantiation (since wavelengths will not change for a given Radtrans), then
        interpolating here does not significantly improve this function's performances.
        Using a (arguably more accurate) 2D interpolation using a scipy RegularGridInterpolator is also slower in the
        linear case. The 10 ** operation is costly, performing it before the wavelength interpolation is faster, but
        leads to > 1e-6 errors in the tox tests.
        '''
        factor = combined_mas_fractions / self.cia_species[key]['weight'] \
            * self.mean_molar_masses / nc.amu / (nc.L0 ** 2) * self.pressures / nc.kB / temperatures

        log10_alpha = np.log10(self.cia_species[key]['alpha'])

        if self.cia_species[key]['temperature'].shape[0] > 1:
            # Interpolation on temperatures for each wavelength point
            interpolating_function = interp1d(
                x=self.cia_species[key]['temperature'],
                y=log10_alpha,
                kind='linear',
                bounds_error=False,
                fill_value=(log10_alpha[:, 0], log10_alpha[:, -1]), axis=1
            )

            cia_opacities = interpolating_function(temperatures)

            interpolating_function = interp1d(
                x=self.cia_species[key]['lambda'],
                y=cia_opacities,
                kind='linear',
                bounds_error=False,
                fill_value=(np.log10(sys.float_info.min)),
                axis=0
            )

            cia_opacities = np.exp(interpolating_function(nc.c / self.frequencies) * np.log(10))

            cia_opacities = np.where(cia_opacities < sys.float_info.min, 0, cia_opacities)

            return cia_opacities * factor
        else:
            raise ValueError(f"petitRADTRANS require a rectangular CIA table, "
                             f"table shape was {self.cia_species[key]['temperature'].shape}")

    @staticmethod
    def _interpolate_species_opacities(pressures, temperatures, n_g, n_frequencies, line_opacities_grid,
                                       line_opacities_temperature_profile_grid, has_custom_line_opacities_tp_grid,
                                       line_opacities_temperature_grid_size, line_opacities_pressure_grid_size
                                       ):
        # Interpolate line opacities to given temperature structure.
        n_layers = pressures.size
        n_line_species = len(line_opacities_grid)

        if n_line_species > 0:
            line_opacities = np.zeros(
                (n_g, n_frequencies, n_line_species, n_layers), dtype='d', order='F'
            )

            for i, species in enumerate(line_opacities_grid):
                line_opacities[:, :, i, :] = fi.interpol_opa_ck(
                    pressures,
                    temperatures,
                    line_opacities_temperature_profile_grid[species],
                    has_custom_line_opacities_tp_grid[species],
                    line_opacities_temperature_grid_size[species],
                    line_opacities_pressure_grid_size[species],
                    line_opacities_grid[species]
                )
        else:
            line_opacities = np.zeros(
                (n_g, n_frequencies, 1, n_layers), dtype='d', order='F'
            )

        return line_opacities

    @staticmethod
    def get_custom_pt_grid(path, mode, species):
        """Check if custom grid exists, if yes return sorted P-T array with corresponding sorted path_input_data names,
        return None otherwise.
        This function can sort the grid appropriately if needed, but it must be specified by the user in
        the opacity folder of the relevant species.
        The custom grid must be rectangular.

        Args:
            path:
            mode:
            species:

        Returns:

        """
        pressure_temperature_grid_file = path + '/opacities/lines/'

        if mode == 'lbl':
            pressure_temperature_grid_file = pressure_temperature_grid_file + 'line_by_line/'
        elif mode == 'c-k':
            pressure_temperature_grid_file = pressure_temperature_grid_file + 'corr_k/'

        pressure_temperature_grid_file = pressure_temperature_grid_file + species + '/PTpaths.ls'

        if not os.path.isfile(pressure_temperature_grid_file):
            return None
        else:
            return Radtrans._sort_pt_grid(pressure_temperature_grid_file)

    @staticmethod
    def get_star_spectrum(t_star, distance, freq, r_star=None):
        """Method to get the PHOENIX spectrum of the star and rebin it
        to the wavelength points. If t_star is not explicitly written, the
        spectrum will be 0. If the distance is not explicitly written,
        the code will raise an error and break to urge the user to
        specify the value.

            Args:
                t_star (float):
                    the stellar temperature in K.
                distance (float):
                    the semi-major axis of the planet in cm.
                freq (float):
                    the frequencies on which to interpolate the spectrum in Hz.
                r_star (float):
                    if specified, uses this radius in cm
                    to scale the flux, otherwise it uses PHOENIX radius.
        """
        if r_star is not None:
            spec = phoenix.get_PHOENIX_spec(t_star)
            rad = r_star
        else:
            spec, rad = phoenix.get_PHOENIX_spec_rad(t_star)

        add_stellar_flux = np.zeros(100)
        add_wavelengths = np.logspace(np.log10(1.0000002e-02), 2, 100)

        wavelengths_interp = np.append(spec[:, 0], add_wavelengths)
        fluxes_interp = np.append(spec[:, 1], add_stellar_flux)

        stellar_intensity = fr.rebin_spectrum(wavelengths_interp, fluxes_interp, nc.c / freq)

        stellar_intensity = stellar_intensity / np.pi * (rad / distance) ** 2

        return stellar_intensity

    def load_all_opacities(self, start_index=None):
        # Load line opacities
        self.load_line_opacities(start_index, self.path_input_data)

        # Read continuum opacities
        # Clouds
        if len(self.cloud_species) > 0:
            # Inherited from ReadOpacities in _read_opacities.py
            self.load_cloud_opacities(self.path_input_data)

        # CIA
        if len(self.gas_continuum_contributors) > 0:
            self.cia_species = self.load_collision_induced_absorptions(
                self.path_input_data,
                self.gas_continuum_contributors
            )

    @staticmethod
    def load_collision_induced_absorptions(path_input_data, continuum_opacities):
        cia_species = {}

        for collision in continuum_opacities:
            if collision == 'H-':
                continue

            print(f"  Loading CIA opacities for {collision}...")

            hdf5_file = os.path.join(
                path_input_data, 'opacities', 'continuum', 'CIA', collision + '.ciatable.petitRADTRANS.h5'
            )

            if os.path.isfile(hdf5_file):
                with h5py.File(hdf5_file, 'r') as f:
                    wavelengths = 1 / f['wavenumbers'][:]  # cm-1 to cm
                    wavelengths = wavelengths[::-1]  # correct ordering

                    species = f['mol_name'][:]
                    species = [s.decode('utf-8') for s in species]

                    collision_dict = {
                        'id': collision,  # TODO remove useless key (duplicate of cia_species dict key)
                        'molecules': species,
                        'weight': np.prod(f['mol_mass'][:]),
                        'lambda': wavelengths,
                        'temperature': f['t'][:],
                        'alpha': np.transpose(f['cross_sections'][:])[::-1, :]  # (wavelength, temperature), cor. order
                    }
            else:
                print(f"HDF5 CIA file '{hdf5_file}' not found, loading from .dat...")

                cia_directory = os.path.join(path_input_data, 'opacities', 'continuum', 'CIA', collision)

                if os.path.isdir(cia_directory) is False:
                    raise FileNotFoundError(f"CIA directory '{cia_directory}' do not exists")

                # TODO what is the purpose of the *_dims variables?
                cia_wavelength_grid, cia_temperature_grid, cia_alpha_grid, \
                    cia_temp_dims, cia_lambda_dims = fi.cia_read(collision, path_input_data)
                cia_alpha_grid = np.array(cia_alpha_grid, dtype='d', order='F')
                cia_temperature_grid = cia_temperature_grid[:cia_temp_dims]
                cia_wavelength_grid = cia_wavelength_grid[:cia_lambda_dims]
                cia_alpha_grid = cia_alpha_grid[:cia_lambda_dims, :cia_temp_dims]

                weight = 1
                colliding_species = collision.split('-')

                for collision_dict in colliding_species:
                    weight = weight * molar_mass.getMM(collision_dict)

                collision_dict = {
                    'id': collision,
                    'molecules': colliding_species,
                    'weight': weight,
                    'lambda': cia_wavelength_grid,
                    'temperature': cia_temperature_grid,
                    'alpha': cia_alpha_grid
                }

            cia_species[collision] = collision_dict

        print('Done.\n')

        return cia_species

    @staticmethod
    def load_hdf5_ktables(file_path_hdf5, freq, g_len, freq_len, temperature_profile_grid_size):
        """Load k-coefficient tables in HDF5 format, based on the ExoMol setup."""
        with h5py.File(file_path_hdf5, 'r') as f:
            n_wavelengths = len(f['bin_centers'][:])
            frequencies = nc.c * f['bin_centers'][:][::-1]
            n_temperatures = len(f['t'][:])
            n_pressures = len(f['p'][:])

            # Swap axes to correctly load ExoMol tables.
            k_table = np.array(f['kcoeff'])
            k_table = np.swapaxes(k_table, 0, 1)
            k_table2 = k_table.reshape((n_pressures * n_temperatures, n_wavelengths, 16))
            k_table2 = np.swapaxes(k_table2, 0, 2)
            k_table2 = k_table2[:, ::-1, :]

            # Initialize an empty array that has the same spectral entries as
            # pRT object has nominally. Only fill those values where the ExoMol tables
            # have entries.
            ret_val = np.zeros(
                g_len * freq_len * temperature_profile_grid_size
            ).reshape(
                (g_len, freq_len, 1, temperature_profile_grid_size)
            )
            index_fill = (freq <= frequencies[0] * (1. + 1e-10)) & \
                         (freq >= frequencies[-1] * (1. - 1e-10))
            index_use = (frequencies <= freq[0] * (1. + 1e-10)) & \
                        (frequencies >= freq[-1] * (1. - 1e-10))
            ret_val[:, index_fill, 0, :] = k_table2[:, index_use, :]

            ret_val[ret_val < 0.] = 0.

            # Divide by mass to convert cross-sections to opacities
            mol_mass = float(f['mol_mass'][0])
            line_opacities_grid = ret_val / mol_mass / nc.amu

        return line_opacities_grid

    @staticmethod
    def load_hdf5_line_opacity_table(file_path_hdf5, freq, lbl_opacity_sampling=1):
        """Load opacities (cm2.g-1) tables in HDF5 format, based on petitRADTRANS pseudo-ExoMol setup."""
        with h5py.File(file_path_hdf5, 'r') as f:
            frequency_grid = nc.c * f['wavenumbers'][:]  # cm-1 to s-1

            selection = np.nonzero(np.logical_and(
                np.greater_equal(frequency_grid, np.min(freq)),
                np.less_equal(frequency_grid, np.max(freq))
            ))[0]
            selection = np.array([selection[0], selection[-1]])

            if lbl_opacity_sampling > 1:
                # Ensure that down-sampled wavelength upper bound >= requested wavelength upper bound
                selection[0] -= lbl_opacity_sampling - 1  # array is ordered by increasing wvn, so decreasing wvl

            line_opacities_grid = f['opacities'][:, :, selection[0]:selection[-1] + 1]
            line_opacities_grid /= f['isotopic_ratio'][()]  # the grid opacities are assuming the Earth isotopic ratio

        line_opacities_grid = line_opacities_grid[:, :, ::-1]

        if lbl_opacity_sampling > 1:
            line_opacities_grid = line_opacities_grid[:, :, ::lbl_opacity_sampling]

        if line_opacities_grid.shape[-1] != freq.size:
            frequency_grid = frequency_grid[selection[0]:selection[-1] + 1]
            frequency_grid = frequency_grid[::-1]

            if lbl_opacity_sampling > 1:
                frequency_grid = frequency_grid[::lbl_opacity_sampling]

            raise ValueError(f"file selected frequencies size is "
                             f"{line_opacities_grid.shape[-1]} ({np.min(frequency_grid)}--{np.max(frequency_grid)}), "
                             f"but frequency grid size is {freq.size} ({np.min(freq)}--{np.max(freq)})")

        line_opacities_grid = np.swapaxes(line_opacities_grid, 0, 1)  # (t, p, wvl)
        line_opacities_grid = line_opacities_grid.reshape(
            (line_opacities_grid.shape[0] * line_opacities_grid.shape[1], line_opacities_grid.shape[2])
        )  # (tp, wvl)
        line_opacities_grid = np.swapaxes(line_opacities_grid, 0, 1)  # (wvl, tp)
        line_opacities_grid = line_opacities_grid[np.newaxis, :, np.newaxis, :]  # (g, wvl, species, tp)

        return line_opacities_grid

    @staticmethod
    def load_dat_line_opacities(has_custom_line_opacities_temperature_profile_grid, opacities_temperature_profile_grid,
                                line_opacities_temperature_profile_grid, custom_line_paths, mode, path_input_data,
                                species, lbl_opacity_sampling, freq, freq_len, g_len, start_index):
        """Load k-coefficient tables or opacities tables in the classic petitRADTRANS .dat format."""
        if not has_custom_line_opacities_temperature_profile_grid:
            len_tp = len(opacities_temperature_profile_grid[:, 0])
        else:
            len_tp = len(line_opacities_temperature_profile_grid[:, 0])

        custom_file_names = ''

        if has_custom_line_opacities_temperature_profile_grid:
            for i_TP in range(len_tp):
                custom_file_names = custom_file_names + custom_line_paths[i_TP] + ':'

        # TODO PAUL EXO_K Project: do index_fill treatment from below!
        # Also needs to read "custom" freq_len and freq here again!
        # FOR ALL TABLES, HERE AND CHUBB: TEST THAT THE GRID IS INDEED THE SAME AS REQUIRED IN THE REGIONS
        # WITH OPACITY. NEXT STEPS AFTER THIS:
        # (i) MAKE ISOLATED EXO_K rebinning test,
        # (ii) Do external bin down script, save as pRT method
        # (iii) enable on the fly down-binning,
        # (iv) look into outsourcing methods from class to separate files, this here is getting too long!

        if mode == 'c-k':
            local_freq_len, local_g_len = fi.get_freq_len(path_input_data, species)
            local_freq_len_full = copy.copy(local_freq_len)

            # Read in the frequency range of the opacity data
            frequencies, bin_edges = fi.get_freq(path_input_data, species, local_freq_len)
        else:
            if lbl_opacity_sampling <= 1:
                local_freq_len_full = freq_len
            else:
                local_freq_len_full = freq_len * lbl_opacity_sampling

            local_g_len = g_len
            frequencies = None

        line_opacities_grid = fi.read_in_molecular_opacities(
            path_input_data,
            species + ':',
            local_freq_len_full,
            local_g_len,
            1,
            len_tp,
            mode,
            start_index,
            has_custom_line_opacities_temperature_profile_grid,
            custom_file_names
        )

        if np.all(line_opacities_grid == -1):
            raise RuntimeError("molecular opacity loading failed, check above outputs to find the cause")

        if mode == 'c-k':
            # Initialize an empty array that has the same spectral entries as
            # pRT object has nominally. Only fill those values where the k-tables
            # have entries.
            ret_val = np.zeros(g_len * freq_len * len_tp).reshape((g_len, freq_len, 1, len_tp))

            # Indices in retVal to be filled with read-in opacities
            index_fill = (freq <= frequencies[0] * (1. + 1e-10)) & \
                         (freq >= frequencies[-1] * (1. - 1e-10))
            # Indices of read-in opacities to be filled into retVal
            index_use = (frequencies <= freq[0] * (1. + 1e-10)) & \
                        (frequencies >= freq[-1] * (1. - 1e-10))

            ret_val[:, index_fill, 0, :] = \
                line_opacities_grid[:, index_use, 0, :]
            line_opacities_grid = ret_val

        # Down-sample opacities in lbl mode if requested
        if mode == 'lbl' and lbl_opacity_sampling > 1:
            line_opacities_grid = line_opacities_grid[:, ::lbl_opacity_sampling, :]

        return line_opacities_grid

    @staticmethod
    def load_line_opacities_pressure_temperature_grid(file_path_hdf5, path_input_data, mode, species,
                                                      opacities_temperature_profile_grid,
                                                      default_temperature_grid_size, default_pressure_grid_size):
        """Load line opacities temperature grids."""
        custom_line_paths = None

        if file_path_hdf5 is None:
            # Check and sort custom grid for species, if defined.
            custom_grid_data = Radtrans.get_custom_pt_grid(
                path_input_data,
                mode,
                species
            )

            # If no custom grid was specified (no PTpaths.ls found), take nominal grid
            # This assumes that the files indeed are following the nominal grid and naming convention
            # Otherwise, it will take the info provided in PTpaths.ls which was filled into custom_grid_data
            if custom_grid_data is None:
                line_opacities_temperature_profile_grid = opacities_temperature_profile_grid
                line_opacities_temperature_grid_size = default_temperature_grid_size
                line_opacities_pressure_grid_size = default_pressure_grid_size
                has_custom_line_opacities_temperature_profile_grid = False
            else:
                line_opacities_temperature_profile_grid = custom_grid_data[0]
                custom_line_paths = custom_grid_data[1]
                line_opacities_temperature_grid_size = custom_grid_data[2]
                line_opacities_pressure_grid_size = custom_grid_data[3]
                has_custom_line_opacities_temperature_profile_grid = True
        else:
            with h5py.File(file_path_hdf5, 'r') as f:
                pressure_grid = f['p'][:]
                temperature_grid = f['t'][:]

            ret_val = np.zeros((temperature_grid.size * pressure_grid.size, 2))

            for i_t in range(temperature_grid.size):
                for i_p in range(pressure_grid.size):
                    ret_val[i_t * pressure_grid.size + i_p, 1] = pressure_grid[i_p] * 1e6  # bar to cgs
                    ret_val[i_t * pressure_grid.size + i_p, 0] = temperature_grid[i_t]

            line_opacities_temperature_profile_grid = ret_val
            line_opacities_temperature_grid_size = temperature_grid.size
            line_opacities_pressure_grid_size = pressure_grid.size
            has_custom_line_opacities_temperature_profile_grid = True

        return line_opacities_temperature_profile_grid, custom_line_paths, line_opacities_temperature_grid_size, \
            line_opacities_pressure_grid_size, has_custom_line_opacities_temperature_profile_grid

    def get_opacities(self, temperatures, mass_fractions, mean_molar_masses, gravity,
                      cloud_particle_radius_distribution_std=None, fsed=None, kzz=None,
                      radius=None,
                      add_cloud_scat_as_abs=False,
                      dist="lognormal", a_hans=None,
                      b_hans=None, get_cloud_contribution=False,
                      give_absorption_opacity=None,
                      give_scattering_opacity=None):
        """Combine total line opacities, according to mass fractions (abundances), also add continuum opacities,
        i.e. clouds, CIA...
        TODO complete docstring

        Args:
            temperatures:
            mass_fractions:
            mean_molar_masses:
            gravity:
            cloud_particle_radius_distribution_std:
            fsed:
            kzz:
            radius:
            add_cloud_scat_as_abs:
            dist:
            a_hans:
            b_hans:
            get_cloud_contribution:
            give_absorption_opacity:
            give_scattering_opacity:

        Returns:

        """
        self.mean_molar_masses = mean_molar_masses
        self.scattering_in_transmission = False

        # Fill line abundance dictionary with provided mass fraction dictionary "abundances"
        for i_spec in range(len(self.line_species)):
            # Check if user provided the detailed line absorber name or if line absorber name should be matched exactly
            if self.line_species[i_spec] in mass_fractions or self.use_detailed_line_absorber_names:
                self.line_species_mass_fractions[:, i_spec] = mass_fractions[self.line_species[i_spec]]
            else:
                # Cut off everything after the first '_', to get rid of, for example, things like "_HITEMP_R_10"
                self.line_species_mass_fractions[:, i_spec] = mass_fractions[self.line_species[i_spec].split('_')[0]]

        self.continuum_opacities = np.zeros_like(self.continuum_opacities)
        self.continuum_opacities_scattering = np.zeros_like(self.continuum_opacities_scattering)

        # Calculate CIA opacity
        for key in self.cia_species.keys():
            combined_mass_fraction = 1

            for m in self.cia_species[key]['molecules']:
                if m in mass_fractions:
                    combined_mass_fraction = combined_mass_fraction * mass_fractions[m]
                else:
                    found = False

                    for species_ in mass_fractions:
                        species = species_.split('_', 1)[0]

                        if species == m:
                            combined_mass_fraction = combined_mass_fraction * mass_fractions[species_]
                            found = True

                            break

                    if not found:
                        raise ValueError(f"species {m} of CIA '{key}' not found in mass mixing ratios dict "
                                         f"(listed species: {list(mass_fractions.keys())})")

            self.continuum_opacities = self.continuum_opacities + self.interpolate_cia(
                temperatures,
                key,
                combined_mass_fraction
            )

        # Calc. H- opacity
        if 'H-' in self.gas_continuum_contributors:
            self.continuum_opacities = \
                self.continuum_opacities + self.calculate_h_minus_opacities(
                    nc.c / self.frequencies * 1e8,  # Hz to Angstroem
                    nc.c / self.frequencies_bin_edges * 1e8,  # Hz to Angstroem
                    temperatures,
                    self.pressures,
                    mean_molar_masses,
                    mass_fractions
                )

        # Add mock gray cloud opacity here
        if self.gray_opacity is not None:
            self.continuum_opacities = self.continuum_opacities + self.gray_opacity

        # Calculate rayleigh scattering opacities
        if len(self.rayleigh_species) != 0:
            self.scattering_in_transmission = True
            self.add_rayleigh_scattering_opacities(temperatures, mass_fractions)

        # Add gray cloud deck
        if self.opaque_layers_top_pressure is not None:
            self.continuum_opacities[:, self.pressures > self.opaque_layers_top_pressure * 1e6] += 1e99  # TODO why '+=' and not '='?  # noqa E501

        # Add power law opacity
        if self.power_law_opacity_350nm is not None:
            self.scattering_in_transmission = True
            wavelengths = nc.c / self.frequencies / 1e-4  # Hz to um
            scattering_add = self.power_law_opacity_350nm * (wavelengths / 0.35) ** self.power_law_opacity_coefficient
            add_term = np.repeat(scattering_add[None], int(len(self.pressures)), axis=0).transpose()

            self.continuum_opacities_scattering += add_term

        # Check if hack_cloud_photospheric_tau is used with
        # a single cloud model. Combining cloud opacities
        # from different models is currently not supported
        # with the hack_cloud_photospheric_tau parameter
        if len(self.cloud_species) > 0 and self.hack_cloud_photospheric_optical_depths is not None:
            if give_absorption_opacity is not None or give_scattering_opacity is not None:
                raise ValueError("The hack_cloud_photospheric_tau can only be "
                                 "used in combination with a single cloud model. "
                                 "Either use a physical cloud model by choosing "
                                 "cloud_species or use parametrized cloud "
                                 "opacities with the give_absorption_opacity "
                                 "and give_scattering_opacity parameters.")

        # Add optional absorption opacity from outside
        if give_absorption_opacity is None:
            if self.hack_cloud_photospheric_optical_depths is not None:
                if not hasattr(self, "hack_cloud_total_abs"):
                    opa_shape = (self.frequencies.shape[0], self.pressures.shape[0])
                    self.hack_cloud_total_abs = np.zeros(opa_shape)
        else:
            cloud_abs = give_absorption_opacity(nc.c / self.frequencies / 1e-4, self.pressures * 1e-6)
            self.continuum_opacities += cloud_abs

            if self.hack_cloud_photospheric_optical_depths is not None:
                # This assumes a single cloud model that is
                # given by the parametrized opacities from
                # give_absorption_opacity and give_scattering_opacity
                self.hack_cloud_total_abs = cloud_abs

        # Add optional scatting opacity from outside
        if give_scattering_opacity is None:
            if self.hack_cloud_photospheric_optical_depths is not None:
                if not hasattr(self, "hack_cloud_total_scattering_anisotropic"):
                    opa_shape = (self.frequencies.shape[0], self.pressures.shape[0])
                    self.hack_cloud_total_scattering_anisotropic = np.zeros(opa_shape)
        else:
            cloud_scat = give_scattering_opacity(nc.c / self.frequencies / 1e-4, self.pressures * 1e-6)
            self.continuum_opacities_scattering += cloud_scat

            if self.hack_cloud_photospheric_optical_depths is not None:
                # This assumes a single cloud model that is
                # given by the parametrized opacities from
                # give_absorption_opacity and give_scattering_opacity
                self.hack_cloud_total_scattering_anisotropic = cloud_scat

        # Add cloud opacity here, will modify self.continuum_opa
        if self._clouds_have_effect(mass_fractions):  # add cloud opacity only if there is actually clouds
            self.scattering_in_transmission = True
            self.get_cloud_opacities(
                temperatures,
                mass_fractions,
                mean_molar_masses,
                gravity,
                cloud_particle_radius_distribution_std,
                fsed,
                kzz,
                radius,
                add_cloud_scat_as_abs,
                cloud_particle_radius_distribution=dist,
                a_hans=a_hans,
                b_hans=b_hans,
                get_cloud_contribution=get_cloud_contribution
            )

        # Interpolate line opacities, combine with continuum opacities
        line_opacities = self._interpolate_species_opacities(
            pressures=self.pressures,
            temperatures=temperatures,
            n_g=self.g_size,
            n_frequencies=self.frequencies.size,
            line_opacities_grid=self.line_opacities_grid,
            line_opacities_temperature_profile_grid=self.line_opacities_temperature_profile_grid,
            has_custom_line_opacities_tp_grid=self.has_custom_line_opacities_tp_grid,
            line_opacities_temperature_grid_size=self.line_opacities_temperature_grid_size,
            line_opacities_pressure_grid_size=self.line_opacities_pressure_grid_size
        )

        line_opacities = fi.mix_opas_ck(
            self.line_species_mass_fractions,
            line_opacities,
            self.continuum_opacities
        )

        # Similar to the line-by-line case below, if test_ck_shuffle_comp is True, we will put the total opacity into
        # the first species slot and then carry the remaining radiative transfer steps only over that 0 index
        if self.line_opacity_mode == 'c-k' and self.test_ck_shuffle_comp:
            line_opacities[:, :, 0, :] = fs.combine_opas_ck(
                line_opacities,
                self.g_gauss,
                self.weights_gauss
            )

        # In the line-by-line case we can simply add the opacities of different species in frequency space
        # All opacities are stored in the first species index slot
        if self.line_opacity_mode == 'lbl' and len(self.line_species) > 1:
            line_opacities[:, :, 0, :] = np.sum(line_opacities, axis=2)

        return line_opacities

    def plot_opacities(self,
                       species,
                       temperature,
                       pressure_bar,
                       mass_fraction=None,
                       co=0.55,
                       feh=0.,
                       return_opacities=False,
                       **kwargs):
        # TODO move outside (no plots in Radtrans)
        import matplotlib.pyplot as plt

        def get_kappa(t):
            """ Method to calculate and return the line opacities (assuming an abundance
            of 100 % for the individual species) of the Radtrans object. This method
            updates the line_struc_kappas attribute within the Radtrans class. For the
            low resolution (`c-k`) mode, the wavelength-mean within every frequency bin
            is returned.

                Args:
                    t:
                        the atmospheric temperature in K, at each atmospheric layer
                        (1-d numpy array, same length as pressure array).

                Returns:
                    * wavelength in cm (1-d numpy array)
                    * dictionary of opacities, keys are the names of the line_species
                      dictionary, entries are 2-d numpy arrays, with the shape
                      being (number of frequencies, number of atmospheric layers).
                      Units are cm^2/g, assuming an absorber abundance of 100 % for all
                      respective species.

            """

            # Function to calc flux, called from outside
            _opacities = self._interpolate_species_opacities(
                pressures=self.pressures,
                temperatures=t,
                n_g=self.g_size,
                n_frequencies=self.frequencies.size,
                line_opacities_grid=self.line_opacities_grid,
                line_opacities_temperature_profile_grid=self.line_opacities_temperature_profile_grid,
                has_custom_line_opacities_tp_grid=self.has_custom_line_opacities_tp_grid,
                line_opacities_temperature_grid_size=self.line_opacities_temperature_grid_size,
                line_opacities_pressure_grid_size=self.line_opacities_pressure_grid_size
            )

            opacities_dict = {}

            weights_gauss = self.weights_gauss.reshape((len(self.weights_gauss), 1, 1))

            for i, s in enumerate(self.line_species):
                opacities_dict[s] = np.sum(_opacities[:, :, i, :] * weights_gauss, axis=0)

            return nc.c / self.frequencies, opacities_dict

        temp = np.array(temperature)
        pressure_bar = np.array(pressure_bar)

        temp = temp.reshape(1)
        pressure_bar = pressure_bar.reshape(1)

        self.pressures, \
            self.continuum_opacities, self.continuum_opacities_scattering, \
            self.continuum_opacities_scattering_emission, \
            self.contribution_emission, self.contribution_transmission, self.radius_hydrostatic_equilibrium, \
            self.mean_molar_masses, \
            self.line_species_mass_fractions, self.cloud_species_mass_fractions, self.r_g = \
            self._init_pressure_dependent_parameters(pressures=pressure_bar)

        wavelengths, opacities = get_kappa(temp)
        wavelengths *= 1e4  # cm to um

        plt_weights = {}
        if mass_fraction is None:
            for spec in species:
                plt_weights[spec] = 1.
        elif mass_fraction == 'eq':
            from .poor_mans_nonequ_chem import interpol_abundances
            ab = interpol_abundances(co * np.ones_like(temp),
                                     feh * np.ones_like(temp),
                                     temp,
                                     pressure_bar)
            # print('ab', ab)
            for spec in species:
                plt_weights[spec] = ab[spec.split('_')[0]]
        else:
            for spec in species:
                plt_weights[spec] = mass_fraction[spec]

        if return_opacities:
            rets = {}

            for spec in species:
                rets[spec] = [
                    wavelengths,
                    plt_weights[spec] * opacities[spec]
                ]

            return rets
        else:
            for spec in species:
                plt.plot(
                    wavelengths,
                    plt_weights[spec] * opacities[spec],
                    label=spec,
                    **kwargs
                )

    @staticmethod
    def calculate_transit_radii(opacities, continuum_opacities_scattering, pressures, temperatures, weights_gauss,
                                mean_molar_masses, gravity, reference_pressure,
                                planet_radius, variable_gravity, line_by_line):
        """Calculate the planetary transmission spectrum.

            Args:
                opacities:
                continuum_opacities_scattering:
                pressures:
                temperatures:
                weights_gauss:
                mean_molar_masses:
                    Mean molecular weight in units of amu.
                    (1-d numpy array, same length as pressure array).
                gravity (float):
                    Atmospheric gravitational acceleration at reference pressure and radius in units of
                    dyne/cm^2
                reference_pressure (float):
                    Reference pressure in bar.
                planet_radius (float):
                    Planet radius in cm.
                variable_gravity (bool):
                    If true, gravity in the atmosphere will vary proportional to 1/r^2, where r is the planet
                    radius.
                line_by_line (bool):
                    If true function assumes that pRT is running in lbl mode.

            Returns:
                * transmission radius in cm (1-d numpy array, as many elements as wavelengths)
                * planet radius as function of atmospheric pressure (1-d numpy array, as many elements as atmospheric
                layers)
        """
        # How many layers are there?
        struc_len = np.size(pressures)
        freq_len = np.size(opacities, axis=1)

        # Calculate planetary radius in hydrostatic equilibrium, using the atmospheric structure
        # (temperature, pressure, mmw), gravity, reference pressure and radius.
        radius = Radtrans.calculate_radius_hydrostatic_equilibrium(
            pressures=pressures,
            temperatures=temperatures,
            mean_molar_masses=mean_molar_masses,
            gravity=gravity,
            reference_pressure=reference_pressure,
            planet_radius=planet_radius,
            variable_gravity=variable_gravity
        )

        radius = np.array(radius, dtype='d', order='F')
        neg_rad = radius < 0.
        radius[neg_rad] = radius[~neg_rad][0]

        # Calculate the density
        # TODO: replace values here with nc.amu and nc.kB.
        # Currently it is kept at the values of the Fortran implementation, such that
        # unit tests are still being passed.
        #                           nc.amu        # nc.kB  # the Fortran values are different from the nc values
        rho = (pressures * 1e6  # bar to cgs
               * mean_molar_masses * 1.66053892e-24 / 1.3806488e-16 / temperatures)
        # Bring in right shape for matrix operations later.
        rho = rho.reshape(1, 1, 1, struc_len)
        rho = np.array(rho, dtype='d', order='F')

        # Bring continuum scattering opacities in right shape for matrix operations later.
        # Reminder: when calling this function, continuum absorption opacities have already
        # been added to line_struc_kappas.
        continuum_opa_scat_reshaped = continuum_opacities_scattering.reshape((1, freq_len, 1, struc_len))

        # Calculate the inverse mean free paths
        if line_by_line:
            alpha_t2 = opacities[:, :, :1, :] * rho
            alpha_t2 += continuum_opa_scat_reshaped * rho
        else:
            alpha_t2 = opacities * rho
            alpha_t2[:, :, :1, :] += continuum_opa_scat_reshaped * rho

        # Calculate average mean free path between neighboring layers for later integration
        # Factor 1/2 is omitted because it cancels with effective planet area integration below.
        alpha_t2[:, :, :, 1:] = alpha_t2[:, :, :, :-1] + alpha_t2[:, :, :, 1:]

        # Prepare matrix for delta path lengths during optical depth integration
        diff_s = np.zeros((struc_len, struc_len), order='F')

        # Calculate matrix of delta path lengths
        r_ik = radius.reshape(1, struc_len) ** 2. - radius.reshape(struc_len, 1) ** 2.
        r_ik[r_ik < 0.] = 0.
        r_ik = np.sqrt(r_ik)
        diff_s[1:, 1:] = - r_ik[1:, 1:] + r_ik[1:, :-1]

        # Calculate optical depths
        t_graze = np.einsum('ijkl,ml', alpha_t2, diff_s, optimize=True)
        # Calculate transmissions
        t_graze = np.exp(-t_graze)
        # Integrate over correlated-k's g-coordinate (self.weights_gauss == np.array([1.]) for lbl mode)
        t_graze = np.einsum('ijkl,i', t_graze, weights_gauss, optimize=True)

        # Multiply transmissions of all absorber species in c-k mode (this will have no effect in lbl mode)
        t_graze = np.swapaxes(t_graze, 0, 1)
        t_graze = np.swapaxes(t_graze, 1, 2)
        t_graze = np.prod(t_graze, axis=0)

        # Prepare planet area integration: this is the transparency.
        t_graze = 1. - t_graze

        # Annulus radius increments
        diffr = -np.diff(radius).reshape(struc_len - 1, 1)
        radreshape = radius.reshape(struc_len, 1)

        # Integrate effective area, omit 2 pi omitted:
        # 2 cancels with 1/2 of average inverse mean free path above.
        # pi cancels when calculating the radius from the area below.
        transm = np.sum(
            diffr * (t_graze[1:, :] * radreshape[1:, :] + t_graze[:-1, :] * radreshape[:-1, :]),
            axis=0
        )
        # Transform area to transmission radius.
        transm = np.sqrt(transm + radius[-1] ** 2.)

        return transm, radius

    def load_line_opacities(self, start_index, path_input_data,
                            default_temperature_grid_size=13, default_pressure_grid_size=10):
        """Read the line opacities for spectral calculation.
        The default pressure-temperature grid is a log-uniform (10, 13) grid.

        Args:
            start_index:
            path_input_data:
            default_temperature_grid_size:
            default_pressure_grid_size:

        Returns:

        """
        # TODO currently all the pressure-temperature grid is loaded, it could be more memory efficient to provide a T an p range at init and only load the relevant parts of the grid # noqa: E501
        # Get the default pressure-temperature grid
        opacities_temperature_profile_grid = np.genfromtxt(
            os.path.join(path_input_data, 'opa_input_files', 'opa_PT_grid.dat')
        )

        opacities_temperature_profile_grid = np.flip(opacities_temperature_profile_grid, axis=1)
        opacities_temperature_profile_grid[:, 1] *= 1e6  # bars to cgs
        opacities_temperature_profile_grid = np.array(opacities_temperature_profile_grid, dtype='d', order='F')

        custom_line_paths = {}

        # Read opacities grid
        if len(self.line_species) > 0:
            for species in self.line_species:
                file_path_hdf5 = None

                if self.line_opacity_mode == 'c-k':
                    path_opacities = os.path.join(path_input_data, 'opacities', 'lines', 'corr_k', species)
                    file_path_hdf5 = glob.glob(path_opacities + '/*.h5')

                    if len(file_path_hdf5) == 0:
                        file_path_hdf5 = None
                    else:
                        file_path_hdf5 = file_path_hdf5[0]
                elif self.line_opacity_mode == 'lbl':
                    path_opacities = os.path.join(path_input_data, 'opacities', 'lines', 'line_by_line')
                    file_path_hdf5 = os.path.join(path_opacities, species + '.otable.petitRADTRANS.h5')

                if file_path_hdf5 is not None:
                    if not os.path.isfile(file_path_hdf5):
                        print(f"HDF5 opacity file '{file_path_hdf5}' not found, "
                              f"loading from .dat...")
                        file_path_hdf5 = None
                else:
                    print("HDF5 opacity file not found, loading from .dat...")

                self.line_opacities_temperature_profile_grid[species], \
                    custom_line_paths[species], \
                    self.line_opacities_temperature_grid_size[species], \
                    self.line_opacities_pressure_grid_size[species], \
                    self.has_custom_line_opacities_tp_grid[species] \
                    = self.load_line_opacities_pressure_temperature_grid(
                    file_path_hdf5=file_path_hdf5,
                    path_input_data=path_input_data,
                    mode=self.line_opacity_mode,
                    species=species,
                    opacities_temperature_profile_grid=opacities_temperature_profile_grid,
                    default_temperature_grid_size=default_temperature_grid_size,
                    default_pressure_grid_size=default_pressure_grid_size
                )

                # Read the opacities
                if file_path_hdf5 is None:
                    self.line_opacities_grid[species] = self.load_dat_line_opacities(
                        has_custom_line_opacities_temperature_profile_grid=self.has_custom_line_opacities_tp_grid[
                            species
                        ],
                        opacities_temperature_profile_grid=opacities_temperature_profile_grid,
                        line_opacities_temperature_profile_grid=self.line_opacities_temperature_profile_grid[
                            species
                        ].shape[0],
                        custom_line_paths=custom_line_paths[species],
                        mode=self.line_opacity_mode,
                        path_input_data=self.path_input_data,
                        species=species,
                        lbl_opacity_sampling=self.lbl_opacity_sampling,
                        freq=self.frequencies,
                        freq_len=self.frequencies.size,
                        g_len=self.g_size,
                        start_index=start_index
                    )
                else:
                    print(f" Loading line opacities of species '{species}'...")

                    if self.line_opacity_mode == 'c-k':
                        self.line_opacities_grid[species] = self.load_hdf5_ktables(
                            file_path_hdf5=file_path_hdf5,
                            freq=self.frequencies,
                            g_len=self.g_size,
                            freq_len=self.frequencies.size,
                            temperature_profile_grid_size=self.line_opacities_temperature_profile_grid[species].shape[0]
                        )
                    elif self.line_opacity_mode == 'lbl':
                        self.line_opacities_grid[species] = self.load_hdf5_line_opacity_table(
                            file_path_hdf5=file_path_hdf5,
                            freq=self.frequencies,
                            lbl_opacity_sampling=self.lbl_opacity_sampling
                        )

                    print(" Done.")

                # Convert into F-ordered array for more efficient processing in the Fortran modules
                self.line_opacities_grid[species] = \
                    np.array(self.line_opacities_grid[species][:, :, 0, :], dtype='d', order='F')

            print('\n')

        # Read in g grid for correlated-k
        if self.line_opacity_mode == 'c-k':
            buffer = np.genfromtxt(
                os.path.join(path_input_data, 'opa_input_files', 'g_comb_grid.dat')
            )
            self.g_gauss = np.array(buffer[:, 0], dtype='d', order='F')
            self.weights_gauss = np.array(buffer[:, 1], dtype='d', order='F')

    def load_cloud_opacities(self, path_input_data):
        # Function to read cloud opacities
        self.cloud_species_mode = []

        hdf5_files = []
        opacities_dir = os.path.join(path_input_data, 'opacities', 'continuum', 'clouds')
        use_hdf5_files = True

        for i in range(len(self.cloud_species)):
            splitstr = self.cloud_species[i].split('_')
            self.cloud_species_mode.append(splitstr[1])
            self.cloud_species[i] = splitstr[0]

            hdf5_file = os.path.join(
                opacities_dir,
                self.cloud_species[i] + '_' + self.cloud_species_mode[i] + '.cotable.petitRADTRANS.h5'
            )

            if not os.path.isfile(hdf5_file):
                '''
                The function that read .dat files is fed with all cloud species at once, so if one HDF5 file is missing,
                it is easier to just read everything from .dat files. It is unlikely that a user will have a mix of
                files HDF5 and .dat files anyway.
                '''
                print(f"HDF5 cloud opacity file '{hdf5_file}' not found, loading all cloud data from .dat...")
                use_hdf5_files = False
                break

            hdf5_files.append(hdf5_file)

        if use_hdf5_files:
            rho_cloud_particles = np.zeros(len(hdf5_files))
            cloud_specs_abs_opa = None
            cloud_specs_scat_opa = None
            cloud_aniso = None
            cloud_lambdas = None
            cloud_rad_bins = None
            cloud_radii = None

            for i, hdf5_file in enumerate(hdf5_files):
                if self.cloud_species_mode[i][0] == 'c':
                    particles_internal_structure = 'crystalline'
                elif self.cloud_species_mode[i][0] == 'a':
                    particles_internal_structure = 'amorphous'
                else:
                    raise ValueError(f"Particle internal structure code must be 'a' or 'c', "
                                     f"but was '{self.cloud_species_mode[i][0]}'")

                if self.cloud_species_mode[i][1] == 'm':
                    scattering_method = 'Mie (spherical shape)'
                elif self.cloud_species_mode[i][1] == 'd':
                    scattering_method = 'DHS (irregular shape)'
                else:
                    raise ValueError(f"Particle shape code must be 'm' or 'd', "
                                     f"but was '{self.cloud_species_mode[i][1]}'")

                print(f" Loading opacities of cloud species '{self.cloud_species[i]}' "
                      f"({particles_internal_structure}, using {scattering_method} scattering)...")

                with h5py.File(hdf5_file, 'r') as f:
                    if i == 0:
                        # Initialize cloud arrays
                        cloud_lambdas = 1 / f['wavenumbers'][:]  # cm-1 to cm
                        cloud_lambdas = cloud_lambdas[::-1]  # correct ordering
                        cloud_rad_bins = f['particle_radius_bins'][:]
                        cloud_radii = f['particles_radii'][:]

                        cloud_specs_abs_opa = np.zeros((cloud_radii.size, cloud_lambdas.size, len(hdf5_files)))
                        cloud_specs_scat_opa = np.zeros((cloud_radii.size, cloud_lambdas.size, len(hdf5_files)))
                        cloud_aniso = np.zeros((cloud_radii.size, cloud_lambdas.size, len(hdf5_files)))

                    rho_cloud_particles[i] = f['particles_density'][()]
                    cloud_specs_abs_opa[:, :, i] = f['absorption_opacities'][:]
                    cloud_specs_scat_opa[:, :, i] = f['scattering_opacities'][:]
                    cloud_aniso[:, :, i] = f['asymmetry_parameters'][:]

            # Flip wavelengths/wavenumbers axis to match wavelengths ordering
            cloud_specs_abs_opa = cloud_specs_abs_opa[:, ::-1, :]
            cloud_specs_scat_opa = cloud_specs_scat_opa[:, ::-1, :]
            cloud_aniso = cloud_aniso[:, ::-1, :]
        else:
            # Prepare single strings delimited by ':' which are then
            # put into F routines
            tot_str_names = ''

            for cloud_species in self.cloud_species:
                tot_str_names = tot_str_names + cloud_species + ':'

            tot_str_modes = ''

            for cloud_species_mode in self.cloud_species_mode:
                tot_str_modes = tot_str_modes + cloud_species_mode + ':'

            n_cloud_wavelength_bins = int(len(np.genfromtxt(
                os.path.join(
                    path_input_data, 'opacities', 'continuum', 'clouds', 'MgSiO3_c', 'amorphous', 'mie', 'opa_0001.dat'
                )
            )[:, 0]))

            # Actual loading of opacities
            rho_cloud_particles, cloud_specs_abs_opa, cloud_specs_scat_opa, \
                cloud_aniso, cloud_lambdas, cloud_rad_bins, cloud_radii \
                = fi.read_in_cloud_opacities(
                    path_input_data, tot_str_names, tot_str_modes, len(self.cloud_species), n_cloud_wavelength_bins
                )

        cloud_specs_abs_opa[cloud_specs_abs_opa < 0.] = 0.
        cloud_specs_scat_opa[cloud_specs_scat_opa < 0.] = 0.

        self.cloud_particles_densities = np.array(rho_cloud_particles, dtype='d', order='F')
        self.cloud_species_absorption_opacities = np.array(cloud_specs_abs_opa, dtype='d', order='F')
        self.cloud_species_scattering_opacities = np.array(cloud_specs_scat_opa, dtype='d', order='F')
        self.cloud_particles_asymmetry_parameters = np.array(cloud_aniso, dtype='d', order='F')
        self.cloud_wavelengths = np.array(cloud_lambdas, dtype='d', order='F')
        self.cloud_particle_radius_bins = np.array(cloud_rad_bins, dtype='d', order='F')
        self.cloud_particles_radii = np.array(cloud_radii, dtype='d', order='F')

    def write_out_rebin(self, resolution, path='', species=None, masses=None):
        # TODO should be removed as it is only used in one retrieval.util function
        import exo_k

        if species is None:
            species = []

        # Define own wavenumber grid, make sure that log spacing is constant everywhere
        n_spectral_points = int(
            resolution * np.log(self.wavelengths_boundaries[1] / self.wavelengths_boundaries[0]) + 1
        )
        wavenumber_grid = np.logspace(np.log10(1 / self.wavelengths_boundaries[1] / 1e-4),
                                      np.log10(1. / self.wavelengths_boundaries[0] / 1e-4),
                                      n_spectral_points)
        dt = h5py.string_dtype(encoding='utf-8')
        # Do the rebinning, loop through species
        for spec in species:
            print('Rebinning species ' + spec + '...')

            # Create hdf5 file that Exo-k can read...
            f = h5py.File('temp.h5', 'w')

            try:
                f.create_dataset('DOI', (1,), data="--", dtype=dt)
            except ValueError:  # TODO check if ValueError is expected here
                f.create_dataset('DOI', data=['--'])

            f.create_dataset('bin_centers', data=self.frequencies[::-1] / nc.c)
            f.create_dataset('bin_edges', data=self.frequencies_bin_edges[::-1] / nc.c)
            ret_opa_table = copy.copy(self.line_opacities_grid[spec])

            # Mass to go from opacities to cross-sections
            ret_opa_table = ret_opa_table * nc.amu * masses[spec.split('_')[0]]

            # Do the opposite of what I do when loading in Katy's ExoMol tables
            # To get opacities into the right format
            ret_opa_table = ret_opa_table[:, ::-1, :]
            ret_opa_table = np.swapaxes(ret_opa_table, 2, 0)
            ret_opa_table = ret_opa_table.reshape((
                self.line_opacities_temperature_grid_size[spec],
                self.line_opacities_pressure_grid_size[spec],
                self.frequencies.size,
                len(self.weights_gauss)
            ))
            ret_opa_table = np.swapaxes(ret_opa_table, 1, 0)
            ret_opa_table[ret_opa_table < 1e-60] = 1e-60
            f.create_dataset('kcoeff', data=ret_opa_table)
            f['kcoeff'].attrs.create('units', 'cm^2/molecule')

            # Add the rest of the stuff that is needed.
            try:
                f.create_dataset('method', (1,), data="petit_samples", dtype=dt)
            except ValueError:  # TODO check if ValueError is expected here
                f.create_dataset('method', data=['petit_samples'])

            f.create_dataset('mol_name', data=spec.split('_')[0], dtype=dt)
            f.create_dataset('mol_mass', data=[masses[spec.split('_')[0]]])
            f.create_dataset('ngauss', data=len(self.weights_gauss))
            f.create_dataset('p', data=self.line_opacities_temperature_profile_grid[spec][
                                       :self.line_opacities_pressure_grid_size[spec], 1] / 1e6)
            f['p'].attrs.create('units', 'bar')
            f.create_dataset('samples', data=self.g_gauss)
            f.create_dataset('t', data=self.line_opacities_temperature_profile_grid[spec][
                                       ::self.line_opacities_pressure_grid_size[spec], 0])
            f.create_dataset('weights', data=self.weights_gauss)
            f.create_dataset('wlrange', data=[np.min(nc.c / self.frequencies_bin_edges / 1e-4),
                                              np.max(nc.c / self.frequencies_bin_edges / 1e-4)])
            f.create_dataset('wnrange', data=[np.min(self.frequencies_bin_edges / nc.c),
                                              np.max(self.frequencies_bin_edges / nc.c)])
            f.close()
            ###############################################
            # Use Exo-k to rebin to low-res, save to desired folder
            ###############################################
            tab = exo_k.Ktable(filename='temp.h5')
            tab.bin_down(wavenumber_grid)

            if path[-1] == '/':
                path = path[:-1]

            os.makedirs(path + '/' + spec + '_R_' + str(int(resolution)), exist_ok=True)
            tab.write_hdf5(path + '/' + spec + '_R_' + str(int(resolution)) + '/' + spec + '_R_' + str(
                int(resolution)) + '.h5')
            os.system('rm temp.h5')

    @staticmethod
    def calculate_h_minus_opacities(lambda_angstroem, border_lambda_angstroem,
                                    temp, press, mmw, abundances):
        """Calc the H- opacity."""

        ret_val = np.array(np.zeros(len(lambda_angstroem) * len(press)).reshape(
            len(lambda_angstroem),
            len(press)), dtype='d', order='F')

        # Calc. electron number fraction
        # e- mass in amu:
        m_e = 5.485799e-4
        n_e = mmw / m_e * abundances['e-']

        # Calc. e- partial pressure
        p_e = press * n_e

        kappa_hminus_bf = Radtrans.calculate_h_minus_bound_free_xsec(border_lambda_angstroem) / nc.amu

        for i_struct in range(len(n_e)):
            kappa_hminus_ff = Radtrans.calculate_h_minus_free_free_xsec(
                lambda_angstroem,
                temp[i_struct],
                p_e[i_struct]
            ) / nc.amu * abundances['H'][i_struct]

            ret_val[:, i_struct] = kappa_hminus_bf * abundances['H-'][i_struct] + kappa_hminus_ff

        return ret_val
