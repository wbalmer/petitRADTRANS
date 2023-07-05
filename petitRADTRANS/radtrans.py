import copy
import glob
import os
import sys
import warnings

import h5py
import numpy as np
from scipy.interpolate import interp1d

from petitRADTRANS.config import petitradtrans_config
from petitRADTRANS import nat_cst as nc
from petitRADTRANS import phoenix
from petitRADTRANS.fort_input import fort_input as fi
from petitRADTRANS.fort_rebin import fort_rebin as fr
from petitRADTRANS.fort_spec import fort_spec as fs


class Radtrans:
    r""" Class defining objects for carrying out spectral calculations for a
    given set of opacities

    Args:
        line_species (Optional):
            list of strings, denoting which line absorber species to include.
        rayleigh_species (Optional):
            list of strings, denoting which Rayleigh scattering species to
            include.
        cloud_species (Optional):
            list of strings, denoting which cloud opacity species to include.
        continuum_opacities (Optional):
            list of strings, denoting which continuum absorber species to
            include.
        wlen_bords_micron (Optional):
            list containing left and right border of wavelength region to be
            considered, in micron. If nothing else is specified, it will be
            equal to ``[0.05, 300]``, hence using the full petitRADTRANS
            wavelength range (0.11 to 250 microns for ``'c-k'`` mode, 0.3 to 30
            microns for the ``'lbl'`` mode). The larger the range the longer the
            computation time.
        mode (Optional[string]):
            if equal to ``'c-k'``: use low-resolution mode, at
            :math:`\\lambda/\\Delta \\lambda = 1000`, with the correlated-k
            assumption. if equal to ``'lbl'``: use high-resolution mode, at
            :math:`\\lambda/\\Delta \\lambda = 10^6`, with a line-by-line
            treatment.
        do_scat_emis (Optional[bool]):
            Will be ``False`` by default.
            If ``True`` scattering will be included in the emission spectral
            calculations. Note that this increases the runtime of pRT!
        lbl_opacity_sampling (Optional[int]):
            Will be ``None`` by default. If integer positive value, and if
            ``mode == 'lbl'`` is ``True``, then this will only consider every
            lbl_opacity_sampling-nth point of the high-resolution opacities.
            This may be desired in the case where medium-resolution spectra are
            required with a :math:`\\lambda/\\Delta \\lambda > 1000`, but much smaller than
            :math:`10^6`, which is the resolution of the ``lbl`` mode. In this case it
            may make sense to carry out the calculations with lbl_opacity_sampling = 10,
            for example, and then rebinning to the final desired resolution:
            this may save time! The user should verify whether this leads to
            solutions which are identical to the rebinned results of the fiducial
            :math:`10^6` resolution. If not, this parameter must not be used.
    """

    def __init__(
            self,
            line_species=None,
            rayleigh_species=None,
            cloud_species=None,
            continuum_opacities=None,
            wlen_bords_micron=None,
            mode='c-k',
            test_ck_shuffle_comp=False,
            do_scat_emis=False,
            lbl_opacity_sampling=None,
            pressures=None,
            temperatures=None,  # TODO temperatures not redefined in functions
            stellar_intensity=None,
            geometry='dayside_ave',
            mu_star=1.,
            anisotropic_cloud_scattering='auto',
            hack_cloud_photospheric_tau=None,
            path_input_data=petitradtrans_config['Paths']['prt_input_data_path']
    ):
        """

        Args:
            line_species:
            rayleigh_species:
            cloud_species:
            continuum_opacities:
            wlen_bords_micron:
            mode:
            test_ck_shuffle_comp:
            do_scat_emis:
            lbl_opacity_sampling:
        """
        # TODO add wavelengths generator
        if line_species is None:
            line_species = []

        if rayleigh_species is None:
            rayleigh_species = []

        if cloud_species is None:
            cloud_species = []

        if continuum_opacities is None:
            continuum_opacities = []  # TODO add continuum_opacities as attribute

        if wlen_bords_micron is None:
            wlen_bords_micron = np.array([0.05, 300.])  # um

        if pressures is None:
            warnings.warn("pressure was not set, initializing one layer at 1 bar")
            pressures = np.array([1.0])  # bar

        if temperatures is None:
            temperatures = 300.0 * np.ones_like(pressures)  # K
        elif np.size(temperatures) != np.size(pressures):
            print(f"The size of the temperature array ({np.size(temperatures)}) "
                  f"must be equal to the size of the pressure array ({np.size(pressures)}), "
                  f"log-interpolating temperatures on the pressure array...")
            pressure_tmp = np.logspace(np.log10(np.min(pressures)), np.log10(np.max(pressures)), np.size(temperatures))
            temperatures = np.interp(pressures, pressure_tmp, temperatures)

        self.path_input_data = path_input_data

        self.wlen_bords_micron = wlen_bords_micron

        # ADD TO SOURCE AND COMMENT PROPERLY LATER!
        self.test_ck_shuffle_comp = test_ck_shuffle_comp
        self.do_scat_emis = do_scat_emis

        # Stellar intensity (scaled by distance)
        self.stellar_intensity = stellar_intensity

        # For feautrier scattering of direct stellar light
        self.geometry = geometry
        self.mu_star = mu_star

        # Line-by-line or corr-k
        self.mode = mode
        self.lbl_opacity_sampling = lbl_opacity_sampling

        # Line opacity species to be considered
        self.line_species = line_species

        # Rayleigh scattering species to be considered
        self.rayleigh_species = rayleigh_species

        # Cloud species to be considered
        self.cloud_species = cloud_species

        # Read in the angle (mu) grid for the emission spectral calculations.
        buffer = np.genfromtxt(os.path.join(self.path_input_data, 'opa_input_files', 'mu_points.dat'))
        self.mu, self.w_gauss_mu = buffer[:, 0], buffer[:, 1]

        self.Pcloud = None
        self.haze_factor = None
        self.gray_opacity = None
        self.scat = False
        self.cloud_scaling_factor = None
        self.scaling_physicality = None

        # Read in frequency grid
        # Any opacities there at all?
        if len(line_species) + len(rayleigh_species) + len(cloud_species) + len(continuum_opacities) > 0:
            self.absorbers_present = True
        else:
            self.absorbers_present = False

        # Line species present? If yes: define wavelength array
        if len(line_species) > 0:  # TODO init Radtrans even if there is no opacity
            self.line_absorbers_present = True
        else:
            self.line_absorbers_present = False

        # Initialize line parameters
        if self.line_absorbers_present:
            self.freq, self.border_freqs, self.lambda_angstroem, self.border_lambda_angstroem, \
                self.freq_len, self.g_len, arr_min = self._init_line_opacities_parameters()
        else:
            self.freq_len = None
            self.g_len = None
            self.freq = None
            self.border_freqs = None
            self.border_lambda_angstroem = None
            arr_min = None

        self.skip_RT_step = False

        #  Default surface albedo and emissivity -- will be used only if the surface scattering is turned on.
        self.reflectance = 0 * np.ones_like(self.freq)
        self.emissivity = 1 * np.ones_like(self.freq)

        # Initialize pressure-dependent parameters
        self.press, self.continuum_opa, self.continuum_opa_scat, self.continuum_opa_scat_emis, \
            self.contr_em, self.contr_tr, self.radius_hse, self.mmw, \
            self.line_struc_kappas, self.line_struc_kappas_comb, \
            self.line_abundances, self.cloud_mass_fracs, self.r_g = \
            self._init_pressure_dependent_parameters(pressures=pressures)

        self.temp = temperatures
        self.gravity = None

        # Some necessary definitions, also prepare arrays for fluxes, transmission radius...
        self.flux = np.array(np.zeros(self.freq_len), dtype='d', order='F')
        self.transm_rad = np.array(np.zeros(self.freq_len), dtype='d', order='F')

        # Initialize cloud parameters
        self.kappa_zero = None
        self.gamma_scat = None
        self.fsed = None
        self.cloud_wlen = None

        # Initialize derived variables  TODO check if some of these can be made private variables instead of attributes
        self.cloud_total_opa_retrieval_check = None
        self.photon_destruction_prob = None
        self.kappa_rosseland = None
        self.tau_cloud = None
        self.tau_rosse = None

        # Initialize special variables
        self.hack_cloud_total_scat_aniso = None
        self.hack_cloud_total_abs = None
        self.hack_cloud_photospheric_tau = hack_cloud_photospheric_tau
        self.phot_radius = None
        self.anisotropic_cloud_scattering = anisotropic_cloud_scattering

        # Initialize line opacities variables
        self.has_custom_line_opacities_temperature_profile_grid = {}
        self.line_opacities_temperature_profile_grid = {}
        self.line_opacities_temperature_grid_size = {}
        self.line_opacities_pressure_grid_size = {}
        self.line_opacities_grid = {}
        self.g_gauss = np.array(np.ones(1), dtype='d', order='F')
        self.w_gauss = np.array(np.ones(1), dtype='d', order='F')

        # Initialize cloud opacities variables
        self.cloud_species_mode = None
        self.rho_cloud_particles = None
        self.cloud_specs_abs_opa = None
        self.cloud_specs_scat_opa = None
        self.cloud_aniso = None
        self.cloud_lambdas = None
        self.cloud_rad_bins = None
        self.cloud_radii = None

        # TODO instead of reading lines here, do it in a separate function
        # START Reading in opacities
        # Read in line opacities
        # Inherited from ReadOpacities in _read_opacities.py
        self.read_line_opacities(arr_min, self.path_input_data)

        # Read continuum opacities
        # Clouds
        if len(self.cloud_species) > 0:
            # Inherited from ReadOpacities in _read_opacities.py
            self.read_cloud_opas(self.path_input_data)

        # CIA
        self.CIA_species = {}
        self.Hminus = False

        if len(continuum_opacities) > 0:
            for c in continuum_opacities:
                mol = c.split('-')

                if not c == 'H-':
                    print(f"  Read CIA opacities for {c}...")
                    cia_directory = os.path.join(self.path_input_data, 'opacities', 'continuum', 'CIA', c)

                    if os.path.isdir(cia_directory) is False:
                        raise ValueError(f"CIA directory '{cia_directory}' do not exists.")
                    else:
                        weight = 1

                        for m in mol:
                            weight = weight * nc.molecular_weight[m]

                        cia_lambda, cia_temp, cia_alpha_grid, \
                            cia_temp_dims, cia_lambda_dims = fi.cia_read(c, self.path_input_data)
                        cia_alpha_grid = np.array(cia_alpha_grid, dtype='d', order='F')
                        cia_temp = cia_temp[:cia_temp_dims]
                        cia_lambda = cia_lambda[:cia_lambda_dims]
                        cia_alpha_grid = cia_alpha_grid[:cia_lambda_dims, :cia_temp_dims]
                        species = {
                            'id': c,
                            'molecules': mol,
                            'weight': weight,
                            'lambda': cia_lambda,
                            'temperature': cia_temp,
                            'alpha': cia_alpha_grid
                        }
                        self.CIA_species[c] = species
                else:
                    self.Hminus = True
            print('Done.\n')

    def _clouds_have_effect(self, mass_mixing_ratios):
        """
        Check if the clouds have any effect, i.e. if the MMR is greater than 0.

        Args:
            mass_mixing_ratios: atmospheric mass mixing ratios

        Returns:

        """
        add_cloud_opacity = False

        if int(len(self.cloud_species)) > 0:
            for i_spec in range(len(self.cloud_species)):
                if np.any(mass_mixing_ratios[self.cloud_species[i_spec]] > 0):
                    add_cloud_opacity = True  # add cloud opacity only if there are actually clouds

                    break

        return add_cloud_opacity

    def _init_line_opacities_parameters(self):
        if self.mode == 'c-k':
            if self.do_scat_emis and not self.test_ck_shuffle_comp:
                print("Emission scattering is enabled: enforcing test_ck_shuffle_comp = True")

                self.test_ck_shuffle_comp = True

            # For correlated-k
            # Get dimensions of molecular opacity arrays for a given P-T point, they define the resolution.
            # Use the first entry of self.line_species for this, if given.
            path_opa = os.path.join(self.path_input_data, 'opacities', 'lines', 'corr_k', self.line_species[0])
            hdf5_path = glob.glob(path_opa + '/*.h5')  # check if first species is hdf5

            if hdf5_path:
                f = h5py.File(hdf5_path[0], 'r')
                g_len = len(f['samples'][:])
                border_freqs = nc.c * f['bin_edges'][:][::-1]
            else:  # if no hdf5 line absorbers are given use the classical pRT format.
                # In the long run: move to hdf5 fully?
                # But: people calculate their own k-tables with my code sometimes now.
                freq_len, g_len = fi.get_freq_len(self.path_input_data, self.line_species[0])

                # Read in the frequency range of the opacity data
                freq, border_freqs = fi.get_freq(self.path_input_data, self.line_species[0], freq_len)

            # Extend the wavelength range if user requests larger
            # range than what first line opa species contains
            wlen = nc.c / border_freqs * 1e4

            if wlen[-1] < self.wlen_bords_micron[1]:
                delta_log_lambda = np.diff(np.log10(wlen))[-1]
                add_high = 1e1 ** np.arange(np.log10(wlen[-1]),
                                            np.log10(self.wlen_bords_micron[-1]) + delta_log_lambda,
                                            delta_log_lambda)[1:]
                wlen = np.concatenate((wlen, add_high))

            if wlen[0] > self.wlen_bords_micron[0]:
                delta_log_lambda = np.diff(np.log10(wlen))[0]
                add_low = 1e1 ** (-np.arange(-np.log10(wlen[0]),
                                             -np.log10(self.wlen_bords_micron[0]) + delta_log_lambda,
                                             delta_log_lambda)[1:][::-1])
                wlen = np.concatenate((add_low, wlen))

            border_freqs = nc.c / (wlen * 1e-4)
            freq = (border_freqs[1:] + border_freqs[:-1]) / 2.

            # Cut the wavelength range if user requests smaller
            # range than what first line opa species contains
            index = (nc.c / freq > self.wlen_bords_micron[0] * 1e-4) & \
                    (nc.c / freq < self.wlen_bords_micron[1] * 1e-4)

            # Use cp_freq to make a bool array of the same length as border freqs.
            cp_freq = np.zeros(len(freq) + 1)

            # Below the bool array, initialize with zero.
            border_ind = cp_freq > 1.

            # Copy indices of frequency midpoint array
            border_ind[:-1] = index

            # Set all values to the right of the old boundary to True
            border_ind[np.cumsum(border_ind) == len(freq[index])] = True

            # Set all values two positions to the right of the old bondary to False
            border_ind[np.cumsum(border_ind) > len(freq[index]) + 1] = False

            # So we have a bool array longer by one element than index now,
            # with one additional position True to the right of the rightmost old one.
            # Should give the correct border frequency indices.
            # Tested this below
            border_freqs = np.array(border_freqs[border_ind], dtype='d', order='F')
            freq = np.array(freq[index], dtype='d', order='F')
            freq_len = len(freq)

            arr_min = -1
        elif self.mode == 'lbl':
            # For high-res line-by-line radiative transfer
            path_length = os.path.join(
                self.path_input_data, 'opacities', 'lines', 'line_by_line', self.line_species[0], 'wlen.dat'
            )
            # Get dimensions of opacity arrays for a given P-T point
            # arr_min, arr_max denote where in the large opacity files
            # the required wavelength range sits.
            freq_len, arr_min, arr_max = fi.get_arr_len_array_bords(
                self.wlen_bords_micron[0] * 1e-4,
                self.wlen_bords_micron[1] * 1e-4,
                path_length
            )

            g_len = 1

            # Read in the frequency range of the opacity data
            if self.lbl_opacity_sampling is not None:
                freq_len += self.lbl_opacity_sampling - 1  # ensure that downsampled upper bound >= requested upp. bound

            wlen = fi.read_wlen(arr_min, freq_len, path_length)
            freq = nc.c / wlen

            # Down-sample frequency grid in lbl mode if requested
            if self.lbl_opacity_sampling is not None:
                freq = freq[::self.lbl_opacity_sampling]
                freq_len = len(freq)

            border_freqs = np.array(nc.c / self.calc_borders(nc.c / freq), dtype='d', order='F')
        else:
            raise ValueError(f"invalid mode value '{self.mode}'; should be 'c-k' or 'lbl'")

        lambda_angstroem = np.array(nc.c / freq * 1e8, dtype='d', order='F')

        if self.mode == 'c-k':
            border_lambda_angstroem = nc.c / border_freqs * 1e8
        elif self.mode == 'lbl':
            border_lambda_angstroem = np.array(self.calc_borders(lambda_angstroem))
        else:
            raise ValueError(f"invalid mode value '{self.mode}'; should be 'c-k' or 'lbl'")

        return freq, border_freqs, lambda_angstroem, border_lambda_angstroem, freq_len, g_len, arr_min

    def _init_pressure_dependent_parameters(self, pressures):
        """ Setup opacity arrays at atmospheric structure dimensions,
        and set the atmospheric pressure array.

        Args:
            pressures:
                the atmospheric pressure (1-d numpy array, sorted in increasing
                order), in units of bar. Will be converted to cgs internally.
        """
        press = pressures * 1e6  # bar to cgs
        p_len = pressures.shape[0]
        continuum_opa = np.zeros((self.freq_len, p_len), dtype='d', order='F')
        continuum_opa_scat = np.zeros((self.freq_len, p_len), dtype='d', order='F')
        continuum_opa_scat_emis = None
        contr_em = None
        contr_tr = None
        radius_hse = np.zeros(p_len, dtype='d', order='F')

        mmw = np.zeros(p_len)

        if len(self.line_species) > 0:
            line_struc_kappas = np.zeros(
                (self.g_len, self.freq_len, len(self.line_species), p_len), dtype='d', order='F'
            )

            if self.mode == 'c-k':
                line_struc_kappas_comb = np.zeros((self.g_len, self.freq_len, p_len), dtype='d', order='F')
            else:
                line_struc_kappas_comb = None

            line_abundances = np.zeros((p_len, len(self.line_species)), dtype='d', order='F')
        else:
            # If there are no specified line species then we need at
            # least an array to contain the continuum opas
            # I'll (mis)use the line_struc_kappas array for that
            line_struc_kappas = np.zeros((self.g_len, self.freq_len, 1, p_len), dtype='d', order='F')
            line_struc_kappas_comb = None
            line_abundances = np.zeros((p_len, 1), dtype='d', order='F')

        if len(self.cloud_species) > 0:
            cloud_mass_fracs = np.zeros((p_len, len(self.cloud_species)), dtype='d', order='F')
            r_g = np.zeros((p_len, len(self.cloud_species)), dtype='d', order='F')
        else:
            cloud_mass_fracs = None
            r_g = None

        return press, continuum_opa, continuum_opa_scat, continuum_opa_scat_emis, contr_em, contr_tr, radius_hse, mmw, \
            line_struc_kappas, line_struc_kappas_comb, line_abundances, cloud_mass_fracs, r_g

    @staticmethod
    def _sort_opa_pt_grid(path_ptg):
        # Read the Ps and Ts
        p_ts = np.genfromtxt(path_ptg)

        # Read the file names
        with open(path_ptg, 'r') as f:
            lines = f.readlines()

        n_entries = len(lines)

        # Prepare the array to contain the
        # pressures, temperatures, indices in the unsorted list.
        # Also prepare the list of unsorted names
        p_tind = np.ones(n_entries * 3).reshape(n_entries, 3)
        names = []

        # Fill the array and name list
        for i_line in range(n_entries):

            line = lines[i_line]
            lsp = line.split(' ')

            p_tind[i_line, 0], p_tind[i_line, 1], p_tind[i_line, 2] = \
                p_ts[i_line, 0], p_ts[i_line, 1], i_line

            if lsp[-1][-1] == '\n':
                names.append(lsp[-1][:-1])
            else:
                names.append(lsp[-1])

        # Sort the array by temperature
        tsortind = np.argsort(p_tind[:, 1])
        p_tind = p_tind[tsortind, :]

        # Sort the array entries with constant
        # temperatures by pressure
        diff_ps = 0
        t_start = p_tind[0, 1]

        for i in range(n_entries):
            if np.abs(p_tind[i, 1] - t_start) > 1e-10:
                break
            diff_ps = diff_ps + 1

        diff_ts = int(n_entries / diff_ps)
        for i_dT in range(diff_ts):
            subsort = p_tind[i_dT * diff_ps:(i_dT + 1) * diff_ps, :]
            psortind = np.argsort(subsort[:, 0])
            subsort = subsort[psortind, :]
            p_tind[i_dT * diff_ps:(i_dT + 1) * diff_ps, :] = subsort

        names_sorted = []
        for i_line in range(n_entries):
            names_sorted.append(names[int(p_tind[i_line, 2] + 0.01)])

        # Convert from bars to cgs
        p_tind[:, 0] = p_tind[:, 0] * 1e6

        return [p_tind[:, :-1][:, ::-1], names_sorted, diff_ts, diff_ps]

    def add_rayleigh(self, abundances):
        # Add Rayleigh scattering cross-sections
        for spec in self.rayleigh_species:
            haze_multiply = 1.

            if self.haze_factor is not None:
                haze_multiply = self.haze_factor

            add_term = haze_multiply * fs.add_rayleigh(
                spec,
                abundances[spec],
                self.lambda_angstroem,
                self.mmw,
                self.temp,
                self.press
            )

            self.continuum_opa_scat += add_term

    @staticmethod
    def calc_borders(x):
        # Return bin borders for midpoints.
        xn = [x[0] - (x[1] - x[0]) / 2.]

        for i in range(int(len(x)) - 1):
            xn.append(x[i] + (x[i + 1] - x[i]) / 2.)

        xn.append(x[int(len(x)) - 1] + (x[int(len(x)) - 1] - x[int(len(x)) - 2]) / 2.)

        return np.array(xn)

    def calc_cloud_opacity(self, abundances, mmw, gravity, sigma_lnorm,
                           fsed=None, kzz=None,
                           radius=None, add_cloud_scat_as_abs=False,
                           dist="lognormal", a_hans=None, b_hans=None):
        # Function to calculate cloud opacities
        # for defined atmospheric structure.
        rho = self.press / nc.kB / self.temp * mmw * nc.amu

        if "hansen" in dist.lower():
            if isinstance(b_hans, np.ndarray):
                if not b_hans.shape == (self.press.shape[0], len(self.cloud_species)):
                    raise ValueError(
                        "b_hans must be a float, a dictionary with arrays for each cloud species, "
                        f"or a numpy array with shape {(self.press.shape[0],len(self.cloud_species))}, "
                        f"but was of shape {np.shape(b_hans)}"
                    )
            elif isinstance(b_hans, dict):
                b_hans = np.array(list(b_hans.values()), dtype='d', order='F').T
            elif isinstance(b_hans, float):
                b_hans = np.array(
                    np.tile(b_hans * np.ones_like(self.press), (len(self.cloud_species), 1)),
                    dtype='d',
                    order='F'
                ).T
            else:
                raise ValueError(f"The Hansen distribution width (b_hans) must be an array, a dict, or a float, "
                                 f"but is of type '{type(b_hans)}' ({b_hans})")

        for i_spec, cloud_name in enumerate(self.cloud_species):
            self.cloud_mass_fracs[:, i_spec] = abundances[cloud_name]

            if radius is not None:
                self.r_g[:, i_spec] = radius[cloud_name]
            elif a_hans is not None:
                self.r_g[:, i_spec] = a_hans[cloud_name]

        if radius is not None or a_hans is not None:
            if dist == "lognormal":
                cloud_abs_opa_tot, cloud_scat_opa_tot, cloud_red_fac_aniso_tot = \
                    py_calc_cloud_opas(rho, self.rho_cloud_particles,
                                       self.cloud_mass_fracs, self.r_g, sigma_lnorm,
                                       self.cloud_rad_bins, self.cloud_radii,
                                       self.cloud_specs_abs_opa,
                                       self.cloud_specs_scat_opa,
                                       self.cloud_aniso)
            else:
                cloud_abs_opa_tot, cloud_scat_opa_tot, cloud_red_fac_aniso_tot = \
                    fs.calc_hansen_opas(
                        rho,
                        self.rho_cloud_particles,
                        self.cloud_mass_fracs,
                        self.r_g,
                        b_hans,
                        self.cloud_rad_bins,
                        self.cloud_radii,
                        self.cloud_specs_abs_opa,
                        self.cloud_specs_scat_opa,
                        self.cloud_aniso
                    )
        else:
            fseds = np.zeros(len(self.cloud_species))
            for i_spec, cloud in enumerate(self.cloud_species):
                if isinstance(fsed, dict):
                    fseds[i_spec] = fsed[cloud.split('_')[0]]
                elif not hasattr(fsed, '__iter__'):
                    fseds[i_spec] = fsed
            if dist == "lognormal":
                self.r_g = fs.get_rg_n(
                    gravity,
                    rho,
                    self.rho_cloud_particles,
                    self.temp,
                    mmw,
                    fseds,
                    sigma_lnorm,
                    kzz
                )

                cloud_abs_opa_tot, cloud_scat_opa_tot, cloud_red_fac_aniso_tot = \
                    py_calc_cloud_opas(
                        rho,
                        self.rho_cloud_particles,
                        self.cloud_mass_fracs,
                        self.r_g,
                        sigma_lnorm,
                        self.cloud_rad_bins,
                        self.cloud_radii,
                        self.cloud_specs_abs_opa,
                        self.cloud_specs_scat_opa,
                        self.cloud_aniso
                    )
            else:
                self.r_g = fs.get_rg_n_hansen(
                    gravity,
                    rho,
                    self.rho_cloud_particles,
                    self.temp,
                    mmw,
                    fseds,
                    b_hans,
                    kzz
                )

                cloud_abs_opa_tot, cloud_scat_opa_tot, cloud_red_fac_aniso_tot = \
                    fs.calc_hansen_opas(
                        rho,
                        self.rho_cloud_particles,
                        self.cloud_mass_fracs,
                        self.r_g,
                        b_hans,
                        self.cloud_rad_bins,
                        self.cloud_radii,
                        self.cloud_specs_abs_opa,
                        self.cloud_specs_scat_opa,
                        self.cloud_aniso
                    )

        # aniso = (1-g)
        cloud_abs, cloud_abs_plus_scat_aniso, aniso, cloud_abs_plus_scat_no_aniso = \
            fs.interp_integ_cloud_opas(
                cloud_abs_opa_tot,
                cloud_scat_opa_tot,
                cloud_red_fac_aniso_tot,
                self.cloud_lambdas,
                self.border_freqs
            )

        if self.anisotropic_cloud_scattering:
            self.continuum_opa_scat += cloud_abs_plus_scat_aniso - cloud_abs
        else:
            if self.do_scat_emis and self.continuum_opa_scat_emis is not None:
                self.continuum_opa_scat_emis = copy.deepcopy(self.continuum_opa_scat)
                self.continuum_opa_scat_emis += cloud_abs_plus_scat_aniso - cloud_abs

            self.continuum_opa_scat += cloud_abs_plus_scat_no_aniso - cloud_abs

        if self.do_scat_emis:
            if self.hack_cloud_photospheric_tau is not None:
                self.hack_cloud_total_scat_aniso = cloud_abs_plus_scat_aniso - cloud_abs
                self.hack_cloud_total_abs = cloud_abs

        if add_cloud_scat_as_abs:
            if add_cloud_scat_as_abs:
                self.continuum_opa += cloud_abs + 0.20 * (cloud_abs_plus_scat_no_aniso - cloud_abs)
            else:
                self.continuum_opa += cloud_abs
        else:
            self.continuum_opa += cloud_abs

        # This included scattering plus absorption
        self.cloud_total_opa_retrieval_check = cloud_abs_plus_scat_aniso

    def calculate_photon_radius(self, r_pl, temp, mmw, gravity):
        try:
            radius_hse = self.calc_radius_hydrostatic_equilibrium(
                temp,
                mmw,
                gravity,
                self.press[-1] * 1e-6,
                r_pl
            )

            rad_press = interp1d(self.press, radius_hse)

            self.phot_radius = np.zeros(self.freq_len)

            if self.mode == 'lbl' or self.test_ck_shuffle_comp:
                total_tau = self.calc_opt_depth(gravity, cloud_wlen=self.cloud_wlen)
                wgauss_reshape = self.w_gauss.reshape(len(self.w_gauss), 1)

                for i_freq in range(self.freq_len):
                    tau_p = np.sum(wgauss_reshape * total_tau[:, i_freq, 0, :], axis=0)
                    press_taup = interp1d(tau_p, self.press)
                    self.phot_radius[i_freq] = rad_press(press_taup(2. / 3.))
        except Exception:  # TODO find what is expected here
            self.phot_radius = -np.ones(self.freq_len)

    def calc_flux(self, temp, abunds, gravity, mmw, r_pl=None, sigma_lnorm=None,
                  fsed=None, kzz=None, radius=None,
                  contribution=False,
                  gray_opacity=None, p_cloud=None,
                  kappa_zero=None,
                  gamma_scat=None,
                  add_cloud_scat_as_abs=False,
                  t_star=None, r_star=None, semimajoraxis=None,
                  geometry='dayside_ave', theta_star=0,
                  hack_cloud_photospheric_tau=None,
                  dist="lognormal", a_hans=None, b_hans=None,
                  stellar_intensity=None,
                  give_absorption_opacity=None,
                  give_scattering_opacity=None,
                  cloud_wlen=None,
                  get_photon_radius=False
                  ):
        """ Method to calculate the atmosphere's emitted flux
        (emission spectrum).

            Args:
                temp:
                    the atmospheric temperature in K, at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                abunds:
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
                sigma_lnorm (Optional[float]):
                    width of the log-normal cloud particle size distribution
                fsed (Optional[float]):
                    cloud settling parameter
                kzz (Optional):
                    the atmospheric eddy diffusion coeffiecient in cgs untis
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
                    Has to be given if kappa_zero is definded, this is the
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
                semimajoraxis (Optional[float]):
                    The distance of the planet from the star. Used to scale
                    the stellar flux when the scattering of the direct light
                    is considered.
                geometry (Optional[string]):
                    if equal to ``'dayside_ave'``: use the dayside average
                    geometry. if equal to ``'planetary_ave'``: use the
                    planetary average geometry. if equal to
                    ``'non-isotropic'``: use the non-isotropic
                    geometry.
                theta_star (Optional[float]):
                    Inclination angle of the direct light with respect to
                    the normal to the atmosphere. Used only in the
                    non-isotropic geometry scenario.
                hack_cloud_photospheric_tau (Optional[float]):
                    Median optical depth (across ``wlen_bords_micron``) of the
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
                    formatted as the kzz argument. Equivilant to radius arg.
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
                cloud_wlen (Optional[Tuple[float, float]]):
                    Tuple with the wavelength range (in micron) that is used
                    for calculating the median optical depth of the clouds at
                    gas-only photosphere and then scaling the cloud optical
                    depth to the value of ``hack_cloud_photospheric_tau``. The
                    range of ``cloud_wlen`` should be encompassed by
                    ``wlen_bords_micron``. The full wavelength range is used
                    when ``cloud_wlen=None``.
        """
        self.hack_cloud_photospheric_tau = hack_cloud_photospheric_tau
        self.Pcloud = p_cloud
        self.kappa_zero = kappa_zero
        self.gamma_scat = gamma_scat
        self.gray_opacity = gray_opacity
        self.geometry = geometry
        self.mu_star = np.cos(theta_star * np.pi / 180.)
        self.fsed = fsed
        self.cloud_wlen = cloud_wlen
        self.gravity = gravity

        if self.cloud_wlen is not None and (
                self.cloud_wlen[0] < 1e-4 * self.lambda_angstroem[0] or
                self.cloud_wlen[1] > 1e-4 * self.lambda_angstroem[-1]):
            raise ValueError('The wavelength range of cloud_wlen should '
                             'lie within the wavelength range of '
                             'self.lambda_angstroem, which is slightly '
                             'smaller than the wavelength range of '
                             'wlen_bords_micron.')

        if self.mu_star <= 0.:
            self.mu_star = 1e-8

        if stellar_intensity is None:
            self.get_star_spectrum(t_star, semimajoraxis, r_star)
        else:
            self.stellar_intensity = stellar_intensity

        self.interpolate_species_opa(temp)

        auto_anisotropic_cloud_scattering = False

        if self.anisotropic_cloud_scattering == 'auto':
            self.anisotropic_cloud_scattering = True
            auto_anisotropic_cloud_scattering = True
        elif not self.anisotropic_cloud_scattering:
            warnings.warn(f"anisotropic cloud scattering is recommended for emission spectra, "
                          f"but 'anisotropic_cloud_scattering' was set to {self.anisotropic_cloud_scattering}; "
                          f"set it to True or 'auto' to disable this warning")

        self.mix_opa_tot(abunds, mmw, gravity, sigma_lnorm, fsed, kzz, radius,
                         add_cloud_scat_as_abs=add_cloud_scat_as_abs,
                         dist=dist, a_hans=a_hans, b_hans=b_hans,
                         give_absorption_opacity=give_absorption_opacity,
                         give_scattering_opacity=give_scattering_opacity)

        if r_pl is not None and get_photon_radius:  # TODO what is the purpose of that?
            self.calculate_photon_radius(r_pl, temp, mmw, gravity)

        if auto_anisotropic_cloud_scattering:
            self.anisotropic_cloud_scattering = 'auto'

        if not self.skip_RT_step:
            self.calc_rt(contribution)

            if self._clouds_have_effect(abunds):
                self.calc_tau_cloud(gravity)

            if (self.mode == 'lbl' or self.test_ck_shuffle_comp) and len(self.line_species) > 1:
                if self.do_scat_emis and self.kappa_rosseland is not None:
                    self.tau_rosse = fs.calc_tau_g_tot_ck(
                        gravity,
                        self.press,
                        self.kappa_rosseland.reshape(1, 1, 1, len(self.press))
                    ).reshape(len(self.press))
        else:
            warnings.warn("Cloud rescaling lead to nan opacities, skipping RT calculation!")

            self.flux = np.ones_like(self.freq) * np.nan
            self.contr_em = None
            self.skip_RT_step = False

    def calc_opt_depth(self, gravity, cloud_wlen=None):
        # Calculate optical depth for the total opacity.
        if self.mode == 'lbl' or self.test_ck_shuffle_comp:
            total_tau = copy.deepcopy(self.line_struc_kappas)

            if self.hack_cloud_photospheric_tau is not None:
                if self.do_scat_emis:
                    continuum_opa_scat_emis = copy.deepcopy(self.continuum_opa_scat)
                else:
                    continuum_opa_scat_emis = np.zeros(self.continuum_opa_scat.shape)

                # TODO is this block structure intended for the user or just here for the developer tests?
                block1 = True
                block2 = True
                block3 = True
                block4 = True

                ab = np.ones_like(self.line_abundances)

                # BLOCK 1, subtract cloud, calc. tau for gas only
                if block1:
                    # Get continuum scattering opacity, without clouds:
                    continuum_opa_scat_emis = continuum_opa_scat_emis - self.hack_cloud_total_scat_aniso

                    self.line_struc_kappas = fi.mix_opas_ck(
                        ab,
                        self.line_struc_kappas,
                        -self.hack_cloud_total_abs
                    )

                    # Calc. cloud-free optical depth
                    total_tau[:, :, :1, :], self.photon_destruction_prob = \
                        fs.calc_tau_g_tot_ck_scat(
                            gravity,
                            self.press,
                            self.line_struc_kappas[:, :, :1, :],
                            self.do_scat_emis,
                            continuum_opa_scat_emis
                        )

                # BLOCK 2, calc optical depth of cloud only!
                total_tau_cloud = np.zeros_like(total_tau)

                if block2:
                    # Reduce total (absorption) line opacity by continuum absorption opacity
                    # (those two were added in  before)
                    mock_line_cloud_continuum_only = np.zeros_like(self.line_struc_kappas)

                    if not block1 and not block3 and not block4:
                        ab = np.ones_like(self.line_abundances)

                    mock_line_cloud_continuum_only = \
                        fi.mix_opas_ck(ab, mock_line_cloud_continuum_only, self.hack_cloud_total_abs)

                    # Calc. optical depth of cloud only
                    total_tau_cloud[:, :, :1, :], photon_destruction_prob_cloud = \
                        fs.calc_tau_g_tot_ck_scat(gravity,
                                                  self.press, mock_line_cloud_continuum_only[:, :, :1, :],
                                                  self.do_scat_emis, self.hack_cloud_total_scat_aniso)

                    if (not block1 and not block3) and not block4:
                        print("Cloud only (for tests purposes...)!")
                        total_tau[:, :, :1, :], self.photon_destruction_prob = \
                            total_tau_cloud[:, :, :1, :], photon_destruction_prob_cloud

                # BLOCK 3, calc. photospheric position of atmo without cloud,
                # determine cloud optical depth there, compare to
                # hack_cloud_photospheric_tau, calculate scaling ratio
                if block3:
                    median = True

                    if cloud_wlen is None:
                        # Use the full wavelength range for calculating the median
                        # optical depth of the clouds
                        wlen_select = np.ones(self.lambda_angstroem.shape[0], dtype=bool)

                    else:
                        # Use a smaller wavelength range for the median optical depth
                        # The units of cloud_wlen are converted from micron to Angstroem
                        wlen_select = (self.lambda_angstroem >= 1e4 * cloud_wlen[0]) & \
                                      (self.lambda_angstroem <= 1e4 * cloud_wlen[1])

                    # Calculate the cloud-free optical depth per wavelength
                    w_gauss_photosphere = self.w_gauss[..., np.newaxis, np.newaxis]
                    optical_depth = np.sum(w_gauss_photosphere * total_tau[:, :, 0, :], axis=0)

                    if median:
                        optical_depth_integ = np.median(optical_depth[wlen_select, :], axis=0)
                    else:
                        optical_depth_integ = np.sum(
                            (optical_depth[1:, :] + optical_depth[:-1, :]) * np.diff(self.freq)[..., np.newaxis],
                            axis=0) / (self.freq[-1] - self.freq[0]) / 2.

                    optical_depth_cloud = np.sum(w_gauss_photosphere * total_tau_cloud[:, :, 0, :], axis=0)

                    if median:
                        optical_depth_cloud_integ = np.median(optical_depth_cloud[wlen_select, :], axis=0)
                    else:
                        optical_depth_cloud_integ = np.sum(
                            (optical_depth_cloud[1:, :] + optical_depth_cloud[:-1, :]) * np.diff(self.freq)[
                                ..., np.newaxis], axis=0) / \
                                                    (self.freq[-1] - self.freq[0]) / 2.

                    # Interpolate the pressure where the optical
                    # depth of cloud-free atmosphere is 1.0

                    press_bol_clear = interp1d(optical_depth_integ, self.press)

                    try:
                        p_phot_clear = press_bol_clear(1.)
                    except ValueError:
                        p_phot_clear = self.press[-1]

                    # Interpolate the optical depth of the
                    # cloud-only atmosphere at the pressure
                    # of the cloud-free photosphere
                    tau_bol_cloud = interp1d(self.press, optical_depth_cloud_integ)
                    tau_cloud_at_phot_clear = tau_bol_cloud(p_phot_clear)

                    # Apply cloud scaling
                    self.cloud_scaling_factor = self.hack_cloud_photospheric_tau / tau_cloud_at_phot_clear

                    if len(self.fsed) > 0:
                        max_rescaling = 1e100

                        for f in self.fsed.keys():
                            mr = 2. * (self.fsed[f] + 1.)
                            max_rescaling = min(max_rescaling, mr)

                        self.scaling_physicality = self.cloud_scaling_factor / max_rescaling
                        print(f"Scaling_physicality: {self.cloud_scaling_factor / max_rescaling}")
                    else:
                        self.scaling_physicality = None

                # BLOCK 4, add scaled cloud back to opacities
                if block4:
                    # Get continuum scattering opacity, including clouds:
                    continuum_opa_scat_emis = \
                        continuum_opa_scat_emis + self.cloud_scaling_factor * self.hack_cloud_total_scat_aniso

                    self.line_struc_kappas = \
                        fi.mix_opas_ck(ab, self.line_struc_kappas,
                                       self.cloud_scaling_factor * self.hack_cloud_total_abs)

                    # Calc. total optical depth, including clouds
                    total_tau[:, :, :1, :], self.photon_destruction_prob = \
                        fs.calc_tau_g_tot_ck_scat(
                            gravity,
                            self.press,
                            self.line_struc_kappas[:, :, :1, :],
                            self.do_scat_emis,
                            continuum_opa_scat_emis
                        )
            else:
                if self.do_scat_emis:
                    total_tau[:, :, :1, :], self.photon_destruction_prob = \
                        fs.calc_tau_g_tot_ck_scat(
                            gravity,
                            self.press,
                            self.line_struc_kappas[:, :, :1, :],
                            self.do_scat_emis,
                            self.continuum_opa_scat
                        )
                else:
                    total_tau[:, :, :1, :], self.photon_destruction_prob = \
                        fs.calc_tau_g_tot_ck_scat(
                            gravity,
                            self.press,
                            self.line_struc_kappas[:, :, :1, :],
                            self.do_scat_emis,
                            np.zeros(self.continuum_opa_scat.shape)
                        )

            # To handle cases without any absorbers, where kappas are zero
            if not self.absorbers_present:
                print('No absorbers present, setting the photon'
                      ' destruction probability in the atmosphere to 1.')
                self.photon_destruction_prob[np.isnan(self.photon_destruction_prob)] = 1.

            # To handle cases when tau_cloud_at_Phot_clear = 0,
            # therefore cloud_scaling_factor = inf,
            # continuum_opa_scat_emis will contain nans and infs,
            # and photon_destruction_prob contains only nans
            if len(self.photon_destruction_prob[np.isnan(self.photon_destruction_prob)]) > 0.:
                print('Region of zero opacity detected, setting the photon'
                      ' destruction probability in this spectral range to 1.')
                self.photon_destruction_prob[np.isnan(self.photon_destruction_prob)] = 1.
                self.skip_RT_step = True
        else:
            total_tau = fs.calc_tau_g_tot_ck(
                gravity,
                self.press,
                self.line_struc_kappas
            )

        return total_tau

    @staticmethod
    def calc_pressure_hydrostatic_equilibrium(mmw, gravity, r_pl, p0, temperature, radii, rk4=True):
        # TODO is it used?
        p0 = p0 * 1e6
        vs = 1. / radii
        r_pl_sq = r_pl ** 2

        def integrand(press):
            temp = temperature(press)
            mu = mmw(press / 1e6, temp)
            if not np.isscalar(mu):
                mu = mu[0]

            integ = mu * nc.amu * gravity * r_pl_sq / nc.kB / temp
            return integ

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

    def calc_radius_hydrostatic_equilibrium(self,
                                            temperatures,
                                            mmws,
                                            gravity,
                                            p0,
                                            r_pl,
                                            variable_gravity=True,
                                            pressures=None):
        if pressures is None:
            pressures = self.press
        else:
            pressures = pressures * 1e6

        p0 = p0 * 1e6

        rho = pressures * mmws * nc.amu / nc.kB / temperatures
        radius = fs.calc_radius(
            pressures,
            gravity,
            rho,
            p0,
            r_pl,
            variable_gravity
        )

        return radius

    def calc_rosse_planck(self, temp, abunds, gravity, mmw, sigma_lnorm=None, fsed=None, kzz=None, radius=None,
                          gray_opacity=None, p_cloud=None, kappa_zero=None, gamma_scat=None,
                          haze_factor=None, add_cloud_scat_as_abs=False, dist="lognormal", b_hans=None, a_hans=None):
        """ Method to calculate the atmosphere's Rosseland and Planck mean opacities.

            Args:
                temp:
                    the atmospheric temperature in K, at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                abunds:
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
                sigma_lnorm (Optional[float]):
                    width of the log-normal cloud particle size distribution
                fsed (Optional[float]):
                    cloud settling parameter
                kzz (Optional):
                    the atmospheric eddy diffusion coeffiecient in cgs untis
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
                    Scarttering opacity at 0.35 micron, in cgs units (cm^2/g).
                gamma_scat (Optional[float]):
                    Has to be given if kappa_zero is definded, this is the
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
                    formatted as the kzz argument. Equivilant to radius arg.
                    If a_hans is not included and dist is "hansen", then it will
                    be computed using Kzz and fsed (recommended).
                b_hans (Optional[dict]):
                    A dictionary of the 'b' parameter values for each
                    included cloud species and for each atmospheric layer,
                    formatted as the kzz argument. This is the width of the hansen
                    distribution normalized by the particle area (1/a_hans^2)
        """
        if not self.do_scat_emis:
            raise ValueError(
                "pRT must run in do_scat_emis = True mode to calculate kappa_Rosseland and kappa_Planck'"
            )

        self.Pcloud = p_cloud
        self.haze_factor = haze_factor
        self.kappa_zero = kappa_zero
        self.gamma_scat = gamma_scat
        self.gray_opacity = gray_opacity
        self.interpolate_species_opa(temp)

        auto_anisotropic_cloud_scattering = False

        if self.anisotropic_cloud_scattering == 'auto':
            self.anisotropic_cloud_scattering = True
            auto_anisotropic_cloud_scattering = True
        elif not self.anisotropic_cloud_scattering:
            warnings.warn(f"anisotropic cloud scattering is recommended for emission spectra, "
                          f"but 'anisotropic_cloud_scattering' was set to {self.anisotropic_cloud_scattering}; "
                          f"set it to 'auto' to disable this warning")

        self.mix_opa_tot(abunds, mmw, gravity, sigma_lnorm, fsed, kzz, radius,
                         add_cloud_scat_as_abs=add_cloud_scat_as_abs,
                         dist=dist, a_hans=a_hans, b_hans=b_hans)

        if auto_anisotropic_cloud_scattering:
            self.anisotropic_cloud_scattering = 'auto'

        self.kappa_rosseland = \
            fs.calc_kappa_rosseland(self.line_struc_kappas[:, :, :1, :], self.temp,
                                    self.w_gauss, self.border_freqs,
                                    self.do_scat_emis, self.continuum_opa_scat)

        kappa_planck = \
            fs.calc_kappa_planck(self.line_struc_kappas[:, :, :1, :], self.temp,
                                 self.w_gauss, self.border_freqs,
                                 self.do_scat_emis, self.continuum_opa_scat)

        return self.kappa_rosseland, kappa_planck

    def calc_rt(self, contribution=False, get_kappa_rosseland=False):
        """Calculate the flux.
        """
        total_tau = self.calc_opt_depth(self.gravity, cloud_wlen=self.cloud_wlen)

        if contribution:
            self.contr_em = np.zeros((np.size(self.temp), self.freq_len), dtype='d', order='F')

        if self.do_scat_emis:
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
                self.flux, self.contr_em = fs.feautrier_rad_trans(
                    self.border_freqs,
                    total_tau[:, :, 0, :],
                    self.temp,
                    self.mu,
                    self.w_gauss_mu,
                    self.w_gauss,
                    self.photon_destruction_prob,
                    contribution,
                    self.reflectance,
                    self.emissivity,
                    self.stellar_intensity,
                    self.geometry,
                    self.mu_star
                )
            else:
                self.flux, _ = fs.feautrier_rad_trans(
                    self.border_freqs,
                    total_tau[:, :, 0, :],
                    self.temp,
                    self.mu,
                    self.w_gauss_mu,
                    self.w_gauss,
                    self.photon_destruction_prob,
                    contribution,
                    self.reflectance,
                    self.emissivity,
                    self.stellar_intensity,
                    self.geometry,
                    self.mu_star
                )

            if get_kappa_rosseland:
                if self.do_scat_emis:
                    self.kappa_rosseland = \
                        fs.calc_kappa_rosseland(
                            self.line_struc_kappas[:, :, 0, :],
                            self.temp,
                            self.w_gauss,
                            self.border_freqs,
                            self.do_scat_emis,
                            self.continuum_opa_scat
                        )
                else:
                    self.kappa_rosseland = \
                        fs.calc_kappa_rosseland(
                            self.line_struc_kappas[:, :, 0, :],
                            self.temp,
                            self.w_gauss,
                            self.border_freqs,
                            self.do_scat_emis,
                            np.zeros(self.continuum_opa_scat.shape)
                        )
        else:
            if (self.mode == 'lbl' or self.test_ck_shuffle_comp) and len(self.line_species) > 1:
                if contribution:
                    self.flux, self.contr_em = fs.flux_ck(
                        self.freq,
                        total_tau[:, :, :1, :],
                        self.temp,
                        self.mu,
                        self.w_gauss_mu,
                        self.w_gauss,
                        contribution
                    )
                else:
                    self.flux, _ = fs.flux_ck(
                        self.freq,
                        total_tau[:, :, :1, :],
                        self.temp,
                        self.mu,
                        self.w_gauss_mu,
                        self.w_gauss,
                        contribution
                    )
            else:
                if contribution:
                    self.flux, self.contr_em = fs.flux_ck(
                        self.freq,
                        total_tau,
                        self.temp,
                        self.mu,
                        self.w_gauss_mu,
                        self.w_gauss,
                        contribution
                    )
                else:
                    self.flux, _ = fs.flux_ck(
                        self.freq,
                        total_tau,
                        self.temp,
                        self.mu,
                        self.w_gauss_mu,
                        self.w_gauss,
                        contribution
                    )

    def calc_tau_cloud(self, gravity):
        """ Method to calculate the optical depth of the clouds as function of
        frequency and pressure. The array with the optical depths is set to the
        ``tau_cloud`` attribute. The optical depth is calculated from the top of
        the atmosphere (i.e. the smallest pressure). Therefore, below the cloud
        base, the optical depth is constant and equal to the value at the cloud
        base.

            Args:
                gravity (float):
                    Surface gravity in cgs. Vertically constant for emission
                    spectra.
        """
        opacity_shape = (1, self.freq_len, 1, len(self.press))
        cloud_opacity = self.cloud_total_opa_retrieval_check.reshape(opacity_shape)
        self.tau_cloud = fs.calc_tau_g_tot_ck(gravity, self.press, cloud_opacity)

    def calc_tr_rad(self, p0_bar, r_pl, gravity, mmw, contribution, variable_gravity):
        if contribution:
            self.contr_tr = np.zeros((np.size(self.press), self.freq_len), dtype='d', order='F')

        # Calculate the transmission spectrum
        if (self.mode == 'lbl' or self.test_ck_shuffle_comp) and len(self.line_species) > 1:
            self.transm_rad, self.radius_hse = self.py_calc_transm_spec(
                line_struc_kappas=self.line_struc_kappas,
                continuum_opa_scat=self.continuum_opa_scat,
                press=self.press,
                temp=self.temp,
                w_gauss=self.w_gauss,
                mmw=mmw,
                gravity=gravity,
                p0_bar=p0_bar,
                r_pl=r_pl,
                variable_gravity=variable_gravity,
                high_res=True
            )

            # TODO: contribution function calculation with python-only implementation
            if contribution:
                self.transm_rad, self.radius_hse = fs.calc_transm_spec(
                    self.line_struc_kappas[:, :, :1, :],
                    self.temp,
                    self.press,
                    gravity,
                    mmw,
                    p0_bar,
                    r_pl,
                    self.w_gauss,
                    self.scat,
                    self.continuum_opa_scat,
                    variable_gravity
                )

                self.contr_tr, self.radius_hse = fs.calc_transm_spec_contr(
                    self.line_struc_kappas[:, :, :1, :],
                    self.temp,
                    self.press,
                    gravity,
                    mmw,
                    p0_bar,
                    r_pl,
                    self.w_gauss,
                    self.transm_rad ** 2,
                    self.scat,
                    self.continuum_opa_scat,
                    variable_gravity
                )
        else:
            self.transm_rad, self.radius_hse = self.py_calc_transm_spec(
                line_struc_kappas=self.line_struc_kappas,
                continuum_opa_scat=self.continuum_opa_scat,
                press=self.press,
                temp=self.temp,
                w_gauss=self.w_gauss,
                mmw=mmw,
                gravity=gravity,
                p0_bar=p0_bar,
                r_pl=r_pl,
                variable_gravity=variable_gravity
            )

            # TODO: contribution function calculation with python-only implementation
            if contribution:
                self.transm_rad, self.radius_hse = fs.calc_transm_spec(
                    self.line_struc_kappas, self.temp,
                    self.press, gravity, mmw, p0_bar, r_pl,
                    self.w_gauss, self.scat,
                    self.continuum_opa_scat, variable_gravity
                )

                self.contr_tr, self.radius_hse = fs.calc_transm_spec_contr(
                    self.line_struc_kappas, self.temp,
                    self.press, gravity, mmw, p0_bar, r_pl,
                    self.w_gauss, self.transm_rad ** 2.,
                    self.scat,
                    self.continuum_opa_scat, variable_gravity
                )

    def calc_transm(self, temp, abunds, gravity, mmw, p0_bar, r_pl,
                    sigma_lnorm=None,
                    fsed=None, kzz=None, radius=None,
                    p_cloud=None,
                    kappa_zero=None,
                    gamma_scat=None,
                    contribution=False, haze_factor=None,
                    gray_opacity=None, variable_gravity=True,
                    dist="lognormal", b_hans=None, a_hans=None,
                    give_absorption_opacity=None,
                    give_scattering_opacity=None):
        """ Method to calculate the atmosphere's transmission radius
        (for the transmission spectrum).

            Args:
                temp:
                    the atmospheric temperature in K, at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                abunds:
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
                sigma_lnorm (Optional[float]):
                    width of the log-normal cloud particle size distribution
                fsed (Optional[float]):
                    cloud settling parameter
                kzz (Optional):
                    the atmospheric eddy diffusion coeffiecient in cgs untis
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
                    Scarttering opacity at 0.35 micron, in cgs units (cm^2/g).
                gamma_scat (Optional[float]):
                    Has to be given if kappa_zero is definded, this is the
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
                    formatted as the kzz argument. Equivilant to radius arg.
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
                give_scattering_opacity (Optional[function]):
                    A python function that takes wavelength arrays in microns and pressure arrays in bars
                    as input, and returns an isotropic scattering opacity matrix in units of cm^2/g, in the shape of
                    number of wavelength points x number of pressure points.
                    This opacity will then be added to the atmospheric absorption opacity.
                    It may be used to add simple cloud absorption laws, for example, which
                    have opacities that vary only slowly with wavelength, such that the current
                    model resolution is sufficient to resolve any variations.
        """
        self.hack_cloud_photospheric_tau = None
        self.Pcloud = p_cloud
        self.gray_opacity = gray_opacity
        self.haze_factor = haze_factor
        self.kappa_zero = kappa_zero
        self.gamma_scat = gamma_scat

        self.interpolate_species_opa(temp)

        auto_anisotropic_cloud_scattering = False

        if self.anisotropic_cloud_scattering == 'auto':
            self.anisotropic_cloud_scattering = False
        elif self.anisotropic_cloud_scattering:
            warnings.warn(f"anisotropic cloud scattering is not recommended for transmission spectra, "
                          f"but 'anisotropic_cloud_scattering' was set to {self.anisotropic_cloud_scattering}; "
                          f"set it to False or 'auto' to disable this warning")

        self.mix_opa_tot(
            abundances=abunds,
            mmw=mmw,
            gravity=gravity,
            sigma_lnorm=sigma_lnorm,
            fsed=fsed,
            kzz=kzz,
            radius=radius,
            dist=dist,
            a_hans=a_hans,
            b_hans=b_hans,
            give_absorption_opacity=give_absorption_opacity,
            give_scattering_opacity=give_scattering_opacity
        )

        if auto_anisotropic_cloud_scattering:
            self.anisotropic_cloud_scattering = 'auto'

        self.calc_tr_rad(p0_bar, r_pl, gravity, mmw, contribution, variable_gravity)

    def interpolate_cia(self, key, mfrac):
        mu_part = np.sqrt(self.CIA_species[key]['weight'])
        factor = (mfrac / mu_part) ** 2 * self.mmw / nc.amu / (nc.L0 ** 2) * self.press / nc.kB / self.temp

        x = self.CIA_species[key]['temperature']
        y = self.CIA_species[key]['lambda']
        z = self.CIA_species[key]['alpha']
        z[z < sys.float_info.min] = sys.float_info.min
        z = np.log10(self.CIA_species[key]['alpha'])

        xnew = self.temp
        ynew = nc.c / self.freq

        if x.shape[0] > 1:
            # Interpolation on temperatures for each wavelength point
            f = interp1d(x, z, kind='linear', bounds_error=False, fill_value=(z[:, 0], z[:, -1]), axis=1)
            z_temp2 = f(xnew)

            f1 = interp1d(
                y, z_temp2, kind='linear', bounds_error=False, fill_value=(np.log10(sys.float_info.min)), axis=0
            )

            znew = 10 ** f1(ynew)
            znew = np.where(znew < sys.float_info.min, 0, znew)

            return np.multiply(znew, factor)
        else:
            raise ValueError(f"petitRADTRANS require a rectangular CIA table, table shape was {x.shape}")

    def interpolate_species_opa(self, temp):
        # Interpolate line opacities to given temperature structure.
        self.temp = temp

        if len(self.line_species) > 0:
            for i, species in enumerate(self.line_species):
                self.line_struc_kappas[:, :, i, :] = fi.interpol_opa_ck(
                    self.press,
                    temp,
                    self.line_opacities_temperature_profile_grid[species],
                    self.has_custom_line_opacities_temperature_profile_grid[species],
                    self.line_opacities_temperature_grid_size[species],
                    self.line_opacities_pressure_grid_size[species],
                    self.line_opacities_grid[species]
                )
        else:
            self.line_struc_kappas = np.zeros_like(self.line_struc_kappas)

    def get_custom_pt_grid(self, path, mode, species):
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
        path_test = path + '/opacities/lines/'

        if mode == 'lbl':
            path_test = path_test + 'line_by_line/'
        elif mode == 'c-k':
            path_test = path_test + 'corr_k/'

        path_test = path_test + species + '/PTpaths.ls'

        if not os.path.isfile(path_test):
            return None
        else:
            return self._sort_opa_pt_grid(path_test)

    def get_opa(self, temp):
        """ Method to calculate and return the line opacities (assuming an abundance
        of 100 % for the inidividual species) of the Radtrans object. This method
        updates the line_struc_kappas attribute within the Radtrans class. For the
        low resolution (`c-k`) mode, the wavelength-mean within every frequency bin
        is returned.

            Args:
                temp:
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
        self.interpolate_species_opa(temp)

        return_opas = {}

        resh_wgauss = self.w_gauss.reshape((len(self.w_gauss), 1, 1))

        for i, species in enumerate(self.line_species):
            return_opas[species] = np.sum(self.line_struc_kappas[:, :, i, :] * resh_wgauss, axis=0)

        return nc.c / self.freq, return_opas

    def get_star_spectrum(self, t_star, distance, r_star=None):
        """Method to get the PHOENIX spectrum of the star and rebin it
        to the wavelength points. If Tstar is not explicitly written, the
        spectrum will be 0. If the distance is not explicitly written,
        the code will raise an error and break to urge the user to
        specify the value.

            Args:
                t_star (float):
                    the stellar temperature in K.
                distance (float):
                    the semi-major axis of the planet in cm.
                r_star (float):
                    if specified, uses this radius in cm
                    to scale the flux, otherwise it uses PHOENIX radius.
        """
        # TODO this could be static
        if t_star is not None:  # TODO whether or not returning a spectrum should not depend on this function
            if r_star is not None:
                spec = phoenix.get_PHOENIX_spec(t_star)
                rad = r_star
            else:
                spec, rad = phoenix.get_PHOENIX_spec_rad(t_star)

            add_stellar_flux = np.zeros(100)
            add_wavelengths = np.logspace(np.log10(1.0000002e-02), 2, 100)

            # import pdb
            # pdb.set_trace()

            interpwavelengths = np.append(spec[:, 0], add_wavelengths)
            interpfluxes = np.append(spec[:, 1], add_stellar_flux)

            self.stellar_intensity = fr.rebin_spectrum(interpwavelengths,
                                                       interpfluxes,
                                                       nc.c / self.freq)

            try:
                # SCALED INTENSITY (Flux/pi)
                self.stellar_intensity = self.stellar_intensity / np.pi * \
                                         (rad / distance) ** 2
            except TypeError as e:
                message = '********************************' + \
                          ' Error! Please set the semi-major axis or turn off the calculation ' + \
                          'of the stellar spectrum by removing Tstar. ********************************'
                raise Exception(message) from e
        else:
            self.stellar_intensity = np.zeros_like(self.freq)

    def mix_opa_tot(self, abundances, mmw, gravity,
                    sigma_lnorm=None, fsed=None, kzz=None,
                    radius=None,
                    add_cloud_scat_as_abs=False,
                    dist="lognormal", a_hans=None,
                    b_hans=None,
                    give_absorption_opacity=None,
                    give_scattering_opacity=None):
        """Combine total line opacities, according to mass fractions (abundances), also add continuum opacities,
        i.e. clouds, CIA...
        TODO complete docstring

        Args:
            abundances:
            mmw:
            gravity:
            sigma_lnorm:
            fsed:
            kzz:
            radius:
            add_cloud_scat_as_abs:
            dist:
            a_hans:
            b_hans:
            give_absorption_opacity:
            give_scattering_opacity:

        Returns:

        """
        self.mmw = mmw
        self.scat = False

        for i_spec in range(len(self.line_species)):
            self.line_abundances[:, i_spec] = abundances[self.line_species[i_spec]]

        self.continuum_opa = np.zeros_like(self.continuum_opa)
        self.continuum_opa_scat = np.zeros_like(self.continuum_opa_scat)

        # Calculate CIA opacity
        for key in self.CIA_species.keys():
            abund = 1

            for m in self.CIA_species[key]['molecules']:
                if m in abundances:
                    abund = abund * abundances[m]
                else:
                    found = False

                    for species_ in abundances:
                        species = species_.split('_', 1)[0]

                        if species == m:
                            abund = abund * abundances[species_]
                            found = True

                            break

                    if not found:
                        raise ValueError(f"species {m} of CIA '{key}' not found in mass mixing ratios dict "
                                         f"(listed species: {list(abundances.keys())})")

            self.continuum_opa = self.continuum_opa + self.interpolate_cia(key, np.sqrt(abund))

        # Calc. H- opacity
        if self.Hminus:
            self.continuum_opa = \
                self.continuum_opa + hminus_opacity(self.lambda_angstroem,
                                                    self.border_lambda_angstroem,
                                                    self.temp, self.press, mmw, abundances)

        # Add mock gray cloud opacity here
        if self.gray_opacity is not None:
            self.continuum_opa = self.continuum_opa + self.gray_opacity

        # Calculate rayleigh scattering opacities
        if len(self.rayleigh_species) != 0:
            self.scat = True
            self.add_rayleigh(abundances)

        # Add gray cloud deck
        if self.Pcloud is not None:
            self.continuum_opa[:, self.press > self.Pcloud * 1e6] += 1e99  # TODO why '+=' and not '='?

        # Add power law opacity
        if self.kappa_zero is not None:
            self.scat = True
            wlen_micron = nc.c / self.freq / 1e-4
            scattering_add = self.kappa_zero * (wlen_micron / 0.35) ** self.gamma_scat
            add_term = np.repeat(scattering_add[None], int(len(self.press)), axis=0).transpose()

            self.continuum_opa_scat += add_term

        # Check if hack_cloud_photospheric_tau is used with
        # a single cloud model. Combining cloud opacities
        # from different models is currently not supported
        # with the hack_cloud_photospheric_tau parameter
        if len(self.cloud_species) > 0 and self.hack_cloud_photospheric_tau is not None:
            if give_absorption_opacity is not None or give_scattering_opacity is not None:
                raise ValueError("The hack_cloud_photospheric_tau can only be "
                                 "used in combination with a single cloud model. "
                                 "Either use a physical cloud model by choosing "
                                 "cloud_species or use parametrized cloud "
                                 "opacities with the give_absorption_opacity "
                                 "and give_scattering_opacity parameters.")

        # Add optional absorption opacity from outside
        if give_absorption_opacity is None:
            if self.hack_cloud_photospheric_tau is not None:
                if not hasattr(self, "hack_cloud_total_abs"):
                    opa_shape = (self.freq.shape[0], self.press.shape[0])
                    self.hack_cloud_total_abs = np.zeros(opa_shape)
        else:
            cloud_abs = give_absorption_opacity(nc.c / self.freq / 1e-4, self.press * 1e-6)
            self.continuum_opa += cloud_abs

            if self.hack_cloud_photospheric_tau is not None:
                # This assumes a single cloud model that is
                # given by the parametrized opacities from
                # give_absorption_opacity and give_scattering_opacity
                self.hack_cloud_total_abs = cloud_abs

        # Add optional scatting opacity from outside
        if give_scattering_opacity is None:
            if self.hack_cloud_photospheric_tau is not None:
                if not hasattr(self, "hack_cloud_total_scat_aniso"):
                    opa_shape = (self.freq.shape[0], self.press.shape[0])
                    self.hack_cloud_total_scat_aniso = np.zeros(opa_shape)
        else:
            cloud_scat = give_scattering_opacity(nc.c / self.freq / 1e-4, self.press * 1e-6)
            self.continuum_opa_scat += cloud_scat

            if self.hack_cloud_photospheric_tau is not None:
                # This assumes a single cloud model that is
                # given by the parametrized opacities from
                # give_absorption_opacity and give_scattering_opacity
                self.hack_cloud_total_scat_aniso = cloud_scat

        # Add cloud opacity here, will modify self.continuum_opa
        if self._clouds_have_effect(abundances):  # add cloud opacity only if there is actually clouds
            self.scat = True
            self.calc_cloud_opacity(
                abundances,
                mmw,
                gravity,
                sigma_lnorm,
                fsed,
                kzz,
                radius,
                add_cloud_scat_as_abs,
                dist=dist,
                a_hans=a_hans,
                b_hans=b_hans
            )

        # Interpolate line opacities, combine with continuum oacities
        self.line_struc_kappas = fi.mix_opas_ck(
            self.line_abundances,
            self.line_struc_kappas,
            self.continuum_opa
        )

        # Similar to the line-by-line case below, if test_ck_shuffle_comp is
        # True, we will put the total opacity into the first species slot and
        # then carry the remaining radiative transfer steps only over that 0
        # index.
        if self.mode == 'c-k' and self.test_ck_shuffle_comp:
            self.line_struc_kappas[:, :, 0, :] = fs.combine_opas_ck(
                self.line_struc_kappas,
                self.g_gauss,
                self.w_gauss
            )

        # In the line-by-line case we can simply
        # add the opacities of different species
        # in frequency space. All opacities are
        # stored in the first species index slot
        if self.mode == 'lbl' and len(self.line_species) > 1:
            self.line_struc_kappas[:, :, 0, :] = \
                np.sum(self.line_struc_kappas, axis=2)

    def plot_opas(self,
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

        temp = np.array(temperature)
        pressure_bar = np.array(pressure_bar)

        temp = temp.reshape(1)
        pressure_bar = pressure_bar.reshape(1)

        self.press, self.continuum_opa, self.continuum_opa_scat, self.continuum_opa_scat_emis, \
            self.contr_em, self.contr_tr, self.radius_hse, self.mmw, \
            self.line_struc_kappas, self.line_struc_kappas_comb, \
            self.line_abundances, self.cloud_mass_fracs, self.r_g = \
            self._init_pressure_dependent_parameters(pressures=pressure_bar)

        wlen_cm, opas = self.get_opa(temp)
        wlen_micron = wlen_cm / 1e-4

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
                    wlen_micron,
                    plt_weights[spec] * opas[spec]
                ]

            return rets
        else:
            for spec in species:
                plt.plot(
                    wlen_micron,
                    plt_weights[spec] * opas[spec],
                    label=spec,
                    **kwargs
                )

    def py_calc_transm_spec(self, line_struc_kappas, continuum_opa_scat, press, temp, w_gauss, mmw, gravity, p0_bar,
                            r_pl, variable_gravity, high_res=False):
        """ Method to calculate the planetary transmission spectrum.

            Args:
                mmw:
                    Mean molecular weight in units of amu.
                    (1-d numpy array, same length as pressure array).
                gravity (float):
                    Atmospheric gravitational acceleration at reference pressure and radius in units of
                    dyne/cm^2
                p0_bar (float):
                    Reference pressure in bar.
                r_pl (float):
                    Reference pressure in cm.
                variable_gravity (bool):
                    If true, gravity in the atmosphere will vary proportional to 1/r^2, where r is the planet
                    radius.
                high_res (bool):
                    If true function assumes that pRT is running in lbl mode.

            Returns:
                * transmission radius in cm (1-d numpy array, as many elements as wavelengths)
                * planet radius as function of atmospheric pressure (1-d numpy array, as many elements as atmospheric
                layers)
        """
        # How many layers are there?
        struc_len = np.size(press)
        freq_len = np.size(line_struc_kappas, axis=1)

        # Calculate planetary radius in hydrostatic equilibrium, using the atmospheric structure
        # (temperature, pressure, mmw), gravity, reference pressure and radius.
        radius = self.calc_radius_hydrostatic_equilibrium(
            temperatures=temp,
            mmws=mmw,
            gravity=gravity,
            p0=p0_bar,
            r_pl=r_pl,
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
        rho = press * mmw * 1.66053892e-24 / 1.3806488e-16 / temp
        # Bring in right shape for matrix operations later.
        rho = rho.reshape(1, 1, 1, struc_len)
        rho = np.array(rho, dtype='d', order='F')

        # Bring continuum scattering opacities in right shape for matrix operations later.
        # Reminder: when calling this function, continuum absorption opacities have already
        # been added to line_struc_kappas.
        continuum_opa_scat_reshaped = continuum_opa_scat.reshape((1, freq_len, 1, struc_len))

        # Calculate the inverse mean free paths
        if high_res:
            alpha_t2 = line_struc_kappas[:, :, :1, :] * rho
            alpha_t2 += continuum_opa_scat_reshaped * rho
        else:
            alpha_t2 = line_struc_kappas * rho
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
        # Integrate over correlated-k's g-coordinate (self.wgauss == np.array([1.]) for lbl mode)
        t_graze = np.einsum('ijkl,i', t_graze, w_gauss, optimize=True)

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

    def read_line_opacities(self, arr_min, path_input_data,
                            default_temperature_grid_size=13, default_pressure_grid_size=10):
        """Read the line opacities for spectral calculation.
        The default pressure-temperature grid is a log-uniform (10, 13) grid.

        Args:
            arr_min:
            path_input_data:
            default_temperature_grid_size:
            default_pressure_grid_size:

        Returns:

        """
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
                is_exomol_hdf5_file = False  # Exomol k-table made by Katy Chubb

                if self.mode == 'c-k':
                    path_opacities = os.path.join(path_input_data, 'opacities', 'lines', 'corr_k', species)

                    if glob.glob(path_opacities + '/*.h5'):
                        is_exomol_hdf5_file = True

                if not is_exomol_hdf5_file:
                    # Check and sort custom grid for species, if defined.
                    custom_grid_data = self.get_custom_pt_grid(
                        path_input_data,
                        self.mode,
                        species
                    )

                    # If no custom grid was specified (no PTpaths.ls found), take nominal grid
                    # This assumes that the files indeed are following the nominal grid and naming convention
                    # Otherwise, it will take the info provided in PTpaths.ls which was filled into custom_grid_data
                    if custom_grid_data is None:
                        custom_line_paths[species] = None

                        self.line_opacities_temperature_profile_grid[species] = opacities_temperature_profile_grid
                        self.line_opacities_temperature_grid_size[species] = default_temperature_grid_size
                        self.line_opacities_pressure_grid_size[species] = default_pressure_grid_size
                        self.has_custom_line_opacities_temperature_profile_grid[species] = False
                    else:
                        self.line_opacities_temperature_profile_grid[species] = custom_grid_data[0]
                        custom_line_paths[species] = custom_grid_data[1]
                        self.line_opacities_temperature_grid_size[species] = custom_grid_data[2]
                        self.line_opacities_pressure_grid_size[species] = custom_grid_data[3]
                        self.has_custom_line_opacities_temperature_profile_grid[species] = True
                else:
                    # Read custom grid from the Exomol hdf5 file
                    path_opacities = os.path.join(path_input_data, 'opacities', 'lines', 'corr_k', species)
                    file_path_hdf5 = glob.glob(path_opacities + '/*.h5')[0]

                    with h5py.File(file_path_hdf5, 'r') as f:
                        lent = len(f['t'][:])
                        lenp = len(f['p'][:])
                        ret_val = np.zeros((lent * lenp, 2))

                        for i_t in range(lent):
                            for i_p in range(lenp):
                                ret_val[i_t * lenp + i_p, 1] = f['p'][i_p] * 1e6  # bar to cgs
                                ret_val[i_t * lenp + i_p, 0] = f['t'][i_t]

                        self.line_opacities_temperature_profile_grid[species] = ret_val
                        self.line_opacities_temperature_grid_size[species] = lent
                        self.line_opacities_pressure_grid_size[species] = lenp
                        self.has_custom_line_opacities_temperature_profile_grid[species] = True

        # Read the opacities
        if len(self.line_species) > 0:
            for species in self.line_species:
                # Check if it is an Exomol hdf5 file that needs to be read
                is_exomol_hdf5_file = False

                if self.mode == 'c-k':
                    path_opacities = os.path.join(path_input_data, 'opacities', 'lines', 'corr_k', species)

                    if glob.glob(path_opacities + '/*.h5'):
                        is_exomol_hdf5_file = True

                if not is_exomol_hdf5_file:
                    if not self.has_custom_line_opacities_temperature_profile_grid[species]:
                        len_tp = len(opacities_temperature_profile_grid[:, 0])
                    else:
                        len_tp = len(self.line_opacities_temperature_profile_grid[species][:, 0])

                    custom_file_names = ''

                    if self.has_custom_line_opacities_temperature_profile_grid[species]:
                        for i_TP in range(len_tp):
                            custom_file_names = custom_file_names + custom_line_paths[species][i_TP] + ':'

                    # TODO PAUL EXO_K Project: do index_fill treatment from below!
                    # Also needs to read "custom" freq_len and freq here again!
                    # FOR ALL TABLES, HERE AND CHUBB: TEST THAT THE GRID IS INDEED THE SAME AS REQUIRED IN THE REGIONS
                    # WITH OPACITY. NEXT STEPS AFTER THIS:
                    # (i) MAKE ISOLATED EXO_K rebinning test,
                    # (ii) Do external bin down script, save as pRT method
                    # (iii) enable on the fly down-binning,
                    # (iv) look into outsourcing methods from class to separate files, this here is getting too long!

                    if self.mode == 'c-k':
                        local_freq_len, local_g_len = fi.get_freq_len(path_input_data, species)
                        local_freq_len_full = copy.copy(local_freq_len)
                        # Read in the frequency range of the opcity data
                        local_freq, local_border_freqs = fi.get_freq(
                            path_input_data, species, local_freq_len
                        )
                    else:
                        if self.lbl_opacity_sampling is None:
                            local_freq_len_full = self.freq_len
                        else:
                            local_freq_len_full = self.freq_len * self.lbl_opacity_sampling

                        local_g_len = self.g_len
                        local_freq = None

                    self.line_opacities_grid[species] = \
                        fi.read_in_molecular_opacities(
                            path_input_data,
                            species + ':',
                            local_freq_len_full,
                            local_g_len,
                            1,
                            len_tp,
                            self.mode,
                            arr_min,
                            self.has_custom_line_opacities_temperature_profile_grid[species],
                            custom_file_names
                        )

                    if np.all(self.line_opacities_grid[species] == -1):
                        raise RuntimeError("molecular opacity loading failed, check above outputs to find the cause")

                    if self.mode == 'c-k':
                        # Initialize an empty array that has the same spectral entries as
                        # pRT object has nominally. Only fill those values where the k-tables
                        # have entries.
                        ret_val = np.zeros(self.g_len * self.freq_len * len_tp).reshape(
                            (self.g_len, self.freq_len, 1, len_tp)
                        )

                        # Indices in retVal to be filled with read-in opacities
                        index_fill = (self.freq <= local_freq[0] * (1. + 1e-10)) & \
                                     (self.freq >= local_freq[-1] * (1. - 1e-10))
                        # Indices of read-in opacities to be filled into retVal
                        index_use = (local_freq <= self.freq[0] * (1. + 1e-10)) & \
                                    (local_freq >= self.freq[-1] * (1. - 1e-10))

                        ret_val[:, index_fill, 0, :] = \
                            self.line_opacities_grid[species][:, index_use, 0, :]
                        self.line_opacities_grid[species] = ret_val

                    # Down-sample opacities in lbl mode if requested
                    if self.mode == 'lbl' and self.lbl_opacity_sampling is not None:
                        self.line_opacities_grid[species] = \
                            self.line_opacities_grid[
                                species][:, ::self.lbl_opacity_sampling, :]
                else:
                    print(f" Reading line opacities of species '{species}'...")

                    path_opacities = os.path.join(path_input_data, 'opacities', 'lines', 'corr_k', species)
                    file_path_hdf5 = glob.glob(path_opacities + '/*.h5')[0]

                    with h5py.File(file_path_hdf5, 'r') as f:
                        lenf = len(f['bin_centers'][:])
                        freqs_chubb = nc.c * f['bin_centers'][:][::-1]
                        lent = len(f['t'][:])
                        lenp = len(f['p'][:])

                        # Swap axes to correctly load Exomol tables.
                        k_table = np.array(f['kcoeff'])
                        k_table = np.swapaxes(k_table, 0, 1)
                        k_table2 = k_table.reshape((lenp * lent, lenf, 16))
                        k_table2 = np.swapaxes(k_table2, 0, 2)
                        k_table2 = k_table2[:, ::-1, :]

                        # Initialize an empty array that has the same spectral entries as
                        # pRT object has nominally. Only fill those values where the Exomol tables
                        # have entries.
                        ret_val = np.zeros(
                            self.g_len * self.freq_len * len(self.line_opacities_temperature_profile_grid[species])
                        ).reshape(
                            (self.g_len, self.freq_len, 1, len(self.line_opacities_temperature_profile_grid[species]))
                        )
                        index_fill = (self.freq <= freqs_chubb[0] * (1. + 1e-10)) & \
                                     (self.freq >= freqs_chubb[-1] * (1. - 1e-10))
                        index_use = (freqs_chubb <= self.freq[0] * (1. + 1e-10)) & \
                                    (freqs_chubb >= self.freq[-1] * (1. - 1e-10))
                        ret_val[:, index_fill, 0, :] = k_table2[:, index_use, :]

                        ret_val[ret_val < 0.] = 0.

                        # Divide by mass to convert cross-sections to opacities
                        exomol_mass = float(f['mol_mass'][0])
                        self.line_opacities_grid[species] = ret_val / exomol_mass / nc.amu
                        print('Done.')

                # Cut the wavelength range of the just-read species to the wavelength range requested by the user
                if self.mode == 'c-k':
                    self.line_opacities_grid[species] = \
                        np.array(
                            self.line_opacities_grid[species][:, :, 0, :],
                            dtype='d', order='F'
                        )
                else:
                    self.line_opacities_grid[species] = \
                        np.array(self.line_opacities_grid[species][:, :, 0, :], dtype='d', order='F')

            print('\n')

        # Read in g grid for correlated-k
        if self.mode == 'c-k':
            buffer = np.genfromtxt(
                os.path.join(path_input_data, 'opa_input_files', 'g_comb_grid.dat')
            )
            self.g_gauss = np.array(buffer[:, 0], dtype='d', order='F')
            self.w_gauss = np.array(buffer[:, 1], dtype='d', order='F')

    def read_cloud_opas(self, path_input_data):
        # Function to read cloud opacities
        self.cloud_species_mode = []

        for i in range(int(len(self.cloud_species))):
            splitstr = self.cloud_species[i].split('_')
            self.cloud_species_mode.append(splitstr[1])
            self.cloud_species[i] = splitstr[0]

        # Prepare single strings delimited by ':' which are then
        # put into F routines
        tot_str_names = ''

        for cloud_species in self.cloud_species:
            tot_str_names = tot_str_names + cloud_species + ':'

        tot_str_modes = ''

        for cloud_species_mode in self.cloud_species_mode:
            tot_str_modes = tot_str_modes + cloud_species_mode + ':'

        n_cloud_wavelength_bins = int(len(np.genfromtxt(
            os.path.join(path_input_data, 'opacities', 'continuum', 'clouds',
                         'MgSiO3_c', 'amorphous', 'mie', 'opa_0001.dat')
        )[:, 0]))

        # Actual reading of opacities
        rho_cloud_particles, cloud_specs_abs_opa, cloud_specs_scat_opa, \
            cloud_aniso, cloud_lambdas, cloud_rad_bins, cloud_radii \
            = fi.read_in_cloud_opacities(
                path_input_data, tot_str_names, tot_str_modes,
                len(self.cloud_species), n_cloud_wavelength_bins
            )

        cloud_specs_abs_opa[cloud_specs_abs_opa < 0.] = 0.
        cloud_specs_scat_opa[cloud_specs_scat_opa < 0.] = 0.

        self.rho_cloud_particles = np.array(rho_cloud_particles, dtype='d', order='F')
        self.cloud_specs_abs_opa = np.array(cloud_specs_abs_opa, dtype='d', order='F')
        self.cloud_specs_scat_opa = np.array(cloud_specs_scat_opa, dtype='d', order='F')
        self.cloud_aniso = np.array(cloud_aniso, dtype='d', order='F')
        self.cloud_lambdas = np.array(cloud_lambdas, dtype='d', order='F')
        self.cloud_rad_bins = np.array(cloud_rad_bins, dtype='d', order='F')
        self.cloud_radii = np.array(cloud_radii, dtype='d', order='F')

    def write_out_rebin(self, resolution, path='', species=None, masses=None):
        import exo_k

        if species is None:
            species = []

        # Define own wavenumber grid, make sure that log spacing is constant everywhere
        n_spectral_points = int(resolution * np.log(self.wlen_bords_micron[1] / self.wlen_bords_micron[0]) + 1)
        wavenumber_grid = np.logspace(np.log10(1 / self.wlen_bords_micron[1] / 1e-4),
                                      np.log10(1. / self.wlen_bords_micron[0] / 1e-4),
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

            f.create_dataset('bin_centers', data=self.freq[::-1] / nc.c)
            f.create_dataset('bin_edges', data=self.border_freqs[::-1] / nc.c)
            ret_opa_table = copy.copy(self.line_opacities_grid[spec])

            # Mass to go from opacities to cross-sections
            ret_opa_table = ret_opa_table * nc.amu * masses[spec.split('_')[0]]

            # Do the opposite of what I do when reading in Katy's Exomol tables
            # To get opacities into the right format
            ret_opa_table = ret_opa_table[:, ::-1, :]
            ret_opa_table = np.swapaxes(ret_opa_table, 2, 0)
            ret_opa_table = ret_opa_table.reshape((
                self.line_opacities_temperature_grid_size[spec],
                self.line_opacities_pressure_grid_size[spec],
                self.freq_len,
                len(self.w_gauss)
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
            f.create_dataset('ngauss', data=len(self.w_gauss))
            f.create_dataset('p', data=self.line_opacities_temperature_profile_grid[spec][
                                       :self.line_opacities_pressure_grid_size[spec], 1] / 1e6)
            f['p'].attrs.create('units', 'bar')
            f.create_dataset('samples', data=self.g_gauss)
            f.create_dataset('t', data=self.line_opacities_temperature_profile_grid[spec][
                                       ::self.line_opacities_pressure_grid_size[spec], 0])
            f.create_dataset('weights', data=self.w_gauss)
            f.create_dataset('wlrange', data=[np.min(nc.c / self.border_freqs / 1e-4),
                                              np.max(nc.c / self.border_freqs / 1e-4)])
            f.create_dataset('wnrange', data=[np.min(self.border_freqs / nc.c),
                                              np.max(self.border_freqs / nc.c)])
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


def __sigma_hm_ff(lambda_angstroem, temp, p_e):
    """
    Returns the H- free-free cross-section in units of cm^2
    per H per e- pressure (in cgs), as defined on page 156 of
    "The Observation and Analysis of Stellar Photospheres"
    by David F. Gray
    """

    index = (lambda_angstroem >= 2600.) & (lambda_angstroem <= 113900.)
    lamb_use = lambda_angstroem[index]

    if temp >= 2500.:
        # Convert to Angstrom (from cgs)
        theta = 5040. / temp

        f0 = -2.2763 - 1.6850 * np.log10(lamb_use) \
            + 0.76661*np.log10(lamb_use)**2. \
            - 0.053346*np.log10(lamb_use)**3.
        f1 = 15.2827 - 9.2846 * np.log10(lamb_use) \
            + 1.99381*np.log10(lamb_use)**2. \
            - 0.142631*np.log10(lamb_use)**3.
        f2 = -197.789 + 190.266 * np.log10(lamb_use) - 67.9775*np.log10(lamb_use)**2. \
            + 10.6913*np.log10(lamb_use)**3. - 0.625151*np.log10(lamb_use)**4.

        ret_val = np.zeros_like(lambda_angstroem)
        ret_val[index] = 1e-26 * p_e * 1e1 ** (
                f0 + f1 * np.log10(theta) + f2 * np.log10(theta) ** 2.)
        return ret_val

    else:

        return np.zeros_like(lambda_angstroem)


def __sigma_bf_mean(border_lambda_angstroem):
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


def hminus_opacity(lambda_angstroem, border_lambda_angstroem,
                   temp, press, mmw, abundances):
    """ Calc the H- opacity."""

    ret_val = np.array(np.zeros(len(lambda_angstroem) * len(press)).reshape(
        len(lambda_angstroem),
        len(press)), dtype='d', order='F')

    # Calc. electron number fraction
    # e- mass in amu:
    m_e = 5.485799e-4
    n_e = mmw / m_e * abundances['e-']

    # Calc. e- partial pressure
    p_e = press * n_e

    kappa_hminus_bf = __sigma_bf_mean(border_lambda_angstroem) / nc.amu

    for i_struct in range(len(n_e)):
        kappa_hminus_ff = __sigma_hm_ff(lambda_angstroem, temp[i_struct],
                                        p_e[i_struct]) / nc.amu * abundances['H'][i_struct]

        ret_val[:, i_struct] = kappa_hminus_bf * abundances['H-'][i_struct] \
            + kappa_hminus_ff

    return ret_val


def py_calc_cloud_opas(
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
