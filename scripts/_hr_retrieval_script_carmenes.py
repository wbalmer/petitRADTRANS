"""Launch a retrieval of CARMENES data. A lot of command line options are available.
Run with:
    mpiexec -n N --use-hwthread-cpus python3 _hr_retrieval_script_carmenes.py
N is the number of processes.
Try:
    sudo mpiexec -n N --allow-run-as-root ...
If for some reason the script crashes.
"""
import argparse
import copy
import os
import pathlib
import shutil
import time
import warnings

import numpy as np
import petitRADTRANS.physical_constants as cst
from astropy.io import fits
from petitRADTRANS.cli.eso_skycalc_cli import get_tellurics_npz
from petitRADTRANS.planet import Planet
from petitRADTRANS.spectral_model import SpectralModel
from petitRADTRANS.retrieval.data import Data
from petitRADTRANS.plotlib.plotlib import contour_corner
from petitRADTRANS.retrieval.preparing import preparing_pipeline_sysrem
from petitRADTRANS.utils import fill_object


def _init_parser():
    # Arguments definition
    _parser = argparse.ArgumentParser(
        description='Launch HR retrieval script'
    )

    _parser.add_argument(
        '--planet-name',
        default='HD 189733 b',
        help='planet name'
    )

    _parser.add_argument(
        '--output-directory',
        default=pathlib.Path.home(),
        help='directory where to save the results'
    )

    _parser.add_argument(
        '--additional-data-directory',
        default=pathlib.Path.home(),
        help='directory where the additional data are stored'
    )

    _parser.add_argument(
        '--mode',
        default='transmission',
        help='spectral model mode, emission or transmission'
    )

    _parser.add_argument(
        '--uncertainties-mode',
        default='default',
        help="set how the uncertainties should be treated in the retrieval ('default'|'optimize'|'retrieve')"
    )

    _parser.add_argument(
        '--retrieval-name',
        default='',
        help='name of the retrieval'
    )

    _parser.add_argument(
        '--retrieval-parameters',
        nargs='+',
        default='temperature',
        help='parameters to retrieve'
    )

    _parser.add_argument(
        '--detector-selection-name',
        default='strict',
        help='detector selection name'
    )

    _parser.add_argument(
        '--n-live-points',
        type=int,
        default=100,
        help='number of live points to use in the retrieval'
    )

    _parser.add_argument(
        '--resume',
        action='store_true',
        help='if activated, resume retrievals'
    )

    _parser.add_argument(
        '--tellurics-mask-threshold',
        type=float,
        default=0.8,
        help='telluric mask threshold for reprocessing'
    )

    _parser.add_argument(
        '--mid-transit-time-range',
        type=float,
        default=600,
        help='mid transit time prior range'
    )

    _parser.add_argument(
        '--add-data-offset',
        action='store_true',
        help='if activated, add the data offset map to the data'
    )

    _parser.add_argument(
        '--use-t14',
        action='store_true',
        help='if activated, use total transit exposures rather than mid transit time prior exposures'
    )

    _parser.add_argument(
        '--use-t23',
        action='store_true',
        help='if activated, use full transit exposures rather than mid transit time prior exposures'
    )

    _parser.add_argument(
        '--use-t1535',
        action='store_true',
        help='if activated, use full transit exposures and half eclipses rather than total transit exposures'
    )

    _parser.add_argument(
        '--use-boxcar-tlloss',
        action='store_true',
        help='if activated, use a box car transit light loss function instead of the more complex one'
    )

    _parser.add_argument(
        '--use-sysrem',
        action='store_true',
        help='if activated, use the SysRem preparing pipeline'
    )

    _parser.add_argument(
        '--use-sysrem-subtract',
        action='store_true'
               '',
        help='if activated, use the subtract mode of SysRem'
    )

    _parser.add_argument(
        '--use-gaussian-priors',
        action='store_true',
        help='if activated, use Gaussian priors for a selection of parameters'
    )

    _parser.add_argument(
        '--retrieve-mock-observations',
        action='store_true',
        help='if activated, retrieve mock observations instead of the real data'
    )

    _parser.add_argument(
        '--retrieve-3d-mock-observations',
        action='store_true',
        help='if activated, retrieve the 3D Orange mock observations instead of the real data'
    )

    _parser.add_argument(
        '--use-simulated-uncertainties',
        action='store_true',
        help='if activated, use simulated uncertainties instead opf the real uncertainties'
    )

    _parser.add_argument(
        '--add-noise',
        action='store_true',
        help='if activated, add noise to the mock observations'
    )

    _parser.add_argument(
        '--noise-seed',
        type=int,
        default=-1,
        help='seed to use for noise matrix'
    )

    _parser.add_argument(
        '--n-transits',
        type=float,
        default=1.0,
        help='number of planetary transits for mock observations'
    )

    _parser.add_argument(
        '--check',
        action='store_true',
        help='if activated, check the validity of the reprocessing pipeline'
    )

    _parser.add_argument(
        '--no-retrieve',
        action='store_false',
        help='if activated, run the retrieval'
    )

    _parser.add_argument(
        '--no-archive',
        action='store_false',
        help='if activated, archive the output directory'
    )

    _parser.add_argument(
        '--no-scale',
        action='store_false',
        help='if activated, scale the model spectrum'
    )

    _parser.add_argument(
        '--no-shift',
        action='store_false',
        help='if activated, shift the model spectrum'
    )

    _parser.add_argument(
        '--no-convolve',
        action='store_false',
        help='if activated, convolve the model spectrum'
    )

    _parser.add_argument(
        '--no-rebin',
        action='store_false',
        help='if activated, rebin the model spectrum'
    )

    _parser.add_argument(
        '--no-tlloss',
        action='store_false',
        help='if activated, add transit light loss'
    )

    return _parser


parser = _init_parser()


def _calculate_transit_fractional_light_loss_uniform(planet_radius_normalized, planet_star_centers_distance,
                                                     planet_radius_normalized_squared=None, **kwargs):
    """Calculate the fractional light loss observed when a planet transit a star.
    This equation neglects the effect of limb-darkening, assuming that the source is uniform.
    This version also assumes that as soon as the transit begins, the planet's disk is fully inside the star's disk,
    i.e. there is no ingress/egress.

    Args:
        planet_radius_normalized: planet radius over its star radius
        planet_radius_normalized_squared: planet radius over its star radius, squared
        planet_star_centers_distance: sky-projected distance between the centers of the planet and the star,
            normalized over the radius of the star
    """
    transit_fractional_light_loss = np.zeros(np.shape(planet_star_centers_distance))

    if planet_radius_normalized_squared is None:
        planet_radius_normalized_squared = planet_radius_normalized ** 2

    partial_transit = np.nonzero(
        np.logical_and(
            np.less_equal(planet_star_centers_distance, 1 + planet_radius_normalized),
            np.greater(planet_star_centers_distance, np.abs(1 - planet_radius_normalized))
        )
    )
    full_transit = np.nonzero(np.less_equal(planet_star_centers_distance, 1 - planet_radius_normalized))
    star_totally_eclipsed = np.nonzero(np.less_equal(planet_star_centers_distance, planet_radius_normalized - 1))

    if partial_transit[0].size > 0:
        transit_fractional_light_loss[partial_transit] = planet_radius_normalized_squared[partial_transit]

    transit_fractional_light_loss[full_transit] = planet_radius_normalized_squared[full_transit]
    transit_fractional_light_loss[star_totally_eclipsed] = 1

    return transit_fractional_light_loss


def calculate_transit_fractional_light_loss_box_car(spectrum, **kwargs):
    """Modified transit light loss function with no ingress/egress."""
    planet_radius_normalized_squared = 1 - spectrum
    planet_radius_normalized = np.sqrt(planet_radius_normalized_squared)

    planet_star_centers_distance = SpectralModel._calculate_planet_star_centers_distance(
        planet_radius_normalized=planet_radius_normalized,
        **kwargs
    )

    spectrum_transit_fractional_light_loss = _calculate_transit_fractional_light_loss_uniform(
        planet_radius_normalized=planet_radius_normalized,
        planet_radius_normalized_squared=planet_radius_normalized_squared,
        planet_star_centers_distance=planet_star_centers_distance
    )

    return 1 - spectrum_transit_fractional_light_loss


def data_selection(wavelengths_instrument, observed_spectra, uncertainties, instrument_snr, instrumental_deformations,
                   simulated_snr, times, mid_transit_time,
                   transit_duration, orbital_phases, airmasses, barycentric_velocities, detector_selection,
                   n_transits=1, use_t23=False, use_t1535=False):
    """Apply the exposures and orders selection for the retrieval to the data."""
    wavelengths_instrument = wavelengths_instrument[detector_selection][:, 0, :]
    observed_spectra = observed_spectra[detector_selection]
    uncertainties = uncertainties[detector_selection]
    instrument_snr = instrument_snr[detector_selection]
    instrumental_deformations = instrumental_deformations[detector_selection]
    simulated_snr = simulated_snr[detector_selection]

    # Select only in-transit observations
    wh = np.where(np.logical_and(times >= mid_transit_time - transit_duration / 2,
                                 times <= mid_transit_time + transit_duration / 2))

    if use_t23 and not use_t1535:
        np.insert(wh, 0, wh[0] - 1)
        np.insert(wh, 0, wh[0] - 1)
        np.insert(wh, -1, wh[-1] + 1)
        np.insert(wh, -1, wh[-1] + 1)

    observed_spectra = observed_spectra[:, wh[0], :]
    uncertainties = uncertainties[:, wh[0], :]
    instrument_snr = instrument_snr[:, wh[0], :]
    instrumental_deformations = instrumental_deformations[:, wh[0], :]
    simulated_snr = simulated_snr[:, wh[0], :]
    orbital_phases = orbital_phases[wh[0]]
    airmasses = airmasses[wh[0]]
    barycentric_velocities = barycentric_velocities[wh[0]]
    times = times[wh[0]]

    instrument_snr *= np.sqrt(n_transits)
    instrument_snr = np.ma.masked_less_equal(instrument_snr, 1)

    uncertainties = np.ma.masked_where(instrument_snr.mask, uncertainties)

    # Completely mask column where at least one value is masked
    masked_value_in_column = np.any(uncertainties.mask, axis=1)
    spectra_mask = np.moveaxis(uncertainties.mask, 1, 2)
    spectra_mask[masked_value_in_column] = True
    uncertainties = np.ma.masked_where(np.moveaxis(spectra_mask, 2, 1), uncertainties)

    observed_spectra = np.ma.masked_where(uncertainties.mask, observed_spectra)

    return wavelengths_instrument, observed_spectra, uncertainties, instrument_snr, instrumental_deformations, \
        simulated_snr, orbital_phases, airmasses, \
        barycentric_velocities, times


def load_additional_data(data_dir, wavelengths_instrument, airmasses, resolving_power, times='TDB',
                         simulate_uncertainties=True, load_offset=False):
    """Load and generate tellurics transmittances, deformation matrix, simulated uncertainties and data offset"""
    # Tellurics
    telluric_transmittance_file = \
        os.path.join(data_dir, f"transmission_carmenes.npz")

    print(f"Loading transmittance from file '{telluric_transmittance_file}'...")
    wavelength_range_rebin = SpectralModel.calculate_optimal_wavelengths_boundaries(
        output_wavelengths=wavelengths_instrument,
        shift_wavelengths_function=SpectralModel.shift_wavelengths,
        relative_velocities=None  # telluric lines are not shifted
    )

    # Ensure that all the necessary wavelengths are fetched for
    # Skycalc returns all wavelengths within the wavelengths boundaries, not including the boundaries themselves
    wavelength_range_rebin[0] -= wavelength_range_rebin[0] / 1e6  # 1e6 is the default resolving power of Skycalc
    wavelength_range_rebin[-1] += wavelength_range_rebin[-1] / 1e6

    wavelengths_telluric, telluric_transmittance_0 = get_tellurics_npz(
        telluric_transmittance_file, wavelength_range_rebin
    )

    # Variable throughput
    variable_throughput_file = os.path.join(data_dir, 'instrumental_deformations_carmenes.npy')
    print(f"Loading variable throughput from file '{variable_throughput_file}'...")
    variable_throughput = np.load(variable_throughput_file)

    # SNR
    if simulate_uncertainties:
        # Convolve telluric transmittances to correctly reproduce the effect on SNR
        telluric_transmittance_snr_0 = SpectralModel.convolve(
            input_wavelengths=wavelengths_telluric,
            input_spectrum=telluric_transmittance_0,
            new_resolving_power=resolving_power,
            constance_tolerance=1e300
        )

        telluric_transmittance_snr_0 = np.ma.masked_less_equal(telluric_transmittance_snr_0, 0.0)
        telluric_transmittance_snr_0 = SpectralModel.rebin_spectrum(
            input_wavelengths=wavelengths_telluric,
            input_spectrum=telluric_transmittance_snr_0,
            output_wavelengths=wavelengths_instrument[:, 0, :]
        )[1]

        telluric_transmittance_snr_0 = np.tile(telluric_transmittance_snr_0, (airmasses.size, 1, 1))
        telluric_transmittance_snr_0 = np.moveaxis(telluric_transmittance_snr_0, 0, 1)

        telluric_transmittance_snr = np.ma.masked_less(
            np.moveaxis(np.exp(np.ma.log(np.moveaxis(telluric_transmittance_snr_0, 1, 2)) * airmasses), 2, 1),
            0.0
        ).filled(0.0)

        simulated_uncertainties = np.sqrt(np.ma.masked_less_equal(telluric_transmittance_snr, 0.0)).filled(0.0)
    else:
        simulated_uncertainties = None

    telluric_transmittance = np.ma.masked_less(telluric_transmittance_0, 1e-6).filled(1e-6)

    # Offset
    if load_offset:
        '''
        In case the data have an unknown offset. This is probably not the case for the CARMENES data.
        data_fit_carmenes contains a fit of the data using a tellurics + stellar lines model. 
        '''
        data_offset_file = os.path.join(data_dir, f"data_fit_carmenes.npz")
        samples = np.load(data_offset_file, allow_pickle=True)['samples']
        offset = np.asarray([[np.median(exposure[:, 2]) for exposure in order]for order in samples])
    else:
        offset = None

    return variable_throughput, wavelengths_telluric, telluric_transmittance, simulated_uncertainties, offset


def load_carmenes_data(directory, mid_transit_jd, times_format='TDB'):
    """Load the CARMENES data. The BJD_TDB times should be used."""
    with fits.open(os.path.join(directory, 'airmass.fits')) as f:
        airmass = f[0].data

    with fits.open(os.path.join(directory, 'bary.fits')) as f:
        barycentric_velocities = f[0].data

    if times_format == 'UTC':
        warnings.warn('loading BJD_UTC times (most time references are in BJD_TDB)')
        with fits.open(os.path.join(directory, 'mod_julian_date.fits')) as f:
            dates = f[0].data  # in MJD - 0.5 format, corresponds to 2017-09-07
    elif times_format == 'TDB':
        print('Loading BJD_TDB times...')
        dates = np.load(os.path.join(directory, 'mod_julian_date_tdb.npz'))
        dates = dates['times']
    else:
        raise ValueError(f"time format must be 'TDB'|'UTC', but was '{times_format}'")

    with fits.open(os.path.join(directory, 'noise.fits')) as f:  # is it actually 1/noise ?... or SNR?
        noise = f[0].data

    with fits.open(os.path.join(directory, 'phase.fits')) as f:  # is it actually 1/noise ?... or SNR?
        orbital_phases = f[0].data

    with fits.open(os.path.join(directory, 'original_spectra.fits')) as f:
        spectra = f[0].data

    with fits.open(os.path.join(directory, 'wavelength.fits')) as f:
        wavelengths = f[0].data

    # Init
    spectra = np.moveaxis(spectra, 0, 2)
    spectra = np.reshape(spectra, (45, int(4080 / 2), 28 * 2), order='F')  # separate the 2 CCD
    spectra = np.moveaxis(spectra, 2, 0)

    wavelengths = np.reshape(wavelengths, (45, int(4080 / 2), 28 * 2), order='F')
    wavelengths = np.moveaxis(wavelengths, 2, 0)

    noise = np.reshape(noise, (45, int(4080 / 2), 28 * 2), order='F')
    noise = np.moveaxis(noise, 2, 0)
    noise = np.ma.masked_invalid(noise)
    noise = np.ma.masked_less_equal(noise, 0)

    mid_transit = np.mod(mid_transit_jd, 1.0) * 24 * 3600  # seconds from BJ day start
    times = np.mod(dates, 1.0) * 24 * 3600  # seconds

    spectra = np.ma.masked_invalid(spectra)
    spectra = np.ma.masked_less_equal(spectra, 0)
    spectra = np.ma.masked_where(noise.mask, spectra)

    noise = np.ma.masked_where(spectra.mask, noise)

    snr = spectra / noise

    return wavelengths, spectra, snr, noise, orbital_phases, airmass, barycentric_velocities, times, mid_transit


def load_orange_simulation_dat(directory, base_name, extension='dat'):
    """Load the pRT-Orange simulation dat files."""
    wavelengths = np.array([])
    data = np.array([])

    i = 0
    filename = f"{os.path.join(os.path.abspath(directory), base_name)}_{i:02d}.{extension}"

    if not os.path.isfile(filename):
        raise FileNotFoundError(f"file '{filename}' not found")

    while os.path.isfile(filename):
        print(f"Loading from '{filename}'...")
        file_data = np.loadtxt(filename)
        wavelengths = np.append(wavelengths, file_data[:, 0])
        data = np.append(data, file_data[:, 1])

        i += 1
        filename = f"{os.path.join(os.path.abspath(directory), base_name)}_{i:02d}.{extension}"

    wavelengths_id_sorted = np.argsort(wavelengths)
    wavelengths = wavelengths[wavelengths_id_sorted]
    data = data[wavelengths_id_sorted]

    wavelengths, wavelengths_id_unique = np.unique(wavelengths, return_index=True)
    data = data[wavelengths_id_unique]

    data = np.sqrt(data / np.pi)  # area to radius

    return wavelengths, data


def load_orange_tellurics(data_dir, wavelengths_instrument, airmasses, resolving_power, simulate_snr=True):
    """Load the pRT-Orange simulation tellurics files."""
    # Tellurics
    telluric_transmittance_file = \
        os.path.join(data_dir, f"transmission_carmenes_orange.npz")

    print(f"Loading transmittance from file '{telluric_transmittance_file}'...")
    wavelength_range_rebin = SpectralModel.calculate_optimal_wavelengths_boundaries(
        output_wavelengths=wavelengths_instrument,
        shift_wavelengths_function=SpectralModel.shift_wavelengths,
        relative_velocities=None  # telluric lines are not shifted
    )

    # Ensure that all the necessary wavelengths are fetched for
    # Skycalc returns all wavelengths within the wavelengths boundaries, not including the boundaries themselves
    wavelength_range_rebin[0] -= wavelength_range_rebin[0] / 1e6  # 1e6 is the default resolving power of Skycalc
    wavelength_range_rebin[-1] += wavelength_range_rebin[-1] / 1e6

    wavelengths_telluric, telluric_transmittance_0 = get_tellurics_npz(
        telluric_transmittance_file, wavelength_range_rebin
    )

    # SNR
    if simulate_snr:
        # Convolve telluric transmittances to correctly reproduce the effect on SNR
        telluric_transmittance_snr_0 = SpectralModel.convolve(
            input_wavelengths=wavelengths_telluric,
            input_spectrum=telluric_transmittance_0,
            new_resolving_power=resolving_power,
            constance_tolerance=1e300
        )

        telluric_transmittance_snr_0 = np.ma.masked_less_equal(telluric_transmittance_snr_0, 0.0)
        telluric_transmittance_snr_0 = SpectralModel.rebin_spectrum(
            input_wavelengths=wavelengths_telluric,
            input_spectrum=telluric_transmittance_snr_0,
            output_wavelengths=wavelengths_instrument
        )[1]

        telluric_transmittance_snr_0 = np.tile(telluric_transmittance_snr_0, (airmasses.size, 1, 1))
        telluric_transmittance_snr_0 = np.moveaxis(telluric_transmittance_snr_0, 0, 1)

        telluric_transmittance_snr = np.ma.masked_less(
            np.moveaxis(np.exp(np.ma.log(np.moveaxis(telluric_transmittance_snr_0, 1, 2)) * airmasses), 2, 1),
            0.0
        ).filled(0.0)

        simulated_snr = np.sqrt(np.ma.masked_less_equal(telluric_transmittance_snr, 0.0)).filled(0.0)
    else:
        simulated_snr = None

    return wavelengths_telluric, telluric_transmittance_0, simulated_snr


def get_orange_simulation_model(directory, base_name,
                                additional_data_directory, spectral_model,
                                extension='dat', reduce=False, **kwargs):
    """Get the modified pRT-Orange model."""
    kwargs_ = copy.deepcopy(kwargs)
    orange_simulation_file = os.path.join(directory, 'orange_hd_189733_b_transmission.npz')

    if not os.path.isfile(orange_simulation_file):
        print(f"Loading Orange model from dat files...")
        wavelengths, data = load_orange_simulation_dat(
            directory=directory,
            base_name=base_name,
            extension=extension
        )

        np.savez_compressed(file=orange_simulation_file, wavelengths=wavelengths, data=data)
    else:
        print(f"Loading Orange model from file '{orange_simulation_file}'...")
        data = np.load(orange_simulation_file)

        wavelengths = data['wavelengths']
        data = data['data']

    telluric_transmittances_wavelengths, telluric_transmittances, simulated_snr = \
        load_orange_tellurics(
            data_dir=additional_data_directory,
            wavelengths_instrument=wavelengths,
            airmasses=kwargs_['airmass'],
            resolving_power=kwargs_['new_resolving_power']
        )

    if 'telluric_transmittances_wavelengths' in kwargs_:
        kwargs_['telluric_transmittances_wavelengths'] = telluric_transmittances_wavelengths

    if 'telluric_transmittances' in kwargs_:
        kwargs_['telluric_transmittances'] = telluric_transmittances

    wavelengths, model, _ = spectral_model.modify_spectrum(
        wavelengths=wavelengths,
        spectrum=data,
        scale_function=spectral_model.scale_spectrum,
        shift_wavelengths_function=spectral_model.shift_wavelengths,
        transit_fractional_light_loss_function=spectral_model.calculate_transit_fractional_light_loss,
        convolve_function=spectral_model.convolve,
        rebin_spectrum_function=spectral_model.rebin_spectrum,
        **kwargs_
    )

    if reduce:
        model, _, _ = spectral_model.pipeline(model, **kwargs_)

    return wavelengths, model


def pipeline_sys(spectrum, **kwargs):
    """Interface with simple_pipeline.
    Modified pipeline function to use SysRem instead of the default preparing pipeline.

    Args:
        spectrum: spectrum to reduce
        **kwargs: simple_pipeline arguments

    Returns:
        The reduced spectrum, matrix, and uncertainties
    """
    # simple_pipeline interface
    if not hasattr(spectrum, 'mask'):
        spectrum = np.ma.masked_array(spectrum)

    if 'uncertainties' in kwargs:  # ensure that spectrum and uncertainties share the same mask
        if hasattr(kwargs['uncertainties'], 'mask'):
            spectrum = np.ma.masked_where(kwargs['uncertainties'].mask, spectrum)

    if np.ndim(spectrum.mask) == 0:
        spectrum.mask = np.zeros(spectrum.shape, dtype=bool)

    return preparing_pipeline_sysrem(spectrum=spectrum, full=True, **kwargs)


def pseudo_retrieval(prt_object, parameters, kps, v_rest, model, data, data_uncertainties,
                     scale, shift, use_transit_light_loss, convolve, rebin, reduce, correct_uncertainties=False):
    p = copy.deepcopy(parameters)
    m = copy.deepcopy(model)
    logls = []
    wavelengths = []
    retrieval_models = []

    if hasattr(data, 'mask'):
        data, data_uncertainties, data_mask = m.remove_mask(
            data=data,
            data_uncertainties=data_uncertainties
        )
    else:
        data_mask = fill_object(copy.deepcopy(data), False)

    def retrieval_model(prt_object_, p_):
        return m.retrieval_model_generating_function(
            prt_object=prt_object_,
            parameters=p_,
            spectrum_model=m,
            mode='transmission',
            update_parameters=True,
            telluric_transmittances=None,
            instrumental_deformations=None,
            noise_matrix=None,
            scale=scale,
            shift=shift,
            use_transit_light_loss=use_transit_light_loss,
            convolve=convolve,
            rebin=rebin,
            reduce=reduce
        )

    p['correct_uncertainties'] = correct_uncertainties
    print(f"correct uncertainties with k_sigma: {correct_uncertainties}")

    for lag in v_rest:
        p['planet_rest_frame_velocity_shift'] = lag
        logls.append([])
        wavelengths.append([])
        retrieval_models.append([])

        for kp_ in kps:
            p['planet_radial_velocity_amplitude'] = kp_

            w, s, _ = retrieval_model(prt_object, p)
            wavelengths[-1].append(w)
            retrieval_models[-1].append(s)

            logl = 0

            for i, det in enumerate(data):
                for j, data in enumerate(det):
                    logl += Data.log_likelihood(
                        model=s[i, j, ~data_mask[i, j, :]],
                        data=data,
                        uncertainties=data_uncertainties[i, j],
                        beta=1.0
                    )

            logls[-1].append(logl)

    logls = np.transpose(logls)

    return logls, retrieval_models


def validity_checks(simulated_data_model, radtrans, telluric_transmittances_wavelengths, telluric_transmittances,
                    instrumental_deformations, noise_matrix,
                    scale, shift, use_transit_light_loss, convolve, rebin,
                    filename='./validity.npz', do_pseudo_retrieval=True, save=True,
                    full=False):
    print('Initializing spectra...')
    p = copy.deepcopy(simulated_data_model.model_parameters)

    for key, value in simulated_data_model.model_parameters['imposed_mass_mixing_ratios'].items():
        p[key] = np.log10(value)

    print(' True spectrum...')
    true_wavelengths, true_spectrum, _ = simulated_data_model.retrieval_model_generating_function(
        prt_object=radtrans,
        parameters=p,
        spectrum_model=simulated_data_model,
        mode='transmission',
        update_parameters=True,
        telluric_transmittances_wavelengths=None,
        telluric_transmittances=None,
        instrumental_deformations=None,
        noise_matrix=None,
        scale=scale,
        shift=shift,
        use_transit_light_loss=use_transit_light_loss,
        convolve=convolve,
        rebin=rebin,
        reduce=False
    )

    print(' Deformed spectrum...')
    _, deformed_spectrum, _ = simulated_data_model.retrieval_model_generating_function(
        prt_object=radtrans,
        parameters=p,
        spectrum_model=simulated_data_model,
        mode='transmission',
        update_parameters=True,
        telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
        telluric_transmittances=telluric_transmittances,
        instrumental_deformations=instrumental_deformations,
        noise_matrix=None,
        scale=scale,
        shift=shift,
        use_transit_light_loss=use_transit_light_loss,
        convolve=convolve,
        rebin=rebin,
        reduce=False
    )

    print(' Reprocessed spectrum...')
    _, reprocessed_spectrum, _ = simulated_data_model.retrieval_model_generating_function(
        prt_object=radtrans,
        parameters=p,
        spectrum_model=simulated_data_model,
        mode='transmission',
        update_parameters=True,
        telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
        telluric_transmittances=telluric_transmittances,
        instrumental_deformations=instrumental_deformations,
        noise_matrix=noise_matrix,
        scale=scale,
        shift=shift,
        use_transit_light_loss=use_transit_light_loss,
        convolve=convolve,
        rebin=rebin,
        reduce=True
    )

    deformation_matrix = deformed_spectrum / true_spectrum

    if noise_matrix is None:
        noise_matrix = np.zeros(reprocessed_spectrum.shape)

    print(' Reprocessed true spectrum...')
    reprocessed_true_spectrum, reprocessed_matrix_true, _ = simulated_data_model.pipeline(
        spectrum=true_spectrum,
        wavelengths=true_wavelengths,
        **simulated_data_model.model_parameters
    )

    print(' Reprocessed deformed spectrum...')
    reprocessed_deformed_spectrum, reprocessed_matrix_deformed, _ = simulated_data_model.pipeline(
        spectrum=true_spectrum * deformation_matrix,
        wavelengths=true_wavelengths,
        **simulated_data_model.model_parameters
    )

    print(' Reprocessed noisy spectrum...')
    reprocessed_noisy_spectrum, reprocessed_matrix_noisy, _ = simulated_data_model.pipeline(
        spectrum=true_spectrum * deformation_matrix + noise_matrix,
        wavelengths=true_wavelengths,
        **simulated_data_model.model_parameters
    )

    print('Checking framework validity (noisy)...', end='')

    try:
        assert np.allclose(
            reprocessed_spectrum,
            (true_spectrum * deformation_matrix + noise_matrix) * reprocessed_matrix_noisy,
            atol=1e-10,
            rtol=1e-10
        )
        print(' OK')
    except AssertionError:
        print('Validity not respected!')

    noiseless_validity = 1 - reprocessed_true_spectrum / reprocessed_deformed_spectrum

    print('Reprocessing pipeline validity (noiseless):')
    print(f" {np.ma.mean(noiseless_validity):.3e} +/- {np.ma.std(noiseless_validity):.3e} "
          f"({np.ma.min(noiseless_validity):.3e} <= val <= {np.ma.max(noiseless_validity):.3e})\n"
          f" Size: {np.size(noiseless_validity)}")

    noisy_validity = 1 - (reprocessed_true_spectrum + noise_matrix * reprocessed_matrix_noisy) \
        / reprocessed_noisy_spectrum

    print('Reprocessing pipeline validity (noisy):')
    print(f" {np.ma.mean(noisy_validity):.3e} +/- {np.ma.std(noisy_validity):.3e} "
          f"({np.ma.min(noisy_validity):.3e} <= val <= {np.ma.max(noisy_validity):.3e})\n"
          f" Size: {np.size(noisy_validity)}")

    if do_pseudo_retrieval:
        print('Running pseudo retrieval...')
        true_log_l, retrieval_models = pseudo_retrieval(
            prt_object=radtrans,
            parameters=p,
            kps=[simulated_data_model.model_parameters['planet_radial_velocity_amplitude']],
            v_rest=[simulated_data_model.model_parameters['planet_rest_frame_velocity_shift']],
            model=simulated_data_model,
            data=reprocessed_spectrum,
            data_uncertainties=simulated_data_model.model_parameters['reduced_uncertainties'],
            scale=scale,
            shift=shift,
            use_transit_light_loss=use_transit_light_loss,
            convolve=convolve,
            rebin=rebin,
            reduce=True
        )

        # assert np.allclose(retrieval_models[0][0], reprocessed_true_spectrum, atol=0.0, rtol=1e-14)

        true_chi2 = -2 * true_log_l[0][0] / np.size(reprocessed_spectrum[~reprocessed_spectrum.mask])

        # Check Log L and chi2 when using the true set of parameter
        print(f'True log L = {true_log_l[0][0]}')
        print(f'True chi2 = {true_chi2}')
        print(f'N non-masked points = {np.size(reprocessed_spectrum[~reprocessed_spectrum.mask])}')
    else:
        print('No pseudo retrieval...')
        true_log_l = None
        true_chi2 = None

    if save:
        np.savez_compressed(
            file=filename,
            noisy_validity=noisy_validity,
            noiseless_validity=noiseless_validity,
            wavelengths=true_wavelengths,
            true_spectrum=true_spectrum,
            deformed_spectrum=deformed_spectrum,
            reprocessed_true_spectrum=reprocessed_true_spectrum,
            reprocessed_deformed_spectrum=reprocessed_deformed_spectrum,
            reprocessed_noisy_spectrum=reprocessed_noisy_spectrum,
            reprocessing_matrix_deformed=reprocessed_matrix_deformed,
            reprocessing_matrix_noisy=reprocessed_matrix_noisy,
            noise_matrix=noise_matrix,
            true_log_l=true_log_l,
            true_chi2=true_chi2
        )

    if full:
        return noisy_validity, true_log_l, true_chi2, noiseless_validity, \
               true_wavelengths, true_spectrum, deformed_spectrum, reprocessed_spectrum, reprocessed_true_spectrum, \
               reprocessed_deformed_spectrum, reprocessed_noisy_spectrum, \
               reprocessed_matrix_true, reprocessed_matrix_deformed, reprocessed_matrix_noisy
    else:
        return noisy_validity, true_log_l, true_chi2


# Main
def main(planet_name, output_directory, additional_data_directory, mode, uncertainties_mode,
         retrieval_name, retrieval_parameters,
         detector_selection_name, n_live_points, resume, tellurics_mask_threshold, mid_transit_time_range,
         add_data_offset, use_t14, use_t23, use_t1535,
         use_boxcar_tlloss, use_sysrem, use_sysrem_subtract, use_gaussian_priors,
         retrieve_mock_observations, retrieve_mock_3d, use_simulated_uncertainties, add_noise, noise_seed,
         n_transits, check,
         retrieve, archive, scale, shift, use_transit_light_loss, convolve, rebin):
    # Initialize MPI
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    mid_transit_time_jd = 58004.424877  # 58004.42319302507  # 58004.425291

    # References
    retrieved_parameters_ref = {
        'temperature': {
            'prior_parameters': [100, 4000],
            'prior_type': 'uniform',
            'figure_title': r'T',
            'figure_label': r'T (K)',
            'retrieval_name': 'tiso'
        },
        'CH4_hargreaves_main_iso': {
            'prior_parameters': [-12, 0],
            'prior_type': 'uniform',
            'figure_title': r'[CH$_4$]',
            'figure_label': r'$\log_{10}$(MMR) CH$_4$',
            'retrieval_name': 'CH4'
        },
        'CO_all_iso': {
            'prior_parameters': [-12, 0],
            'prior_type': 'uniform',
            'figure_title': r'[CO]',
            'figure_label': r'$\log_{10}$(MMR) CO',
            'retrieval_name': 'CO'
        },
        'H2O_main_iso': {
            'prior_parameters': [-12, 0],
            'prior_type': 'uniform',
            'figure_title': r'[H$_2$O]',
            'figure_label': r'$\log_{10}$(MMR) H$_2$O',
            'retrieval_name': 'H2O'
        },
        'H2S_main_iso': {
            'prior_parameters': [-12, 0],
            'prior_type': 'uniform',
            'figure_title': r'[H$_2$S]',
            'figure_label': r'$\log_{10}$(MMR) H$_2$S',
            'retrieval_name': 'H2S'
        },
        'HCN_main_iso': {
            'prior_parameters': [-12, 0],
            'prior_type': 'uniform',
            'figure_title': r'[HCN]',
            'figure_label': r'$\log_{10}$(MMR) HCN',
            'retrieval_name': 'HCN'
        },
        'NH3_main_iso': {
            'prior_parameters': [-12, 0],
            'prior_type': 'uniform',
            'figure_title': r'[NH$_3$]',
            'figure_label': r'$\log_{10}$(MMR) NH$_3$',
            'retrieval_name': 'NH3'
        },
        'mean_molar_masses_offset': {
            'prior_parameters': [-1, 10],
            'prior_type': 'uniform',
            'retrieval_name': 'mmwo'
        },  # correlated with gravity
        'log10_cloud_pressure': {
            'prior_parameters': [-10, 2],
            'prior_type': 'uniform',
            'figure_title': r'[$P_c$]',
            'figure_label': r'$\log_{10}(P_c)$ ([Pa])',
            'figure_offset': 5,  # [bar] to [Pa]
            'retrieval_name': 'Pc'
        },
        'log10_haze_factor': {
            'prior_parameters': [-3, 3],
            'prior_type': 'uniform',
            'figure_title': r'[$h_x$]',
            'figure_label': r'$\log_{10}(h_x)$',
            'retrieval_name': 'hx'
        },
        'log10_scattering_opacity_350nm': {
            'prior_parameters': [-6, 2],
            'prior_type': 'uniform',
            'figure_title': r'[$\kappa_0$]',
            'figure_label': r'$\log_{10}(\kappa_0)$',
            'retrieval_name': 'k0'
        },
        'scattering_opacity_coefficient': {
            'prior_parameters': [-12, 1],
            'prior_type': 'uniform',
            'figure_title': r'$\gamma$',
            'figure_label': r'$\gamma$',
            'retrieval_name': 'gams'
        },
        'planet_radius': {
            'prior_parameters': np.array([0.8, 1.25]),
            'prior_type': 'uniform',
            'figure_title': r'$R_p$',
            'figure_label': r'$R_p$ (km)',
            'figure_coefficient': 1e-5,
            'retrieval_name': 'Rp'
        },
        'log10_planet_surface_gravity': {
            'prior_parameters': [2.5, 4.0],
            'prior_type': 'uniform',
            'figure_title': r'$[g]$',
            'figure_label': r'$\log_{10}(g)$ ([cm$\cdot$s$^{-2}$])',
            'retrieval_name': 'g'
        },
        'planet_radial_velocity_amplitude': {
            'prior_parameters': np.array([0.4589, 1.6388]),  # Kp must be close to the true value to help the retrieval
            'prior_type': 'uniform',
            'figure_title': r'$K_p$',
            'figure_label': r'$K_p$ (km$\cdot$s$^{-1}$)',
            'figure_coefficient': 1e-5,
            'retrieval_name': 'Kp'
        },
        'planet_rest_frame_velocity_shift': {
            'prior_parameters': [-20e5, 20e5],
            'prior_type': 'uniform',
            'figure_title': r'$V_\mathrm{rest}$',
            'figure_label': r'$V_\mathrm{rest}$ (km$\cdot$s$^{-1}$)',
            'figure_coefficient': 1e-5,
            'retrieval_name': 'V0'
        },
        'new_resolving_power': {
            'prior_parameters': [1e3, 1e5],
            'prior_type': 'uniform',
            'figure_name': 'Resolving power',
            'figure_title': r'$\mathcal{R}_C$',
            'figure_label': 'Resolving power',
            'retrieval_name': 'R'
        },
        'mid_transit_time': {
            'prior_parameters': [-mid_transit_time_range, mid_transit_time_range],
            'prior_type': 'uniform',
            'figure_title': r'$T_0$',
            'figure_label': r'$T_0$ (s)',
            'figure_offset': - (mid_transit_time_jd % 1 * cst.s_cst.day),
            'retrieval_name': 'T0'
        },
        'beta': {
            'prior_parameters': [1, 1e2],
            'prior_type': 'uniform',
            'figure_title': r'$\beta$',
            'figure_label': r'$\beta$',
            'retrieval_name': 'beta'
        },
        'log10_beta': {
            'prior_parameters': [-15, 2],
            'prior_type': 'uniform',
            'figure_title': r'[$\beta$]',
            'figure_label': r'$\log_{10}(\beta)$',
            'retrieval_name': 'lbeta'
        }
    }

    order_selection_ref = {
        'all': np.arange(0, 56),
        'alex4': np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
             33, 34, 35, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]),
        'test2': np.array([46, 47]),
        'test1': np.array([46]),
        'test0': np.array([30])
    }

    # Initialization
    if rank == 0:
        t_start = time.time()
    else:
        t_start = 0.0

    # Gaussian priors
    if use_gaussian_priors:
        retrieved_parameters_ref['mid_transit_time']['prior_type'] = 'gaussian'
        retrieved_parameters_ref['mid_transit_time']['prior_parameters'] = [
            mid_transit_time_jd % 1 * cst.s_cst.day,
            12.53
        ]

        retrieved_parameters_ref['log10_planet_surface_gravity']['prior_type'] = 'gaussian'
        retrieved_parameters_ref['log10_planet_surface_gravity']['prior_parameters'] = [
            3.34,
            0.03
        ]

        retrieved_parameters_ref['planet_radial_velocity_amplitude']['prior_type'] = 'gaussian'
        retrieved_parameters_ref['planet_radial_velocity_amplitude']['prior_parameters'] = [
            152.1e5,
            2.9e5
        ]

    # Retrieved parameters
    retrieved_parameters = {}

    if uncertainties_mode == "retrieve" and 'beta' not in retrieval_parameters:
        print(f"Asked for 'retrieve' uncertainties mode without setting beta as a retrieved parameter; "
              f"adding 'beta' to retrieval_parameters parameters")
        retrieval_parameters.append('beta')
    elif uncertainties_mode == "retrieve_add" and 'log10_beta' not in retrieval_parameters:
        print(f"Asked for 'retrieve_add' uncertainties mode without setting beta as a retrieved parameter; "
              f"adding 'log10_beta' to retrieval_parameters parameters")
        retrieval_parameters.append('log10_beta')
    elif (uncertainties_mode != "retrieve" and uncertainties_mode != "retrieve_add") \
            and 'beta' in retrieval_parameters:
        raise KeyError(f"'beta' should not be retrieved if uncertainties mode is not set to "
                       f"'retrieve' ore 'retrieve_add'"
                       f"(was '{uncertainties_mode}')")

    for parameter in retrieval_parameters:
        if parameter not in retrieved_parameters_ref:
            raise KeyError(f"parameter '{parameter}' was not initialized")
        else:
            retrieved_parameters[parameter] = retrieved_parameters_ref[parameter]

    # Order selection
    order_selection = order_selection_ref[detector_selection_name]

    # Retrieval name
    if retrieval_name != '':
        retrieval_name = '_' + retrieval_name

    for parameter_dict in retrieved_parameters.values():
        retrieval_name += f"_{parameter_dict['retrieval_name']}"

    if use_gaussian_priors:
        retrieval_name += '_gp'

    if uncertainties_mode == 'optimize':
        retrieval_name += '_opu'

    retrieval_name += f'_tmt{tellurics_mask_threshold:.2f}'

    if use_t14:
        retrieval_name += f'_t14'
    else:
        retrieval_name += f'_t0r{mid_transit_time_range:.0f}'

    retrieval_name += f'_{detector_selection_name}'

    if add_data_offset:
        retrieval_name += f'_adddoff'

    if use_boxcar_tlloss:
        retrieval_name += f'_bxctlloss'

    if use_t23:
        if use_t1535:
            retrieval_name += f'_t1535'

        retrieval_name += f'_t23'

    if retrieve_mock_observations:
        if retrieve_mock_3d:
            retrieval_name += '_sim3d'
        else:
            retrieval_name += '_sim'

        if add_noise:
            if noise_seed < 0:
                noise_seed = None

            retrieval_name += f'n{noise_seed}'

    if use_simulated_uncertainties:
        retrieval_name += '_su'

    if use_sysrem:
        retrieval_name += '_sys'

        if use_sysrem_subtract:
            retrieval_name += '-sub'
        else:
            retrieval_name += '-div'

    if not use_transit_light_loss:
        retrieval_name += '_notll'

    retrieval_name += '_c817'  # script version
    retrieval_name = f"{planet_name.replace(' ', '_')}_{mode}{retrieval_name}_{n_live_points}lp"

    # Planet
    planet = Planet.get(planet_name)

    # Overriding to Rosenthal et al. 2021
    planet.star_radius = 0.78271930158600000 * cst.r_sun
    planet.star_radius_error_upper = +0.01396094224705000 * cst.r_sun
    planet.star_radius_error_lower = -0.01396094224705000 * cst.r_sun

    # Others
    resolving_power = 8.04e4  # CARMENES resolving power
    retrieval_directory = os.path.join(output_directory, retrieval_name)

    # Load data and instantiate model
    if rank == 0:  # only use 1 MPI rank to prevent competing loading
        print(f'Setup for {retrieval_name}')
        print('Loading data...')

        wavelengths_instrument, observed_spectra, instrument_snr, uncertainties, orbital_phases, airmasses, \
            barycentric_velocities, times, mid_transit_time = load_carmenes_data(
                directory=os.path.join(additional_data_directory, 'carmenes', 'hd_189733_b'),
                mid_transit_jd=mid_transit_time_jd
            )

        instrumental_deformations, telluric_transmittances_wavelengths, telluric_transmittances, simulated_snr, \
            offset = \
            load_additional_data(
                data_dir=additional_data_directory,
                wavelengths_instrument=wavelengths_instrument,
                airmasses=airmasses,
                resolving_power=resolving_power,
                load_offset=add_data_offset
            )

        if add_data_offset:
            print("Adding offset...")
            observed_spectra = np.moveaxis(observed_spectra, -1, 0) - offset

            if np.ndim(observed_spectra.mask) == 0:
                observed_spectra.mask = np.zeros(observed_spectra.shape, dtype=bool)

            observed_spectra = np.ma.masked_where(np.less_equal(observed_spectra, -1.1 * offset), observed_spectra)
            observed_spectra = np.moveaxis(observed_spectra, 0, -1)

            uncertainties = np.ma.masked_where(observed_spectra.mask, uncertainties)

        if use_t23:
            print(f"Using full transit time (T23), not total transit time (T14)")
            planet_transit_duration = planet.calculate_full_transit_duration(
                total_transit_duration=planet.transit_duration,
                planet_radius=planet.radius,
                star_radius=planet.star_radius,
                impact_parameter=planet.calculate_impact_parameter(
                    planet_orbit_semi_major_axis=planet.orbit_semi_major_axis,
                    planet_orbital_inclination=planet.orbital_inclination,
                    star_radius=planet.star_radius
                )
            )

            if use_t1535:
                print(f"Adding exposures of half-eclipses")
                planet_transit_duration += (planet.transit_duration - planet_transit_duration) / 2
        else:
            planet_transit_duration = planet.transit_duration

        if not use_t14:
            if not use_gaussian_priors:
                planet_transit_duration += np.max(retrieved_parameters_ref['mid_transit_time']['prior_parameters']) \
                                           - np.min(retrieved_parameters_ref['mid_transit_time']['prior_parameters'])
            else:
                planet_transit_duration += 2 * 5 * retrieved_parameters_ref['mid_transit_time']['prior_parameters'][1]

        wavelengths_instrument, observed_spectra, uncertainties, instrument_snr, instrumental_deformations, \
            simulated_snr, orbital_phases, airmasses, \
            barycentric_velocities, times = \
            data_selection(
                wavelengths_instrument=wavelengths_instrument,
                observed_spectra=observed_spectra,
                uncertainties=uncertainties,
                instrument_snr=instrument_snr,
                instrumental_deformations=instrumental_deformations,
                simulated_snr=simulated_snr,
                times=times,
                mid_transit_time=mid_transit_time,
                transit_duration=planet_transit_duration,
                orbital_phases=orbital_phases,
                airmasses=airmasses,
                barycentric_velocities=barycentric_velocities,
                detector_selection=order_selection,
                n_transits=n_transits,
                use_t23=use_t23,
                use_t1535=use_t1535
            )

        simulated_uncertainties = np.moveaxis(
            np.moveaxis(simulated_snr, 2, 0) / np.mean(simulated_snr, axis=2) * np.mean(uncertainties, axis=2),
            0,
            2
        )

        if use_simulated_uncertainties:
            model_uncertainties = simulated_uncertainties
        else:
            model_uncertainties = copy.deepcopy(uncertainties) * 1  # TODO * 1

        data_shape = observed_spectra.shape

        if add_noise:
            print(f'Generating noise (seed={noise_seed})...')
            noise_matrix = np.random.default_rng(seed=noise_seed).normal(loc=0, scale=uncertainties, size=data_shape)
        else:
            noise_matrix = None

        # Mock_observations
        print('Initializing model...')
        spectral_model = SpectralModel(
            # Radtrans object parameters
            pressures=np.logspace(-10, 2, 100),  # bar
            line_species=[
                'CH4_hargreaves_main_iso',
                'CO_all_iso',
                'H2O_main_iso',
                'H2S_main_iso',
                'HCN_main_iso',
                'NH3_main_iso'
            ],
            rayleigh_species=['H2', 'He'],
            continuum_opacities=['H2-H2', 'H2-He'],
            line_opacity_mode='lbl',
            line_by_line_opacity_sampling=4,
            # Temperature profile parameters
            temperature_profile_mode='isothermal',
            temperature=planet.equilibrium_temperature,  # K
            # Chemical parameters
            use_equilibrium_chemistry=False,
            imposed_mass_mixing_ratios={
                'CH4_hargreaves_main_iso': 1e-12,
                'CO_all_iso': 1e-12,
                'H2O_main_iso': 1e-12,  # TODO 1e-12
                'H2S_main_iso': 1e-12,
                'HCN_main_iso': 1e-12,
                'NH3_main_iso': 1e-12
            },
            fill_atmosphere=True,
            # Transmission spectrum parameters (radtrans.calc_transm)
            planet_radius=planet.radius,  # cm
            planet_surface_gravity=planet.reference_gravity,  # cm.s-2
            reference_pressure=1e-2,  # bar
            cloud_pressure=1e2,
            haze_factor=1,
            scattering_opacity_350nm=1e-6,
            scattering_opacity_coefficient=-12,
            # Instrument parameters
            new_resolving_power=8.04e4,
            output_wavelengths=wavelengths_instrument,  # um
            # Scaling parameters
            star_radius=planet.star_radius,  # cm
            # Orbital parameters
            times=times,
            star_mass=planet.star_mass,  # g
            orbit_semi_major_axis=planet.orbit_semi_major_axis,  # cm
            mid_transit_time=mid_transit_time,  # TODO +0
            orbital_period=planet.orbital_period,
            orbital_inclination=planet.orbital_inclination,
            transit_duration=planet.transit_duration,
            system_observer_radial_velocities=planet.star_radial_velocity - barycentric_velocities * 1e5,  # cm.s-1
            planet_rest_frame_velocity_shift=0.0,  # cm.s-1
            planet_orbital_inclination=planet.orbital_inclination,
            # Reprocessing parameters
            uncertainties=model_uncertainties,
            airmass=airmasses,
            tellurics_mask_threshold=tellurics_mask_threshold,
            polynomial_fit_degree=2,
            apply_throughput_removal=True,
            apply_telluric_lines_removal=True,
            # Special parameters
            mean_molar_masses_offset=0.5,
            rebin_range_margin_power=3,
            constance_tolerance=1e300,  # force constant convolve
            detector_selection=order_selection,
            add_data_offset=add_data_offset,
            use_boxcar_tlloss=use_boxcar_tlloss
        )

        if 'planet_radial_velocity_amplitude' in retrieved_parameters and not use_gaussian_priors:
            retrieved_parameters['planet_radial_velocity_amplitude']['prior_parameters'] *= \
                spectral_model.model_parameters['planet_radial_velocity_amplitude']

        if 'planet_radius' in retrieved_parameters:
            retrieved_parameters['planet_radius']['prior_parameters'] *= \
                spectral_model.model_parameters['planet_radius']

        if 'mid_transit_time' in retrieved_parameters and not use_gaussian_priors:
            retrieved_parameters['mid_transit_time']['prior_parameters'] += \
                spectral_model.model_parameters['mid_transit_time']
            mid_transit_time_range = retrieved_parameters['mid_transit_time']['prior_parameters']
        else:
            mid_transit_time_range = [0, 0]

        retrieval_velocities = spectral_model.get_retrieval_velocities(
            planet_radial_velocity_amplitude_range=retrieved_parameters[
                'planet_radial_velocity_amplitude']['prior_parameters'],
            planet_rest_frame_velocity_shift_range=np.array([retrieved_parameters[
                                                                 'planet_rest_frame_velocity_shift'][
                                                                 'prior_parameters'][0],
                                                             retrieved_parameters[
                                                                 'planet_rest_frame_velocity_shift'][
                                                                 'prior_parameters'][1]
                                                             ]),
            mid_transit_times_range=mid_transit_time_range
        )

        spectral_model.wavelengths_boundaries = spectral_model.get_optimal_wavelength_boundaries(
            relative_velocities=retrieval_velocities
        )

        if use_sysrem:
            print("Using SysRem pipeline")
            spectral_model.pipeline = pipeline_sys
            spectral_model.model_parameters['preparing'] = 'SysRem'
            spectral_model.model_parameters['n_iterations_max'] = 15
            spectral_model.model_parameters['convergence_criterion'] = -1
            # spectral_model.model_parameters['verbose'] = True

            if use_sysrem_subtract:
                print("Using SysRem subtract")
                spectral_model.model_parameters['subtract'] = True
            else:
                print("Using SysRem divide")
                spectral_model.model_parameters['subtract'] = False
        else:
            print("Using polyfit pipeline")
            spectral_model.model_parameters['preparing'] = 'Polyfit'

        if use_boxcar_tlloss:
            spectral_model.calculate_transit_fractional_light_loss = calculate_transit_fractional_light_loss_box_car

    else:
        print(f"Rank {rank} waiting for main process to finish...")

        spectral_model = None
        telluric_transmittances_wavelengths = None
        telluric_transmittances = None
        instrumental_deformations = None
        noise_matrix = None
        retrieved_parameters = None
        observed_spectra = None
        wavelengths_instrument = None

    # Broadcast loaded data and instantiated model
    if rank == 0:
        print('Broadcasting model and data...')

    comm.barrier()

    spectral_model = comm.bcast(spectral_model, root=0)
    telluric_transmittances_wavelengths = comm.bcast(telluric_transmittances_wavelengths, root=0)
    telluric_transmittances = comm.bcast(telluric_transmittances, root=0)
    instrumental_deformations = comm.bcast(instrumental_deformations, root=0)
    noise_matrix = comm.bcast(noise_matrix, root=0)
    retrieved_parameters = comm.bcast(retrieved_parameters, root=0)
    observed_spectra = comm.bcast(observed_spectra, root=0)
    wavelengths_instrument = comm.bcast(wavelengths_instrument, root=0)

    if rank == 0:
        print("Broadcasting done!")

    comm.barrier()

    # Instantiate Radtrans (Radtrans objects are too large to be broadcasted)
    radtrans = spectral_model.get_radtrans()

    comm.barrier()

    # Check model validity
    if rank == 0 and check:
        print('Validity checks...')
        validity, true_log_l, true_chi2 = validity_checks(
            simulated_data_model=copy.deepcopy(spectral_model),
            radtrans=radtrans,
            telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
            telluric_transmittances=telluric_transmittances,
            instrumental_deformations=instrumental_deformations,
            noise_matrix=noise_matrix,
            scale=scale,
            shift=shift,
            use_transit_light_loss=use_transit_light_loss,
            convolve=convolve,
            rebin=rebin
        )

        spectral_model.model_parameters['pipeline_validity'] = validity
        spectral_model.model_parameters['true_log_l'] = true_log_l
        spectral_model.model_parameters['true_chi2'] = true_chi2

    # Initialize simulated data
    if retrieve_mock_observations:
        if rank == 0:
            if retrieve_mock_3d:
                print("Reprocessing simulated 3D data...")
            else:
                print("Initializing simulated data...")

        comm.barrier()

        # Copy spectral model to not mess up the fixed parameters values
        simulated_data_model = copy.deepcopy(spectral_model)

        simulated_data_model.model_parameters['cloud_pressure'] = 1
        simulated_data_model.model_parameters['scattering_opacity_350nm'] = 1e-3
        simulated_data_model.model_parameters['scattering_opacity_coefficient'] = -4

        if retrieve_mock_3d:
            wavelengths_instrument, simulated_data = get_orange_simulation_model(
                directory=os.path.join(additional_data_directory, 'carmenes', 'hd_189733_b', 'simu_orange'),
                additional_data_directory=additional_data_directory,
                spectral_model=simulated_data_model,
                base_name='range',
                mode=mode,
                telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
                telluric_transmittances=telluric_transmittances,
                instrumental_deformations=instrumental_deformations,
                noise_matrix=noise_matrix,
                scale=scale,
                shift=shift,
                use_transit_light_loss=use_transit_light_loss,
                convolve=convolve,
                rebin=rebin,
                reduce=False,
                rank=rank,
                **simulated_data_model.model_parameters
            )

            prepared_data, preparing_matrix, prepared_data_uncertainties = simulated_data_model.pipeline(
                simulated_data,
                wavelengths=wavelengths_instrument,
                **simulated_data_model.model_parameters
            )

            # Ensure that everything is correctly masked, then prepare with the new mask
            simulated_data = np.ma.masked_where(prepared_data.mask, simulated_data)

            prepared_data, preparing_matrix, prepared_data_uncertainties = simulated_data_model.pipeline(
                simulated_data,
                wavelengths=wavelengths_instrument,
                **simulated_data_model.model_parameters
            )

            # Save simulated model for reproducibility
            if rank == 0:
                filename3d = os.path.join(retrieval_directory, f'model_3d.npz')
                print(f"Saving 3D model in file {filename3d}...")

                if not os.path.isdir(retrieval_directory):
                    os.mkdir(retrieval_directory)

                np.savez_compressed(
                    filename3d,
                    wavelengths_instrument=wavelengths_instrument,
                    reprocessed_data=prepared_data,
                    mask=prepared_data.mask
                )
        else:
            simulated_data_model.model_parameters['imposed_mass_mixing_ratios'] = {
                'CH4_hargreaves_main_iso': 3.4e-5,
                'CO_all_iso': 1.8e-2,
                'H2O_main_iso': 5.4e-3,
                'H2S_main_iso': 1.0e-3,
                'HCN_main_iso': 2.7e-7,
                'NH3_main_iso': 7.9e-6
            }

            wavelengths_instrument, simulated_data = simulated_data_model.get_spectrum_model(
                radtrans=radtrans,
                mode=mode,
                update_parameters=True,
                telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
                telluric_transmittances=telluric_transmittances,
                instrumental_deformations=instrumental_deformations,
                noise_matrix=noise_matrix,
                scale=scale,
                shift=shift,
                use_transit_light_loss=use_transit_light_loss,
                convolve=convolve,
                rebin=rebin,
                reduce=False
            )

            prepared_data, preparing_matrix, prepared_data_uncertainties = simulated_data_model.pipeline(
                simulated_data,
                wavelengths=wavelengths_instrument,
                **simulated_data_model.model_parameters
            )

            # Ensure that everything is correctly masked, then prepare with the new mask
            simulated_data = np.ma.masked_where(prepared_data.mask, simulated_data)

            prepared_data, preparing_matrix, prepared_data_uncertainties = simulated_data_model.pipeline(
                simulated_data,
                wavelengths=wavelengths_instrument,
                **simulated_data_model.model_parameters
            )

            # Save simulated model for reproducibility
            if rank == 0:
                print(f"Saving simulated data...")
                if not os.path.isdir(os.path.abspath(retrieval_directory)):
                    os.mkdir(os.path.abspath(retrieval_directory))

                np.savez_compressed(
                    file=os.path.join(os.path.abspath(retrieval_directory), 'simulated_data.npz'),
                    simulated_data=simulated_data,
                    reprocessed_data=prepared_data,
                    mask=prepared_data.mask,
                    wavelengths=wavelengths_instrument,
                    times=simulated_data_model.model_parameters['times'],
                    telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
                    telluric_transmittances=telluric_transmittances,
                    instrumental_deformations=instrumental_deformations,
                    noise_matrix=noise_matrix
                )

        plot_true_values = True
    else:
        prepared_data, preparing_matrix, prepared_data_uncertainties = spectral_model.pipeline(
            observed_spectra,
            wavelengths=wavelengths_instrument,
            **spectral_model.model_parameters
        )
        plot_true_values = False
        simulated_data_model = None

    # Ensure that all 0s uncertainties are masked to prevent division by 0
    prepared_data_uncertainties = np.ma.masked_less_equal(prepared_data_uncertainties, 0)
    prepared_data = np.ma.masked_where(prepared_data_uncertainties.mask, prepared_data)

    # Save model for reproducibility
    if rank == 0:
        print("Saving model...")
        if not os.path.isdir(retrieval_directory):
            os.mkdir(retrieval_directory)

        spectral_model.model_parameters['n_transits'] = n_transits
        spectral_model.save(os.path.join(retrieval_directory, 'base_model.h5'))

        if simulated_data_model is not None:
            simulated_data_model.save(os.path.join(retrieval_directory, 'simulated_data_model.h5'))

    # Initialize retrieval
    if rank == 0:
        print("Initializing retrieval...")

    comm.barrier()

    retrieval_model = copy.deepcopy(spectral_model)

    retrieval = retrieval_model.init_retrieval(
        radtrans=radtrans,
        data=prepared_data,
        data_wavelengths=wavelengths_instrument,
        data_uncertainties=prepared_data_uncertainties,
        retrieval_directory=retrieval_directory,
        retrieved_parameters=retrieved_parameters,
        retrieval_name=retrieval_name,
        mode=mode,
        uncertainties_mode=uncertainties_mode,
        update_parameters=True,
        telluric_transmittances=None,
        instrumental_deformations=None,
        scale=scale,
        shift=shift,
        use_transit_light_loss=use_transit_light_loss,
        convolve=convolve,
        rebin=rebin,
        reduce=True,
        run_mode='retrieval',
        scattering=True
    )

    # Retrieval
    if retrieve:
        # Save retrieval setup for reproducibility
        if rank == 0:
            save = True

            print("Saving retrieved parameters...")
            np.savez_compressed(
                os.path.join(os.path.abspath(retrieval_directory), 'retrieved_parameters.npz'),
                data=prepared_data,
                data_wavelengths=wavelengths_instrument,
                data_uncertainties=prepared_data_uncertainties,
                data_mask=prepared_data.mask,
                raw_data=observed_spectra,
                raw_data_mask=observed_spectra.mask,
                **retrieved_parameters
            )
        else:
            save = False

        # Run retrieval
        spectral_model.run_retrieval(
            retrieval=retrieval,
            n_live_points=n_live_points,
            resume=resume,
            save=save
        )

        if rank == 0:
            # Plot quick look contour corner
            sample_dict, parameter_dict = retrieval.get_samples(
                output_dir=retrieval_directory + os.path.sep,
                ret_names=[retrieval_name]
            )

            n_param = len(parameter_dict[retrieval_name])
            parameter_plot_indices = {retrieval_name: np.arange(0, n_param)}

            if plot_true_values:
                true_values = {retrieval_name: []}

                for p in parameter_dict[retrieval_name]:
                    if p not in spectral_model.model_parameters and 'log10_' not in p:
                        true_values[retrieval_name].append(
                            np.mean(np.log10(spectral_model.model_parameters['imposed_mass_mixing_ratios'][p]))
                        )
                    elif p not in spectral_model.model_parameters and 'log10_' in p:
                        p = p.split('log10_', 1)[1]
                        true_values[retrieval_name].append(np.mean(np.log10(spectral_model.model_parameters[p])))
                    else:
                        true_values[retrieval_name].append(np.mean(spectral_model.model_parameters[p]))
            else:
                true_values = None

            contour_corner(
                sample_dict, parameter_dict, os.path.join(retrieval_directory, f'corner_{retrieval_name}.png'),
                parameter_plot_indices=parameter_plot_indices,
                true_values=true_values, prt_plot_style=False
            )

            # Archive retrieval directory
            if archive:
                print(f"Creating archive '{retrieval_directory}.tar.gz'...")
                root_dir, base_dir = retrieval_directory.rsplit(os.sep, 1)
                shutil.make_archive(retrieval_directory, 'gztar', root_dir, base_dir)

    comm.barrier()

    if rank == 0:
        run_time = np.array([time.time() - t_start])
        np.savetxt(os.path.join(retrieval_directory, 'run_time.dat'), run_time, header=f"{retrieval_name} run time (s)")

    comm.barrier()


if __name__ == '__main__':
    t0 = time.time()

    in_args = parser.parse_args()

    print(f'rm: {in_args.retrieve_mock_observations}')
    print(f'rm3: {in_args.retrieve_3d_mock_observations}')
    print(f'a: {in_args.no_archive}')
    print(f'convolve: {in_args.no_convolve}')
    print(f'tll: {in_args.no_tlloss}')
    print(f'sys: {in_args.use_sysrem}, subtract: {in_args.use_sysrem_subtract}')
    print(f'u mode: {in_args.uncertainties_mode}')

    main(
        planet_name=in_args.planet_name,
        output_directory=in_args.output_directory,
        additional_data_directory=in_args.additional_data_directory,
        mode=in_args.mode,
        uncertainties_mode=in_args.uncertainties_mode,
        retrieval_name=in_args.retrieval_name,
        retrieval_parameters=in_args.retrieval_parameters,
        detector_selection_name=in_args.detector_selection_name,
        n_live_points=in_args.n_live_points,
        resume=in_args.resume,
        tellurics_mask_threshold=in_args.tellurics_mask_threshold,
        mid_transit_time_range=in_args.mid_transit_time_range,
        add_data_offset=in_args.add_data_offset,
        use_t14=in_args.use_t14,
        use_t23=in_args.use_t23,
        use_t1535=in_args.use_t1535,
        use_boxcar_tlloss=in_args.use_boxcar_tlloss,
        use_sysrem=in_args.use_sysrem,
        use_sysrem_subtract=in_args.use_sysrem_subtract,
        use_gaussian_priors=in_args.use_gaussian_priors,
        retrieve_mock_observations=in_args.retrieve_mock_observations,
        retrieve_mock_3d=in_args.retrieve_3d_mock_observations,
        use_simulated_uncertainties=in_args.use_simulated_uncertainties,
        add_noise=in_args.add_noise,
        noise_seed=in_args.noise_seed,
        n_transits=in_args.n_transits,
        check=in_args.check,
        retrieve=in_args.no_retrieve,
        archive=in_args.no_archive,
        scale=in_args.no_scale,
        shift=in_args.no_shift,
        use_transit_light_loss=in_args.no_tlloss,
        convolve=in_args.no_convolve,
        rebin=in_args.no_rebin
    )

    print(f"Done in {time.time() - t0} s")
