"""
Run with:
    mpiexec -n N --use-hwthread-cpus python3 _test_high_resolution.py
N is the number of processes.
Try:
    sudo mpiexec -n N --allow-run-as-root ...
If for some reason the script crashes.
"""
import argparse
import copy
import json
import os
import pathlib
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

import petitRADTRANS.nat_cst as nc
from petitRADTRANS.cli.eso_skycalc_cli import get_tellurics_npz
from petitRADTRANS.containers.planet import Planet
from petitRADTRANS.containers.spectral_model import SpectralModel
from petitRADTRANS.retrieval.data import Data
from petitRADTRANS.retrieval.plotting import contour_corner
from petitRADTRANS.retrieval.reprocessing import reprocessing_pipeline
from petitRADTRANS.retrieval.retrieval import Retrieval
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
        default=0.5,
        help='telluric mask threshold for reprocessing'
    )

    _parser.add_argument(
        '--retrieve-mock-observations',
        action='store_true',
        help='if activated, retrieve mock observations instead of the real data'
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
        '--no-retrieval',
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

    return _parser


parser = _init_parser()


def get_orange_simulation_model(directory, base_name,
                                additional_data_directory,
                                extension='dat', reduce=False, **kwargs):
    kwargs_ = copy.deepcopy(kwargs)
    orange_simulation_file = os.path.join(directory, 'orange_hd_189733_b_transmission.npz')

    if not os.path.isfile(orange_simulation_file):
        wavelengths, data = load_orange_simulation_dat(
            directory=directory,
            base_name=base_name,
            extension=extension
        )

        np.savez_compressed(file=orange_simulation_file, wavelengths=wavelengths, data=data)
    else:
        data = np.load(orange_simulation_file)

        wavelengths = data['wavelengths']
        data = data['data']

    telluric_transmittances_wavelengths, telluric_transmittances, simulated_snr = \
        load_orange_tellurics(
            data_dir=additional_data_directory,
            wavelengths_instrument=kwargs_['output_wavelengths'],
            airmasses=kwargs_['airmass'],
            resolving_power=kwargs_['new_resolving_power']
        )

    if 'telluric_transmittances_wavelengths' in kwargs_:
        kwargs_['telluric_transmittances_wavelengths'] = telluric_transmittances_wavelengths

    if 'telluric_transmittances' in kwargs_:
        kwargs_['telluric_transmittances'] = telluric_transmittances

    wavelengths, model, _ = SpectralModel.modify_spectrum(
        wavelengths=wavelengths,
        spectrum=data,
        **kwargs_
    )

    if reduce:
        model, _, _ = SpectralModel.pipeline(model, **kwargs_)

    return wavelengths, model


def load_additional_data(data_dir, wavelengths_instrument, airmasses, times, resolving_power, simulate_snr=True):
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
    variable_throughput_file = os.path.join(data_dir, 'metis', 'brogi_crires_test', 'algn.npy')

    print(f"Loading variable throughput from file '{variable_throughput_file}'...")
    variable_throughput = load_variable_throughput_brogi(
        variable_throughput_file, times.size, wavelengths_instrument.shape[-1]
    )

    variable_throughput = np.tile(variable_throughput, (wavelengths_instrument.shape[0], 1, 1))

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
            output_wavelengths=wavelengths_instrument[:, 0, :]
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

    # Add airmass
    print('Adding airmass effect to transmittances...')
    telluric_transmittance = np.tile(telluric_transmittance_0, (airmasses.size, 1))
    wavelengths_telluric = np.tile(wavelengths_telluric, (airmasses.size, 1))

    telluric_transmittance = np.ma.masked_less(
        np.moveaxis(np.exp(np.ma.log(np.moveaxis(telluric_transmittance, 0, 1)) * airmasses), 1, 0),
        0.0
    ).filled(0.0)

    return variable_throughput,  wavelengths_telluric, telluric_transmittance, simulated_snr


def load_orange_simulation_dat(directory, base_name, extension='dat'):
    wavelengths = np.array([])
    data = np.array([])

    i = 0
    filename = f"{os.path.join(os.path.abspath(directory), base_name)}_{i:02d}.{extension}"

    while os.path.isfile(filename):
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

    # Add airmass
    print('Adding airmass effect to transmittances...')
    telluric_transmittance = np.tile(telluric_transmittance_0, (airmasses.size, 1))
    wavelengths_telluric = np.tile(wavelengths_telluric, (airmasses.size, 1))

    telluric_transmittance = np.ma.masked_less(
        np.moveaxis(np.exp(np.ma.log(np.moveaxis(telluric_transmittance, 0, 1)) * airmasses), 1, 0),
        0.0
    ).filled(0.0)

    return wavelengths_telluric, telluric_transmittance, simulated_snr


def load_variable_throughput_brogi(file, times_size, wavelengths_size):
    variable_throughput = np.load(file)

    variable_throughput = np.max(variable_throughput[0], axis=1)
    variable_throughput = variable_throughput / np.max(variable_throughput)

    xp = np.linspace(0, 1, np.size(variable_throughput))
    x = np.linspace(0, 1, times_size)
    variable_throughput = np.interp(x, xp, variable_throughput)

    return np.tile(variable_throughput, (wavelengths_size, 1)).T


def pseudo_retrieval(prt_object, parameters, kps, v_rest, model, data, data_uncertainties,
                     scale, shift, convolve, rebin, reduce, correct_uncertainties=False):
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
            convolve=convolve,
            rebin=rebin,
            reduce=reduce
        )

    p['correct_uncertainties'] = correct_uncertainties
    print(f"c u = {correct_uncertainties}")

    for lag in v_rest:
        p['planet_rest_frame_velocity_shift'] = lag
        logls.append([])
        wavelengths.append([])
        retrieval_models.append([])

        for kp_ in kps:
            p['planet_radial_velocity_amplitude'] = kp_

            w, s = retrieval_model(prt_object, p)
            wavelengths[-1].append(w)
            retrieval_models[-1].append(s)

            logl = 0

            for i, det in enumerate(data):
                for j, data in enumerate(det):
                    logl += Data.log_likelihood_gibson(
                        model=s[i, j, ~data_mask[i, j, :]],
                        data=data,
                        uncertainties=data_uncertainties[i, j],
                        alpha=1.0,
                        beta=1.0
                    )

            logls[-1].append(logl)

    logls = np.transpose(logls)

    return logls, retrieval_models


def validity_checks(simulated_data_model, radtrans, telluric_transmittances_wavelengths, telluric_transmittances,
                    instrumental_deformations, noise_matrix,
                    scale, shift, convolve, rebin, filename='./validity.npz', do_pseudo_retrieval=True, save=True,
                    full=False):
    print('Initializing spectra...')
    p = copy.deepcopy(simulated_data_model.model_parameters)

    for key, value in simulated_data_model.model_parameters['imposed_mass_mixing_ratios'].items():
        p[key] = np.log10(value)

    print(' True spectrum...')
    true_wavelengths, true_spectrum = simulated_data_model.retrieval_model_generating_function(
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
        convolve=convolve,
        rebin=rebin,
        reduce=False
    )

    print(' Deformed spectrum...')
    _, deformed_spectrum = simulated_data_model.retrieval_model_generating_function(
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
        convolve=convolve,
        rebin=rebin,
        reduce=False
    )

    print(' Reprocessed spectrum...')
    _, reprocessed_spectrum = simulated_data_model.retrieval_model_generating_function(
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
        airmass=simulated_data_model.model_parameters['airmass'],
        uncertainties=simulated_data_model.model_parameters['uncertainties'],
        apply_throughput_removal=simulated_data_model.model_parameters['apply_throughput_removal'],
        apply_telluric_lines_removal=simulated_data_model.model_parameters['apply_telluric_lines_removal'],
        polynomial_fit_degree=simulated_data_model.model_parameters['polynomial_fit_degree'],
        tellurics_mask_threshold=simulated_data_model.model_parameters['tellurics_mask_threshold']
    )

    print(' Reprocessed deformed spectrum...')
    reprocessed_deformed_spectrum, reprocessed_matrix_deformed, _ = simulated_data_model.pipeline(
        spectrum=true_spectrum * deformation_matrix,
        wavelengths=true_wavelengths,
        airmass=simulated_data_model.model_parameters['airmass'],
        uncertainties=simulated_data_model.model_parameters['uncertainties'],
        apply_throughput_removal=simulated_data_model.model_parameters['apply_throughput_removal'],
        apply_telluric_lines_removal=simulated_data_model.model_parameters['apply_telluric_lines_removal'],
        polynomial_fit_degree=simulated_data_model.model_parameters['polynomial_fit_degree'],
        tellurics_mask_threshold=simulated_data_model.model_parameters['tellurics_mask_threshold']
    )

    print(' Reprocessed noisy spectrum...')
    reprocessed_noisy_spectrum, reprocessed_matrix_noisy, _ = simulated_data_model.pipeline(
        spectrum=true_spectrum * deformation_matrix + noise_matrix,
        wavelengths=true_wavelengths,
        airmass=simulated_data_model.model_parameters['airmass'],
        uncertainties=simulated_data_model.model_parameters['uncertainties'],
        apply_throughput_removal=simulated_data_model.model_parameters['apply_throughput_removal'],
        apply_telluric_lines_removal=simulated_data_model.model_parameters['apply_telluric_lines_removal'],
        polynomial_fit_degree=simulated_data_model.model_parameters['polynomial_fit_degree'],
        tellurics_mask_threshold=simulated_data_model.model_parameters['tellurics_mask_threshold']
    )

    print('Checking framework validity (noisy)...', end='')

    assert np.allclose(
        reprocessed_spectrum,
        (true_spectrum * deformation_matrix + noise_matrix) * reprocessed_matrix_noisy,
        atol=1e-10,
        rtol=1e-10
    )
    print(' OK')

    noiseless_validity = 1 - reprocessed_true_spectrum / reprocessed_deformed_spectrum

    print('Reprocessing pipeline validity (noiseless):')
    print(f" {np.ma.mean(noiseless_validity):.3e} +/- {np.ma.std(noiseless_validity):.3e} "
          f"({np.ma.min(noiseless_validity):.3e} <= val <= {np.ma.max(noiseless_validity):.3e})")

    noisy_validity = 1 - (reprocessed_true_spectrum + noise_matrix * reprocessed_matrix_noisy) \
        / reprocessed_noisy_spectrum

    print('Reprocessing pipeline validity (noisy):')
    print(f" {np.ma.mean(noisy_validity):.3e} +/- {np.ma.std(noisy_validity):.3e} "
          f"({np.ma.min(noisy_validity):.3e} <= val <= {np.ma.max(noisy_validity):.3e})")

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
            convolve=convolve,
            rebin=rebin,
            reduce=True
        )

        assert np.allclose(retrieval_models[0][0], reprocessed_true_spectrum, atol=0.0, rtol=1e-14)

        true_chi2 = -2 * true_log_l[0][0] / np.size(reprocessed_spectrum[~reprocessed_spectrum.mask])

        # Check Log L and chi2 when using the true set of parameter
        print(f'True log L = {true_log_l[0][0]}')
        print(f'True chi2 = {true_chi2}')
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


def load_carmenes_data(directory, mid_transit_jd):
    # 58004.425291 # for 2017-09-07 source https://astro.swarthmore.edu/transits

    with fits.open(os.path.join(directory, 'airmass.fits')) as f:
        airmass = f[0].data

    with fits.open(os.path.join(directory, 'bary.fits')) as f:
        barycentric_velocities = f[0].data

    with fits.open(os.path.join(directory, 'mod_julian_date.fits')) as f:
        dates = f[0].data  # in MJD - 0.5 format, corresponds to 2017-09-07

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

    mid_transit = np.mod(mid_transit_jd, 1.0) * 24 * 3600  # for 2017-09-07 source https://astro.swarthmore.edu/transits
    times = np.mod(dates, 1.0) * 24 * 3600  # seconds
    # orbital_phases = np.mod((times - mid_transit) / planet_orbital_period - 0.5, 1.0) - 0.5

    spectra = np.ma.masked_invalid(spectra)
    snr = spectra / noise

    return wavelengths, spectra, snr, noise, orbital_phases, airmass, barycentric_velocities, times, mid_transit


# Figures
# Matplotlib sizes
TINY_FIGURE_FONT_SIZE = 40  # 0.5 text width 16/9
SMALL_FIGURE_FONT_SIZE = 22  # 0.25 text width
MEDIUM_FIGURE_FONT_SIZE = 16  # 0.5 text width
LARGE_FIGURE_FONT_SIZE = 22  # 1.0 text width

large_figsize = [19.20, 10.80]  # 1920 x 1080 for 100 dpi (default)

wavenumber_units = r'cm$^{-1}$'
wavelength_units = r'm'
spectral_radiosity_units = r'W$\cdot$m${-2}$/cm$^{-1}$'

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


def plot_model_steps(spectral_model, radtrans, mode, ccd_id,
                     path_outputs, figure_name='model_steps', image_format='pdf'):
    update_figure_font_size(MEDIUM_FIGURE_FONT_SIZE)
    orbital_phases = spectral_model.model_parameters['orbital_phases']

    # Step 1-3
    true_wavelengths_instrument, true_spectrum_instrument = spectral_model.get_spectrum_model(
            radtrans=radtrans,
            mode=mode,
            update_parameters=True,
            # telluric_transmittances=telluric_transmittance,
            telluric_transmittances=None,
            # instrumental_deformations=variable_throughput,
            instrumental_deformations=None,
            noise_matrix=None,
            scale=False,
            shift=False,
            convolve=False,
            rebin=False,
            reduce=False
        )

    # Step 4
    _, spectra_scale = spectral_model.get_spectrum_model(
            radtrans=radtrans,
            mode=mode,
            update_parameters=True,
            # telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
            # telluric_transmittances=telluric_transmittances,
            telluric_transmittances=None,
            # instrumental_deformations=instrumental_deformations,
            instrumental_deformations=None,
            noise_matrix=None,
            scale=True,
            shift=False,
            convolve=False,
            rebin=False,
            reduce=False
        )

    # Step 5
    w_shift, spectra_shift = spectral_model.get_spectrum_model(
            radtrans=radtrans,
            mode=mode,
            update_parameters=True,
            # telluric_transmittances=telluric_transmittance,
            telluric_transmittances=None,
            # instrumental_deformations=variable_throughput,
            instrumental_deformations=None,
            noise_matrix=None,
            scale=True,
            shift=True,
            convolve=False,
            rebin=False,
            reduce=False
        )

    # Step 6
    _, spectra_convolve = spectral_model.get_spectrum_model(
            radtrans=radtrans,
            mode=mode,
            update_parameters=True,
            # telluric_transmittances=telluric_transmittance,
            telluric_transmittances=None,
            # instrumental_deformations=variable_throughput,
            instrumental_deformations=None,
            noise_matrix=None,
            scale=True,
            shift=True,
            convolve=True,
            rebin=False,
            reduce=False
        )

    # Step 7
    wavelengths_instrument, spectra_final = spectral_model.get_spectrum_model(
            radtrans=radtrans,
            mode=mode,
            update_parameters=True,
            # telluric_transmittances=telluric_transmittance,
            telluric_transmittances=None,
            # instrumental_deformations=variable_throughput,
            instrumental_deformations=None,
            noise_matrix=None,
            scale=True,
            shift=True,
            convolve=True,
            rebin=True,
            reduce=False
        )

    # Plots
    w_shift = w_shift * 1e-6  # um to m
    wavelengths_instrument = wavelengths_instrument[ccd_id] * 1e-6  # um to m
    true_wavelengths_instrument *= 1e-6  # um to m
    true_spectrum_instrument *= 1e-2  # cm to m

    fig, axes = plt.subplots(nrows=5, ncols=1, sharex='col', figsize=(6.4, 5 * 1.6))

    axes[0].plot(true_wavelengths_instrument, true_spectrum_instrument, color='C2')
    axes[0].set_title('Step 3: base model')

    axes[1].plot(true_wavelengths_instrument, spectra_scale, color='C2')
    axes[1].set_title('Step 4: scaling')

    axes[2].plot(w_shift[0], spectra_shift[0], label=rf'$\Phi$ = {orbital_phases[0]:.3f}', color='C0')
    axes[2].plot(w_shift[-1], spectra_shift[-1], label=rf'$\Phi$ = {orbital_phases[-1]:.3f}', color='C3')
    axes[2].legend(loc=4)
    axes[2].set_title('Step 5: shifting')

    axes[3].plot(w_shift[0], spectra_convolve[0], label=rf'$\Phi$ = {orbital_phases[0]:.3f}', color='C0')
    axes[3].plot(w_shift[-1], spectra_convolve[-1], label=rf'$\Phi$ = {orbital_phases[-1]:.3f}', color='C3')
    axes[3].legend(loc=4)
    axes[3].set_title('Step 6: convolving')

    axes[4].pcolormesh(
        wavelengths_instrument,
        orbital_phases,
        spectra_final[ccd_id],
        shading='nearest',
        cmap='viridis'
    )
    axes[4].set_title('Step 7: re-binning')

    axes[-1].set_xlim([wavelengths_instrument[0], wavelengths_instrument[-1]])
    x_ticks = axes[-1].get_xticks()
    axes[-1].set_xticks(x_ticks[1::2])
    axes[-1].set_xlim([wavelengths_instrument[0], wavelengths_instrument[-1]])

    plt.tight_layout()

    gs = axes[0].get_gridspec()

    spectral_axes = fig.add_subplot(gs[0:1], frameon=False)
    spectral_axes.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    spectral_axes.set_ylabel(r'$m_{\theta,0}$ (m)', labelpad=20)

    spectral_axes = fig.add_subplot(gs[1:4], frameon=False)
    spectral_axes.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    spectral_axes.set_ylabel('Arbitrary units', labelpad=20)

    spectral_axes = fig.add_subplot(gs[4:], frameon=False)
    spectral_axes.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    spectral_axes.set_ylabel(r'$\Phi$', labelpad=20)

    plt.savefig(os.path.join(path_outputs, figure_name + '.' + image_format))


def plot_model_steps_model(spectral_model, radtrans, mode, ccd_id,
                           telluric_transmittances_wavelengths, telluric_transmittances, instrumental_deformations,
                           noise_matrix,
                           path_outputs, figure_name='simulated_data_steps', image_format='pdf', noise_factor=100.0):
    update_figure_font_size(MEDIUM_FIGURE_FONT_SIZE)
    orbital_phases = spectral_model.model_parameters['orbital_phases']

    # Step 5 bis
    w_shift, spectra_shift = spectral_model.get_spectrum_model(
            radtrans=radtrans,
            mode=mode,
            update_parameters=True,
            # telluric_transmittances=telluric_transmittance,
            telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
            telluric_transmittances=telluric_transmittances,
            # instrumental_deformations=variable_throughput,
            instrumental_deformations=None,
            noise_matrix=None,
            scale=True,
            shift=True,
            convolve=False,
            rebin=False,
            reduce=False
        )

    # Step 6
    _, spectra_convolve = spectral_model.get_spectrum_model(
            radtrans=radtrans,
            mode=mode,
            update_parameters=True,
            telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
            telluric_transmittances=telluric_transmittances,
            # telluric_transmittances=None,
            # instrumental_deformations=variable_throughput,
            instrumental_deformations=None,
            noise_matrix=None,
            scale=True,
            shift=True,
            convolve=True,
            rebin=False,
            reduce=False
        )

    # Step 7
    wavelengths_instrument, spectra_final = spectral_model.get_spectrum_model(
            radtrans=radtrans,
            mode=mode,
            update_parameters=True,
            telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
            telluric_transmittances=telluric_transmittances,
            # telluric_transmittances=None,
            # instrumental_deformations=variable_throughput,
            instrumental_deformations=None,
            noise_matrix=None,
            scale=True,
            shift=True,
            convolve=True,
            rebin=True,
            reduce=False
        )

    # Step 8
    _, spectra_tt = spectral_model.get_spectrum_model(
            radtrans=radtrans,
            mode=mode,
            update_parameters=True,
            telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
            telluric_transmittances=telluric_transmittances,
            instrumental_deformations=instrumental_deformations,
            noise_matrix=None,
            scale=True,
            shift=True,
            convolve=True,
            rebin=True,
            reduce=False
        )

    # Step 9
    _, spectra_n = spectral_model.get_spectrum_model(
            radtrans=radtrans,
            mode=mode,
            update_parameters=True,
            telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
            telluric_transmittances=telluric_transmittances,
            instrumental_deformations=instrumental_deformations,
            noise_matrix=noise_matrix * noise_factor,
            scale=True,
            shift=True,
            convolve=True,
            rebin=True,
            reduce=False
        )

    # Plots
    w_shift = w_shift * 1e-6  # um to m
    wavelengths_instrument = wavelengths_instrument[ccd_id] * 1e-6  # um to m

    fig, axes = plt.subplots(nrows=5, ncols=1, sharex='col', figsize=(6.4, 5 * 1.6))

    axes[0].plot(w_shift[0], spectra_shift[0], label=rf'$\Phi$ = {orbital_phases[0]:.3f}', color='C0')
    axes[0].plot(w_shift[-1], spectra_shift[-1], label=rf'$\Phi$ = {orbital_phases[-1]:.3f}', color='C3')
    axes[0].legend(loc=4)
    axes[0].set_title('Step 5 bis: adding telluric transmittance')

    axes[1].plot(w_shift[0], spectra_convolve[0], label=rf'$\Phi$ = {orbital_phases[0]:.3f}', color='C0')
    axes[1].plot(w_shift[-1], spectra_convolve[-1], label=rf'$\Phi$ = {orbital_phases[-1]:.3f}', color='C3')
    axes[1].legend(loc=4)
    axes[1].set_title('Step 6: convolving')

    axes[2].pcolormesh(
        wavelengths_instrument,
        orbital_phases,
        spectra_final[ccd_id],
        shading='nearest',
        cmap='viridis'
    )
    axes[2].set_title('Step 7: re-binning')

    axes[3].pcolormesh(
        wavelengths_instrument,
        orbital_phases,
        spectra_tt[ccd_id],
        shading='nearest',
        cmap='viridis'
    )
    axes[3].set_title('Step 8: adding instrumental deformations')

    axes[4].pcolormesh(
        wavelengths_instrument,
        orbital_phases,
        spectra_n[ccd_id],
        shading='nearest',
        cmap='viridis'
    )
    axes[4].set_title(f'Step 9: adding noise ({100:.0f} times increased)')
    axes[4].set_xlabel('Wavelength (m)')

    axes[-1].set_xlim([wavelengths_instrument[0], wavelengths_instrument[-1]])
    x_ticks = axes[-1].get_xticks()
    axes[-1].set_xticks(x_ticks[1::2])
    axes[-1].set_xlim([wavelengths_instrument[0], wavelengths_instrument[-1]])

    plt.tight_layout()

    gs = axes[0].get_gridspec()

    spectral_axes = fig.add_subplot(gs[0:2], frameon=False)
    spectral_axes.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    spectral_axes.set_ylabel('Arbitrary units', labelpad=20)

    spectral_axes = fig.add_subplot(gs[2:], frameon=False)
    spectral_axes.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    spectral_axes.set_ylabel(r'$\Phi$', labelpad=20)

    plt.savefig(os.path.join(path_outputs, figure_name + '.' + image_format))


def plot_reprocessing_effect_1d(spectral_model, radtrans, uncertainties, mode,
                                telluric_transmittances_wavelengths, telluric_transmittances, instrumental_deformations,
                                ccd_id, orbital_phase_id,
                                path_outputs, figure_name='reprocessing_steps', image_format='pdf'):
    # Ref
    wavelengths_ref, spectra_ref = spectral_model.get_spectrum_model(
            radtrans=radtrans,
            mode=mode,
            update_parameters=True,
            telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
            telluric_transmittances=telluric_transmittances,
            # telluric_transmittances=None,
            instrumental_deformations=instrumental_deformations,
            # instrumental_deformations=None,
            noise_matrix=None,
            scale=True,
            shift=True,
            convolve=True,
            rebin=True,
            reduce=False
        )

    # Start
    _, spectra_start = spectral_model.get_spectrum_model(
            radtrans=radtrans,
            mode=mode,
            update_parameters=True,
            telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
            telluric_transmittances=telluric_transmittances,
            # telluric_transmittances=None,
            instrumental_deformations=instrumental_deformations,
            # instrumental_deformations=deformation_matrix,
            noise_matrix=None,
            scale=True,
            shift=True,
            convolve=True,
            rebin=True,
            reduce=False
        )

    spectra_start = np.ma.masked_where(uncertainties.mask, spectra_start)

    # Step 1
    spectra_vt_corrected, vt_matrix, vt_uncertainties = reprocessing_pipeline(
            spectrum=spectra_start,
            uncertainties=uncertainties,
            wavelengths=wavelengths_ref,
            airmass=spectral_model.model_parameters['airmass'],
            tellurics_mask_threshold=spectral_model.model_parameters['tellurics_mask_threshold'],
            polynomial_fit_degree=spectral_model.model_parameters['polynomial_fit_degree'],
            apply_throughput_removal=True,
            apply_telluric_lines_removal=False,
            full=True
        )

    # Step 2
    spectra_corrected, r_matrix, r_uncertainties = reprocessing_pipeline(
            spectrum=spectra_vt_corrected,
            uncertainties=vt_uncertainties,
            wavelengths=wavelengths_ref,
            airmass=spectral_model.model_parameters['airmass'],
            tellurics_mask_threshold=spectral_model.model_parameters['tellurics_mask_threshold'],
            polynomial_fit_degree=spectral_model.model_parameters['polynomial_fit_degree'],
            apply_throughput_removal=False,
            apply_telluric_lines_removal=True,
            full=True
        )

    # Plots
    wavelengths_ref *= 1e-6  # um to m
    update_figure_font_size(MEDIUM_FIGURE_FONT_SIZE)

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex='col', figsize=(6.4, 4.8))

    axes[0].plot(wavelengths_ref[ccd_id], spectra_start[ccd_id, orbital_phase_id])
    axes[0].set_title('Base noiseless spectrum')

    axes[1].plot(wavelengths_ref[ccd_id], spectra_vt_corrected[ccd_id, orbital_phase_id])
    axes[1].set_title('Reprocessing step 1')

    axes[2].plot(
        wavelengths_ref[ccd_id], spectra_corrected[ccd_id, orbital_phase_id],
        label='reprocessed spectrum'
    )

    axes[2].set_title('Reprocessing step 2')
    axes[2].set_xlabel('Wavelength (m)')
    axes[2].set_xlim([wavelengths_ref[ccd_id].min(), wavelengths_ref[ccd_id].max()])
    axes[2].ticklabel_format(useOffset=True)

    axes[-1].set_xlim([wavelengths_ref[ccd_id][0], wavelengths_ref[ccd_id][-1]])
    x_ticks = axes[-1].get_xticks()
    axes[-1].set_xticks(x_ticks[1::2])
    axes[-1].set_xlim([wavelengths_ref[ccd_id][0], wavelengths_ref[ccd_id][-1]])

    plt.tight_layout()
    # axes[2].legend()

    # gs = axes[0].get_gridspec()
    #
    # spectral_axes = fig.add_subplot(gs[:], frameon=False)
    # spectral_axes.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    # spectral_axes.set_ylabel('Orbital phase', labelpad=20)

    plt.savefig(os.path.join(path_outputs, figure_name + '.' + image_format))


def plot_reprocessing_effect(spectral_model, radtrans, reprocessed_data, mode, simulated_uncertainties, ccd_id,
                             telluric_transmittances_wavelengths, telluric_transmittances, instrumental_deformations,
                             noise_matrix,
                             path_outputs, figure_name='reprocessing_effect', image_format='pdf'):
    update_figure_font_size(MEDIUM_FIGURE_FONT_SIZE)
    orbital_phases = spectral_model.model_parameters['orbital_phases']

    wavelengths, data_noiseless = spectral_model.get_spectrum_model(
            radtrans=radtrans,
            mode=mode,
            update_parameters=True,
            telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
            telluric_transmittances=telluric_transmittances,
            # telluric_transmittances=None,
            instrumental_deformations=instrumental_deformations,
            # instrumental_deformations=deformation_matrix,
            noise_matrix=None,
            scale=True,
            shift=True,
            convolve=True,
            rebin=True,
            reduce=False
        )

    fake_model = copy.deepcopy(spectral_model)
    fake_model.model_parameters['uncertainties'] = simulated_uncertainties
    _, reprocessed_data_noiseless_fake = fake_model.get_spectrum_model(
            radtrans=radtrans,
            mode=mode,
            update_parameters=True,
            telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
            telluric_transmittances=telluric_transmittances,
            # telluric_transmittances=None,
            instrumental_deformations=instrumental_deformations,
            # instrumental_deformations=deformation_matrix,
            noise_matrix=None,
            scale=True,
            shift=True,
            convolve=True,
            rebin=True,
            reduce=True
        )

    _, reprocessed_data_noiseless = spectral_model.get_spectrum_model(
            radtrans=radtrans,
            mode=mode,
            update_parameters=True,
            telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
            telluric_transmittances=telluric_transmittances,
            # telluric_transmittances=None,
            instrumental_deformations=instrumental_deformations,
            # instrumental_deformations=deformation_matrix,
            noise_matrix=None,
            scale=True,
            shift=True,
            convolve=True,
            rebin=True,
            reduce=True
        )

    _, reprocessed_data_noisy = spectral_model.get_spectrum_model(
            radtrans=radtrans,
            mode=mode,
            update_parameters=True,
            telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
            telluric_transmittances=telluric_transmittances,
            # telluric_transmittances=None,
            instrumental_deformations=instrumental_deformations,
            # instrumental_deformations=deformation_matrix,
            noise_matrix=noise_matrix,
            scale=True,
            shift=True,
            convolve=True,
            rebin=True,
            reduce=True
        )

    # Plots
    wavelengths = wavelengths[ccd_id] * 1e-6  # um to m

    fig, axes = plt.subplots(nrows=4, ncols=1, sharex='col', figsize=(6.4, 6.4))

    # axes[0].imshow(
    #     data_noiseless[0],
    #     origin='lower',
    #     extent=[wavelengths[0], wavelengths[-1], orbital_phases[0], orbital_phases[-1]],
    #     aspect='auto',
    #     vmin=None,
    #     vmax=None,
    #     cmap='viridis'
    # )
    # axes[0].set_title('Noiseless mock observations')

    axes[0].pcolormesh(
        wavelengths,
        orbital_phases,
        reprocessed_data_noiseless_fake[ccd_id],
        cmap='viridis'
    )
    axes[0].set_title(r'Reprocessed noiseless mock data ($\sigma_{\epsilon,\mathrm{sim}}$)')

    axes[1].pcolormesh(
        wavelengths,
        orbital_phases,
        reprocessed_data_noiseless[ccd_id],
        cmap='viridis'
    )
    axes[1].set_title(r'Reprocessed noiseless mock data ($\sigma_\epsilon$)')

    axes[2].pcolormesh(
        wavelengths,
        orbital_phases,
        reprocessed_data_noisy[ccd_id],
        cmap='viridis'
    )
    axes[2].set_title(r'Reprocessed noisy mock data ($\sigma_\epsilon$)')

    axes[3].pcolormesh(
        wavelengths,
        orbital_phases,
        reprocessed_data[ccd_id],
        cmap='viridis'
    )
    axes[3].set_title('Reprocessed CARMENES data')
    axes[3].set_xlabel('Wavelength (m)')

    axes[-1].set_xlim([wavelengths[0], wavelengths[-1]])
    x_ticks = axes[-1].get_xticks()
    axes[-1].set_xticks(x_ticks[1::2])
    axes[-1].set_xlim([wavelengths[0], wavelengths[-1]])

    plt.tight_layout()

    gs = axes[0].get_gridspec()

    spectral_axes = fig.add_subplot(gs[:], frameon=False)
    spectral_axes.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    spectral_axes.set_ylabel('Orbital phase', labelpad=20)

    plt.savefig(os.path.join(path_outputs, figure_name + '.' + image_format))


def plot_corner_comparison(true_parameters, retrieval_directory):
    retrieval_names = ['t0l1_vttt_mm_true_kp_vr_CO_H2O_79-80_transit_1000lp',
                       't0l1_vttt_mm_p_kp_vr_CO_H2O_79-80_transit_1000lp',
                       't0l1_vttt_mm_p_approx_kp_vr_CO_H2O_79-80_transit_1000lp']
    sample_dicts = {}
    parameter_dicts = {}
    true_values = {}
    parameter_plot_indices = {}

    for retrieval_name in retrieval_names:
        sample_dict, parameter_dict = Retrieval._get_samples(ultranest=False, names=[retrieval_name],
                                                             output_dir=f'./petitRADTRANS/__tmp/test_retrieval/'
                                                                        f'{retrieval_name}/',
                                                             ret_names=[retrieval_name]
                                                             )
        n_param = len(parameter_dict[retrieval_name])
        parameter_plot_indices[retrieval_name] = np.arange(0, n_param)
        sample_dicts[retrieval_name] = sample_dict[retrieval_name]
        parameter_dicts[retrieval_name] = parameter_dict[retrieval_name]

        true_values[retrieval_name] = []
        for p in parameter_dict[retrieval_name]:
            true_values[retrieval_name].append(np.mean(true_parameters[p].value))

    contour_corner(
        sample_dicts, parameter_dicts, os.path.join(retrieval_directory, f'corner_cmp.png'),
        parameter_plot_indices=parameter_plot_indices,
        true_values=true_values, prt_plot_style=False, hist2d_kwargs={'plot_density': False}
    )


def plot_init(retrieved_parameters, expected_retrieval_directory, sm):
    sd = static_get_sample(expected_retrieval_directory)
    true_values = []
    true_values_dict = {}

    for p in sd:
        if p not in sm.model_parameters and 'log10_' not in p:
            true_values.append(
                np.mean(np.log10(sm.model_parameters['imposed_mass_mixing_ratios'][p]))
            )
        elif p not in sm.model_parameters and 'log10_' in p:
            p = p.split('log10_', 1)[1]
            true_values.append(np.mean(np.log10(sm.model_parameters[p])))
        else:
            true_values.append(np.mean(sm.model_parameters[p]))

    i = -1
    for key, value in retrieved_parameters.items():
        i += 1

        if 'figure_coefficient' in value:
            sd[key] *= value['figure_coefficient']
            true_values[i] *= value['figure_coefficient']

        true_values_dict[key] = true_values[i]

    return sd, true_values_dict


def plot_partial_corners(retrieved_parameters, sd, sm, true_values, figure_directory, image_format, split_at=5):
    update_figure_font_size(11)

    parameter_ranges, fig_labels, fig_titles, _ = get_parameter_range(sd, sm, retrieved_parameters)

    contour_corner(
        {'': np.array(list(sd.values())).T[:, :split_at]}, {'': fig_labels[:split_at]},
        os.path.join(r'C:\Users\Doriann\Documents\work\run_outputs\petitRADTRANS\figures\HD_189733_b_CARMENES',
                     f'corner_mock_mplus_ttt0.5_1.pdf'),
        parameter_plot_indices={'': np.arange(0, len(fig_labels[:split_at]))},
        parameter_ranges={'': parameter_ranges[:split_at]},
        true_values={'': true_values[:split_at]},
        prt_plot_style=False,
    )
    plt.savefig(os.path.join(figure_directory, 'corner_mock_mplus_ttt0.5_1' + '.' + image_format))

    contour_corner(
        {'': np.array(list(sd.values())).T[:, split_at:]}, {'': fig_labels[split_at:]},
        os.path.join(r'C:\Users\Doriann\Documents\work\run_outputs\petitRADTRANS\figures\HD_189733_b_CARMENES',
                     f'corner_mock_mplus_ttt0.5_2.pdf'),
        parameter_plot_indices={'': np.arange(0, len(fig_labels[split_at:]))},
        parameter_ranges={'': parameter_ranges[split_at:]},
        true_values={'': true_values[split_at:]},
        prt_plot_style=False,
    )
    plt.savefig(os.path.join(figure_directory, 'corner_mock_mplus_ttt0.5_2' + '.' + image_format))


def plot_result_corner(result_directory, sm, retrieved_parameters,
                       figure_directory, figure_name, image_format='pdf', true_values=None, save=True, **kwargs):
    sd = static_get_sample(result_directory)
    parameter_ranges, fig_labels, fig_titles, coefficients = get_parameter_range(sd, sm, retrieved_parameters)

    if true_values is not None:
        if isinstance(true_values, dict):
            if list(true_values.keys())[0] != '':
                true_values = [true_values[key] for key in sd]
                true_values = {'': true_values}
            else:
                true_values = {'': true_values}
        else:
            true_values = {'': true_values}

    if 'hist2d_kwargs' in kwargs:
        kwargs['hist2d_kwargs']['titles'] = fig_titles
    else:
        kwargs['hist2d_kwargs'] = {'titles': fig_titles}

    update_figure_font_size(11)

    contour_corner(
        sampledict={'': np.array(list(sd.values())).T * coefficients},
        parameter_names={'': fig_labels},
        output_file=None,
        parameter_plot_indices={'': np.arange(0, len(parameter_ranges))},
        parameter_ranges={'': parameter_ranges},
        true_values=true_values,
        prt_plot_style=False,
        **kwargs
    )

    figure_size = plt.gcf().get_size_inches()

    if np.max(figure_size) > 19.2:
        plt.gcf().set_size_inches(19.2, 19.2)

    if save:
        plt.savefig(os.path.join(figure_directory, figure_name + '.' + image_format))


def plot_corner(retrieved_parameters, sd, sm, true_values, figure_directory, image_format, save=False):
    update_figure_font_size(11)

    parameter_ranges, fig_names, _ = get_parameter_range(sd, sm, retrieved_parameters)

    contour_corner(
        {'': np.array(list(sd.values())).T[:, :]}, {'': fig_names},
        os.path.join(r'C:\Users\Doriann\Documents\work\run_outputs\petitRADTRANS\figures\HD_189733_b_CARMENES',
                     f'corner_mock_mplus_ttt0.5_1.pdf'),
        parameter_plot_indices={'': np.arange(0, len(fig_names))},
        parameter_ranges={'': parameter_ranges},
        true_values={'': true_values},
        prt_plot_style=False,
    )

    if save:
        plt.savefig(os.path.join(figure_directory, 'corner_mock_mplus_ttt0.5_1' + '.' + image_format))


def plot_validity(sm, radtrans, figure_directory, image_format):
    deformation_matrix = sm.model_parameters['instrumental_deformations']
    noise_matrix = np.random.default_rng().normal(
        loc=0, scale=sm.model_parameters['uncertainties'], size=sm.model_parameters['uncertainties'].shape
    )

    validity, true_log_l, true_chi2, noiseless_validity, true_wavelengths, true_spectrum, deformed_spectrum, \
        reprocessed_spectrum, reprocessed_true_spectrum, reprocessed_deformed_spectrum, reprocessed_noisy_spectrum, \
        reprocessed_matrix_true, reprocessed_matrix_deformed, reprocessed_matrix_noisy = validity_checks(
            simulated_data_model=copy.deepcopy(sm),
            radtrans=radtrans,
            # telluric_transmittances=telluric_transmittance,
            telluric_transmittances_wavelengths=None,
            telluric_transmittances=None,
            # instrumental_deformations=variable_throughput,
            instrumental_deformations=deformation_matrix,
            noise_matrix=noise_matrix,
            scale=True,
            shift=True,
            convolve=True,
            rebin=True,
            save=True,
            filename=os.path.join(figure_directory, 'validity.npz'),
            full=True
        )

    plot_hist(np.log10(np.abs(validity).flatten()), r'$\log_{10}$(|Validity|)')
    no_pipeline = np.log10(
        np.abs(1 - (true_spectrum + noise_matrix) / (deformed_spectrum + noise_matrix))
    )
    data_only_pipeline = np.log10(
        np.abs(1 - (true_spectrum + noise_matrix * reprocessed_matrix_noisy) / reprocessed_noisy_spectrum)
    )
    colors = ['k', 'C1', 'C0']
    labels = ['No pipeline', 'Data-only pipeline', 'Data+model pipeline']

    for i, d in enumerate([no_pipeline, data_only_pipeline]):
        plt.vlines(np.median(d), 0, 1.1, color=colors[i], ls='-')
        plt.vlines(np.quantile(d, 0.16), 0, 1.1, color=colors[i], ls='--')
        plt.vlines(np.quantile(d, 0.84), 0, 1.1, color=colors[i], ls='--')

    for i, c in enumerate(colors):
        plt.plot([-np.inf, -np.inf], color=c, label=labels[i])

    plt.legend(loc=2)
    plt.savefig(os.path.join(figure_directory, 'validity' + '.' + image_format))


def plot_contribution(sm, radtrans, figure_directory, image_format):
    plt.figure()
    plt.imshow(radtrans.contr_tr, aspect='auto', origin='upper',
               extent=[np.min(sm.wavelengths) * 1e-6, np.max(sm.wavelengths) * 1e-6, np.log10(sm.pressures[-1]) + 5,
                       np.log10(sm.pressures[0]) + 5])
    plt.colorbar(label='Contribution density')
    plt.xlabel('Wavelength (m)')
    plt.ylabel(r'$\log_{10}$(pressure) [Pa]')
    plt.ylim([7, -1])
    plt.tight_layout()
    plt.savefig(os.path.join(figure_directory, 'contribution' + '.' + image_format))


def plot_species_contribution(wavelengths_instrument, figure_directory, image_format, observations=None):
    # Species contribution
    update_figure_font_size(18)
    fig, axe = plt.subplots(figsize=(6.4 * 3, 4.8 * 1.5))
    plot_transmission_contribution_spectra(
        r'\\wsl$\Debian\home\dblain\exorem\outputs\exorem\hd_1899733_b_z3_t100_co0.55_nocloud.h5',
        exclude=['clouds', 'CO2', 'PH3', 'TiO', 'VO'],
        wvn2wvl=True
    )
    ymax = 22850
    axe.set_xlim([np.min(wavelengths_instrument) * 1e-6 - 0.01e-6, np.max(wavelengths_instrument) * 1e-6 + 0.11e-6])
    axe.set_ylim([22175, ymax])

    if observations is not None:
        axe_twin = axe.twinx()
        mean_observations = np.mean(observations, axis=1)

        for i, wvl in enumerate(wavelengths_instrument):
            if i == 0:
                axe_twin.plot(wvl * 1e-6, mean_observations[i], color='k', alpha=0.3, label='Data')
            else:
                axe_twin.plot(wvl * 1e-6, mean_observations[i], color='k', alpha=0.3)

        axe_twin.set_ylabel('Radiosity (arbitrary units)')
        axe_twin.set_ylim([-0.25, 0.5])
        axe_twin.legend(loc=4)

    for i, wvl in enumerate(wavelengths_instrument):
        axe.fill_betweenx([0, 1e30], wvl.min() * 1e-6, wvl.max() * 1e-6, color='grey', alpha=0.3)

        if np.mod(i, 2) == 0:
            axe.text(np.mean((wvl.min(), wvl.max())) * 1e-6, 0.9999 * ymax, f'{i}', fontsize=16, ha='center', va='top')

    fig.tight_layout()
    axe.legend(loc=1)
    fig.set_rasterized(True)
    fig.savefig(os.path.join(figure_directory, 'species_contribution' + '.' + image_format))


def plot_all_figures(retrieved_parameters,
                     figure_directory=r'C:\Users\Doriann\Documents\work\run_outputs\petitRADTRANS\figures\
                                        HD_189733_b_CARMENES',
                     image_format='pdf'):
    # Init
    retrieval_directory = r'C:\Users\Doriann\Documents\work\run_outputs\petitRADTRANS\retrievals\carmenes_retrievals'
    sm = SpectralModel.load(
        retrieval_directory +
        r'\HD_189733_b_transmission_'
        r'R_Kp_V0_Rp_g_tiso_CH4_CO_H2O_H2S_HCN_NH3_Pc_k0_gams_strictt1535_t1535_t23_sim_1000lp\simulated_data_model.h5'
    )
    radtrans = sm.get_radtrans()

    sd, true_values = plot_init(
        retrieved_parameters,
        retrieval_directory +
        r'\HD_189733_b_transmission_'
        r'R_Kp_V0_Rp_g_tiso_CH4_CO_H2O_H2S_HCN_NH3_Pc_k0_gams_strictt1535_t1535_t23_sim_1000lp',
        sm
    )

    wavelengths_instrument, observed_spectra, instrument_snr, uncertainties, orbital_phases, airmasses, \
        barycentric_velocities, times, mid_transit_time = load_carmenes_data(
            directory=os.path.join(
                r'C:\Users\Doriann\Documents\work\run_outputs\petitRADTRANS\data',
                'carmenes',
                'hd_189733_b'
            ),
            mid_transit_jd=58004.425291
        )

    instrumental_deformations, telluric_transmittances_wavelengths, telluric_transmittances, simulated_snr = \
        load_additional_data(
            data_dir=r'C:\Users\Doriann\Documents\work\run_outputs\petitRADTRANS\data',
            wavelengths_instrument=wavelengths_instrument,
            airmasses=airmasses,
            times=times,
            resolving_power=sm.model_parameters['new_resolving_power']
        )

    detector_selection = np.array([1, 3, 9, 14, 25, 28, 29, 30, 46, 54])  # strictt1535
    ccd_id = np.asarray(detector_selection == 46).nonzero()[0][0]
    planet = Planet.get('HD 189733 b')

    # Modify TD
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

    print(f"Adding exposures of half-eclipses")
    planet_transit_duration += (planet.transit_duration - planet_transit_duration) / 2

    wavelengths_instrument_0 = copy.deepcopy(wavelengths_instrument[:, 0])
    observed_spectra_0 = copy.deepcopy(observed_spectra)

    wavelengths_instrument, observed_spectra, uncertainties, instrument_snr, instrumental_deformations, \
        telluric_transmittances_wavelengths, telluric_transmittances, simulated_snr, orbital_phases, airmasses, \
        barycentric_velocities, times = \
        data_selection(
            wavelengths_instrument=wavelengths_instrument,
            observed_spectra=observed_spectra,
            uncertainties=uncertainties,
            instrument_snr=instrument_snr,
            instrumental_deformations=instrumental_deformations,
            telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
            telluric_transmittances=telluric_transmittances,
            simulated_snr=simulated_snr,
            times=times,
            mid_transit_time=mid_transit_time,
            transit_duration=planet_transit_duration,
            orbital_phases=orbital_phases,
            airmasses=airmasses,
            barycentric_velocities=barycentric_velocities,
            detector_selection=detector_selection,
            n_transits=1,
            use_t23=True,
            use_t1535=True
        )

    simulated_uncertainties = np.moveaxis(
        np.moveaxis(simulated_snr, 2, 0) / np.mean(simulated_snr, axis=2) * np.mean(uncertainties, axis=2),
        0,
        2
    )

    noise_matrix = np.random.default_rng().normal(loc=0, scale=uncertainties, size=observed_spectra.shape)

    reprocessed_data, reprocessing_matrix, reprocessed_data_uncertainties = sm.pipeline(
        observed_spectra,
        wavelengths=wavelengths_instrument,
        **sm.model_parameters
    )

    # Figure 1 from Alex

    # Model steps
    plot_model_steps(
        spectral_model=sm,
        radtrans=radtrans,
        mode='transmission',
        ccd_id=ccd_id,
        path_outputs=figure_directory,
        image_format=image_format
    )

    plot_model_steps_model(
        spectral_model=sm,
        radtrans=radtrans,
        mode='transmission',
        ccd_id=ccd_id,
        telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
        telluric_transmittances=telluric_transmittances,
        instrumental_deformations=instrumental_deformations,
        noise_matrix=noise_matrix,
        path_outputs=figure_directory,
        image_format=image_format,
        noise_factor=100
    )

    # TODO Expected MMRs
    # TODO Expected TP
    # Contribution (not kept in final version)
    plot_contribution(sm, radtrans, figure_directory, image_format)

    # Species contribution
    plot_species_contribution(wavelengths_instrument_0, figure_directory, image_format, observations=observed_spectra_0)

    # Reprocessing steps
    plot_reprocessing_effect_1d(
        spectral_model=sm,
        radtrans=radtrans,
        uncertainties=uncertainties,
        mode='transmission',
        telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
        telluric_transmittances=telluric_transmittances,
        instrumental_deformations=instrumental_deformations,
        ccd_id=ccd_id,
        orbital_phase_id=10,
        path_outputs=figure_directory,
        image_format=image_format
    )

    # Reprocessing effect
    plot_reprocessing_effect(
        spectral_model=sm,
        radtrans=radtrans,
        reprocessed_data=reprocessed_data,
        mode='transmission',
        simulated_uncertainties=simulated_uncertainties,
        ccd_id=ccd_id,
        telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
        telluric_transmittances=telluric_transmittances,
        instrumental_deformations=instrumental_deformations,
        noise_matrix=noise_matrix,
        path_outputs=figure_directory,
        image_format=image_format
    )

    # Validity
    plot_validity(sm, radtrans, figure_directory, image_format)

    # Expected retrieval corner plot
    plot_result_corner(
        result_directory=retrieval_directory +
                         r'\HD_189733_b_transmission_'
                         r'R_Kp_V0_Rp_g_tiso_CH4_CO_H2O_H2S_HCN_NH3_Pc_k0_gams_strictt1535_t1535_t23_sim_1000lp',
        sm=sm,
        retrieved_parameters=retrieved_parameters,
        figure_directory=figure_directory,
        figure_name='corner_R_Kp_V0_g_tiso_CH4_CO_H2O_H2S_HCN_NH3_Pc_k0_gams_strictt1535_t1535_t23_sim_1000lp',
        label_kwargs={'fontsize': 10},
        title_kwargs={'fontsize': 8},
        true_values=true_values,
        save=True
    )


# Exo-REM
def load_result(file, **kwargs):
    """
    Load an Exo-REM data file.
    :param file: data file
    :param kwargs: keyword arguments for loadtxt or h5py.File
    :return: the data
    """
    import h5py
    data_dict = h5py.File(file, mode='r', **kwargs)

    return data_dict


def plot_transmission_contribution_spectra(file, offset=0.0, cloud_altitude=None, wvn2wvl=False,
                                           xlim=None, legend=False, exclude=None,
                                           **kwargs):
    """
    Plot the different contributions in the transmission spectrum.
    :param file: spectrum file
    :param offset: (m) altitude offset of the transmission spectrum
    :param cloud_altitude: (m) add an opaque cloud deck at the given altitude
    :param wvn2wvl: convert wavenumbers (cm-1) into wavelengths (m)
    :param xlim: x-axis boundaries
    :param legend: plot the legend
    :param exclude: list of label to exclude (e.g. ['H2O', 'clouds'])
    :param kwargs: keyword arguments for plot
    """
    if exclude is None:
        exclude = np.array([None])
    else:
        exclude = np.asarray(exclude)

    data_dict = load_result(file)

    x_axis = np.asarray(data_dict['outputs']['spectra']['wavenumber'])

    if wvn2wvl:
        x_axis = 1e-2 / x_axis

    if wvn2wvl:
        x_axis_label = rf'Wavelength ({wavelength_units})'
    else:
        x_axis_label = rf'Wavenumber ({wavenumber_units})'

    for key in data_dict['outputs']['spectra']['transmission']['contributions']:
        if key == 'cia_rayleigh' or key == 'clouds':
            continue

        color = None

        for species in species_color:
            if species == key:
                color = species_color[species]
                break

        label = key

        if np.any(exclude == label):
            continue

        label = get_species_string(label)

        y_axis = np.asarray(data_dict['outputs']['spectra']['transmission']['contributions'][key])

        if offset != 0:
            star_radius = data_dict['model_parameters']['light_source']['radius'][()]
            planet_radius_0 = star_radius * np.sqrt(y_axis)
            y_axis = ((planet_radius_0 + offset) / star_radius) ** 2

        plt.plot(x_axis, y_axis * 1e6, color=color, label=label, **kwargs)

    if cloud_altitude is not None:
        star_radius = data_dict['model_parameters']['light_source']['radius'][()]
        planet_radius = data_dict['model_parameters']['target']['radius_1e5Pa'][()]

        planet_radius_0 = planet_radius + cloud_altitude
        y_axis = np.ones(np.size(x_axis)) * ((planet_radius_0 + offset) / star_radius) ** 2

        plt.plot(x_axis, y_axis * 1e6, color='k', ls='--', label='cloud')
    elif 'clouds' not in exclude:
        y_axis = np.asarray(data_dict['outputs']['spectra']['transmission']['contributions']['clouds'])

        if offset != 0:
            star_radius = data_dict['model_parameters']['light_source']['radius'][()]
            planet_radius_0 = star_radius * np.sqrt(y_axis)
            y_axis = ((planet_radius_0 + offset) / star_radius) ** 2

        plt.plot(x_axis, y_axis * 1e6, color='k', ls='--', label='clouds')

    if 'cia' not in exclude:
        y_axis = np.asarray(data_dict['outputs']['spectra']['transmission']['contributions']['cia_rayleigh'])

        if offset != 0:
            star_radius = data_dict['model_parameters']['light_source']['radius'][()]
            planet_radius_0 = star_radius * np.sqrt(y_axis)
            y_axis = ((planet_radius_0 + offset) / star_radius) ** 2

        plt.plot(x_axis, y_axis * 1e6, color='k', ls=':', label='CIA+Ray')

    y_axis = np.asarray(data_dict['outputs']['spectra']['transmission']['transit_depth'])

    if 'Total' not in exclude:
        if offset != 0:
            star_radius = data_dict['model_parameters']['light_source']['radius'][()]
            planet_radius_0 = star_radius * np.sqrt(y_axis)
            y_axis = ((planet_radius_0 + offset) / star_radius) ** 2

        plt.plot(x_axis, y_axis * 1e6, color='k', label='Total', **kwargs)

    plt.gca().ticklabel_format(useMathText=True)

    if xlim is None:
        plt.xlim([np.min(x_axis), np.max(x_axis)])
    else:
        plt.xlim(xlim)

    plt.ylim([None, None])
    plt.xlabel(x_axis_label)
    plt.ylabel(f'Transit depth (ppm)')

    if legend:
        plt.legend()


# Others
def plot_param_effect(retrieved_parameters, spectral_model2, radtrans2):
    fig, axes = plt.subplots(nrows=len(retrieved_parameters), ncols=1, sharex='col',
                             figsize=(6.4, 3.2 * len(retrieved_parameters)))
    i = -1
    for p, dic in retrieved_parameters.items():
        i += 1
        print(i)
        sm = copy.deepcopy(spectral_model2)
        pp = copy.deepcopy(spectral_model2.model_parameters)

        for j, v in enumerate(dic['prior_parameters']):
            if 'log10_' in p and j == 0:
                del pp[p.split('log10_', 1)[-1]]

            pp[p] = v
            w, s = sm.retrieval_model_generating_function(
                radtrans2,
                pp,
                spectrum_model=sm,
                mode='transmission',
                update_parameters=True,
                telluric_transmittances=None,
                instrumental_deformations=None,
                noise_matrix=None,
                scale=True,
                shift=True,
                convolve=True,
                rebin=True,
                reduce=True
            )
            axes[i].plot(w[0, :200], s[0, 0, :200], label=f'{v:.3e}')
            if j == 0:
                axes[i].set_title(f'{p}')
            if j == 1:
                axes[i].legend()


def plot_stepfig(w, s, label, imshow=False, y=None, vmin=1, vmax=1):
    plt.figure(figsize=(12, 4))

    if imshow:
        if y[0] > 0.5:
            y0 = y[0] - 1
        else:
            y0 = y[0]
        plt.imshow(s, aspect='auto', origin='lower', extent=[w[0], w[-1], y0, y[-1]], vmin=vmin * np.min(s),
                   vmax=vmax * np.max(s))
    else:
        plt.plot(w, s)
        plt.xlim([w[0], w[-1]])
    plt.xlabel('Wavelength (µm)')
    plt.ylabel(label)
    plt.tight_layout()


def plot_hist(d, label=None, true_value=None, cmp=None, bins=15, color='C0',
              axe=None, y_label='Probability density', tight_layout=True):
    if axe is None:
        fig, axe = plt.subplots(1, 1)

    median = np.median(d)
    sm = np.quantile(d, 0.16)
    sp = np.quantile(d, 0.84)

    c = axe.hist(d, bins=bins, histtype='step', color=color, density=True)
    c = np.max(c[0]) * 1.1
    axe.vlines(median, 0, c, color=color, ls='--')
    axe.vlines(sm, 0, c, color=color, ls='--')
    axe.vlines(sp, 0, c, color=color, ls='--')

    if true_value is not None:
        axe.vlines(true_value, 0, c, color='r', ls='-')
        ts = f' ({true_value:.2f})'
    else:
        ts = ''

    if cmp is not None:
        plt.errorbar(true_value, c * 0.1, xerr=[[cmp[0]], [cmp[1]]], color='C1', capsize=2, marker='o')

    fmt = "{{0:{0}}}".format('.2f').format
    title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
    title = title.format(fmt(median), fmt(median - sm), fmt(sp - median))

    axe.set_xlabel(label + ' = ' + title + ts)

    if y_label is not None:
        axe.set_ylabel('Probability density')
    axe.set_ylim([0, c])

    if tight_layout:
        plt.tight_layout()


def plot_multiple_hists(data, labels, true_values=None, bins=15, color='C0'):
    if isinstance(data, dict):
        if labels is None:
            labels = list(data.keys())

        data = list(data.values())

    if true_values is None:
        true_values = {}

    nrows = int(np.ceil(len(data) / np.sqrt(len(data))))
    ncols = int(np.ceil(len(data) / nrows))
    fig_size = 6.4

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_size, fig_size / ncols * nrows))
    data_i = -1

    for i in range(nrows):
        for j in range(ncols):
            data_i += 1

            if data_i >= len(data):
                break

            if data_i not in true_values and isinstance(true_values, dict):
                true_values[data_i] = None

            plot_hist(
                data[data_i],
                label=labels[data_i],
                true_value=true_values[data_i],
                cmp=None,
                bins=bins,
                color=color,
                axe=axes[i, j],
                y_label=None,
                tight_layout=False
            )


# Utils
def get_contribution_density(spectral_model: SpectralModel, radtrans, wavelengths, resolving_power=8.04e4,
                             contribution=None):
    sm = copy.deepcopy(spectral_model)

    if contribution is None:
        sm.model_parameters['calculate_contribution'] = True

        wavelengths, _ = spectral_model.get_spectrum_model(
            radtrans=radtrans,
            mode='transmission',
            update_parameters=True
        )

        contribution = copy.deepcopy(radtrans.contr_tr)

    contribution_convolve = np.zeros(contribution.shape)

    for i, c in enumerate(contribution):
        contribution_convolve[i] = sm.convolve(
            input_wavelengths=wavelengths,
            input_spectrum=c,
            new_resolving_power=resolving_power,
            constance_tolerance=1e30
        )

    average_integral_contribution = [
        np.sum(np.mean(contribution_convolve, axis=1)[i:]) for i in range(len(sm.pressures))
    ]

    wh_68 = np.argwhere(np.logical_and(
        np.array(average_integral_contribution) > 0.16,
        np.array(average_integral_contribution) < 0.84)
    )
    wh_95 = np.argwhere(np.logical_and(
        np.array(average_integral_contribution) > 0.025,
        np.array(average_integral_contribution) < 0.975)
    )

    p_95_min = np.log10(np.min(sm.pressures[wh_95]))
    p_95_max = np.log10(np.max(sm.pressures[wh_95]))
    p_68_min = np.log10(np.min(sm.pressures[wh_68]))
    p_68_max = np.log10(np.max(sm.pressures[wh_68]))
    p_max = np.log10(sm.pressures[np.argmax(np.mean(contribution, axis=1))])

    print(f"Interval 95%: [{p_95_min:.2f}, {p_95_max:.2f}] [bar]")
    print(f"Interval 68%: [{p_68_min:.2f}, {p_68_max:.2f}] [bar]")
    print(f"Max: [{p_max}] [bar]")

    return contribution_convolve, \
        [10 ** p_95_min * 1e5, 10 ** p_95_max * 1e5], [10 ** p_68_min * 1e5, 10 ** p_68_max * 1e5], 10 ** p_max * 1e5


def get_parameter_range(sd, sm, retrieved_parameters):
    parameter_ranges = []
    parameter_titles = []
    parameter_labels = []
    coefficients = []

    for key, dictionary in retrieved_parameters.items():
        if key not in sd:
            print(f"Key '{key}' not in sample directory")
            continue

        # pRT corner range
        mean = np.mean(sd[key])
        std = np.std(sd[key])
        low_ref = mean - 4 * std
        high_ref = mean + 4 * std

        if 'figure_coefficient' in dictionary:
            if key == 'planet_radial_velocity_amplitude':
                figure_coefficient = sm.model_parameters['planet_radial_velocity_amplitude']
            elif key == 'planet_radius':
                figure_coefficient = sm.model_parameters['planet_radius']
            else:
                figure_coefficient = 1

            figure_coefficient *= dictionary['figure_coefficient']

            coefficients.append(dictionary['figure_coefficient'])
            low_ref *= dictionary['figure_coefficient']
            high_ref *= dictionary['figure_coefficient']
        else:
            figure_coefficient = 1
            coefficients.append(1)

        low, high = np.array(dictionary['prior_parameters']) * figure_coefficient
        low = np.max((low_ref, low))
        high = np.min((high_ref, high))

        parameter_ranges.append([low, high])

        if 'figure_label' in dictionary:
            parameter_labels.append(dictionary['figure_label'])
        else:
            parameter_labels.append(key)

        if 'figure_title' in dictionary:
            parameter_titles.append(dictionary['figure_title'])
        else:
            parameter_titles.append(None)

    return parameter_ranges, parameter_labels, parameter_titles, np.array(coefficients)


def get_species_string(string):
    """
    Get the string of a species from an Exo-REM data label.
    Example: volume_mixing_ratio_H2O -> H2O
    :param string: an Exo-REM data label
    :return: the species string
    """
    import re

    subscripts = re.findall(r'\d+', string)
    string = re.sub(r'\d+', '$_%s$', string)

    return string % tuple(subscripts)


def data_selection(wavelengths_instrument, observed_spectra, uncertainties, instrument_snr, instrumental_deformations,
                   telluric_transmittances_wavelengths, telluric_transmittances, simulated_snr, times, mid_transit_time,
                   transit_duration, orbital_phases, airmasses, barycentric_velocities, detector_selection,
                   n_transits=1, use_t23=False, use_t1535=False):
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
    telluric_transmittances_wavelengths = telluric_transmittances_wavelengths[wh[0], :]
    telluric_transmittances = telluric_transmittances[wh[0], :]
    simulated_snr = simulated_snr[:, wh[0], :]
    orbital_phases = orbital_phases[wh[0]]
    airmasses = airmasses[wh[0]]
    barycentric_velocities = barycentric_velocities[wh[0]]
    times = times[wh[0]]

    instrument_snr *= np.sqrt(n_transits)
    instrument_snr = np.ma.masked_less_equal(instrument_snr, 1)

    uncertainties = np.ma.masked_where(instrument_snr.mask, uncertainties)
    observed_spectra = np.ma.masked_where(uncertainties.mask, observed_spectra)

    return wavelengths_instrument, observed_spectra, uncertainties, instrument_snr, instrumental_deformations, \
        telluric_transmittances_wavelengths, telluric_transmittances, simulated_snr, orbital_phases, airmasses, \
        barycentric_velocities, times


def static_get_sample(output_dir, name=None):
    if name is None:
        name = output_dir.rsplit(os.sep, 1)[1]

    samples = np.genfromtxt(os.path.join(output_dir, 'out_PMN', name + '_post_equal_weights.dat'))

    with open(os.path.join(output_dir, 'out_PMN', name + '_params.json'), 'r') as f:
        parameters_read = json.load(f)

    samples_dict = {}

    for i, key in enumerate(parameters_read):
        samples_dict[key] = samples[:, i]

    return samples_dict


# Main
def main(planet_name, output_directory, additional_data_directory, mode, retrieval_name, retrieval_parameters,
         detector_selection_name, n_live_points, resume, tellurics_mask_threshold,
         retrieve_mock_observations, use_simulated_uncertainties, add_noise, n_transits, check, retrieve, archive,
         scale, shift, convolve, rebin, use_t23, use_t1535):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    retrieve_mock_3d = False

    retrieved_parameters_ref = {
        'new_resolving_power': {
            'prior_parameters': [1e4, 2e5],
            'prior_type': 'uniform',
            'figure_title': r'$\mathcal{R}$',
            'figure_label': 'Resolving power',
            'retrieval_name': 'R'
        },
        'planet_radial_velocity_amplitude': {
            'prior_parameters': np.array([0.8, 1.25]),  # Kp must be close to the true value to help the retrieval
            'prior_type': 'uniform',
            'figure_title': r'$K_p$',
            'figure_label': r'$K_p$ (km$\cdot$s$^{-1}$)',
            'figure_coefficient': 1e-5,
            'retrieval_name': 'Kp'
        },
        'planet_rest_frame_velocity_shift': {
            'prior_parameters': [-30e5, 30e5],
            'prior_type': 'uniform',
            'figure_title': r'$V_r$',
            'figure_label': r'$V_r$ (km$\cdot$s$^{-1}$)',
            'figure_coefficient': 1e-5,
            'retrieval_name': 'V0'
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
        'temperature': {
            'prior_parameters': [300, 3000],
            'prior_type': 'uniform',
            'figure_title': r'T',
            'figure_label': r'T (K)',
            'retrieval_name': 'tiso'
        },
        'CH4_hargreaves_main_iso': {
            'prior_parameters': [-12, 0],
            'prior_type': 'uniform',
            'figure_title': r'[CH$_4$]',
            'figure_label': r'$\log_{10}$(CH$_4$) MMR',
            'retrieval_name': 'CH4'
        },
        'CO_all_iso': {
            'prior_parameters': [-12, 0],
            'prior_type': 'uniform',
            'figure_title': r'[CO]',
            'figure_label': r'$\log_{10}$(CO) MMR',
            'retrieval_name': 'CO'
        },
        'H2O_main_iso': {
            'prior_parameters': [-12, 0],
            'prior_type': 'uniform',
            'figure_title': r'[H$_2$O]',
            'figure_label': r'$\log_{10}$(H$_2$O) MMR',
            'retrieval_name': 'H2O'
        },
        'H2S_main_iso': {
            'prior_parameters': [-12, 0],
            'prior_type': 'uniform',
            'figure_title': r'[H2S]',
            'figure_label': r'$\log_{10}$(H2S) MMR',
            'retrieval_name': 'H2S'
        },
        'HCN_main_iso': {
            'prior_parameters': [-12, 0],
            'prior_type': 'uniform',
            'figure_title': r'[HCN]',
            'figure_label': r'$\log_{10}$(HCN) MMR',
            'retrieval_name': 'HCN'
        },
        'NH3_main_iso': {
            'prior_parameters': [-12, 0],
            'prior_type': 'uniform',
            'figure_title': r'[NH$_3$]',
            'figure_label': r'$\log_{10}$(NH$_3$) MMR',
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
            'figure_label': r'$\log_{10}(P_c)$ ([bar])',
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
    }

    detector_selection_ref = {
        'older': np.array([4, 6, 7, 9, 10, 12, 13, 25, 26, 28, 29, 30, 46, 47, 49, 54]),
        'old': np.array([1, 6, 9, 10, 12, 22, 25, 27, 28, 29, 30, 46, 47, 54]),  # mplus_ttt0.5 r->c (old)
        'old2': np.array([1, 6, 9, 10, 22, 25, 27, 28, 29, 30, 46, 47, 52, 54]),  # mplus_ttt0.5 r->c (old2)
        'ds': np.array([1, 7, 9, 10, 22, 25, 29, 30, 44, 46, 47, 49, 52, 54]),  # mplus_ttt0.5 c->r (ds)
        'altnew': np.array([1, 7, 9, 10, 22, 25, 28, 29, 30, 46, 47, 49, 52, 54]),  # (altnew or nothing)
        'altnew2': np.array([1, 9, 10, 22, 25, 28, 29, 30, 46, 47, 49, 52, 54]),  # (altnew2)
        'strict': np.array([1, 7, 9, 25, 28, 46, 47, 49, 54]),  # (strict)
        'strict2': np.array([7, 9, 13, 25, 26, 29, 46, 47, 54]),
        'strict2_alt': np.array([7, 9, 13, 25, 30, 46]),
        'strictt14': np.array([3, 7, 9, 13, 25, 28, 29, 46]),
        'strictt1535': np.array([1, 3, 9, 14, 25, 28, 29, 30, 46, 54]),
        'strict23': np.array([1, 2, 7, 12, 13, 17, 20, 21, 24, 25, 26, 28, 29, 30, 46, 48, 52, 54]),
        'nh3d': np.array([1, 6, 7, 9, 10, 12, 13, 25, 26, 28, 29, 30, 46, 47, 49, 52, 54]),  # (nh3d)
        'nh3h2sd': np.array([1, 6, 7, 9, 10, 12, 13, 25, 26, 28, 29, 30, 32, 33, 46, 47, 49, 52, 54]),  # (nh3h2sd)
        'alex': np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 14, 15, 16, 17, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 44,
             45, 46, 47, 48, 49, 50, 51, ]),
        'alexstart': np.array(
            [0, 1, 3, 5, 6, 7, 9, 10, 13, 15, 17, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 44, 45, 46, 48, 49, 50,
             51, 52]),
        'bad': np.array([4, 5, 8, 13, 21, 23, 24, 25, 29, 44, 54]),
        'testd': np.array([46, 47])
    }

    # Initialisation
    retrieved_parameters = {}

    for parameter in retrieval_parameters:
        if parameter not in retrieved_parameters_ref:
            raise KeyError(f"parameter '{parameter}' was not initialized")
        else:
            retrieved_parameters[parameter] = retrieved_parameters_ref[parameter]

    detector_selection = detector_selection_ref[detector_selection_name]

    if retrieval_name != '':
        retrieval_name = '_' + retrieval_name

    for parameter_dict in retrieved_parameters.values():
        retrieval_name += f"_{parameter_dict['retrieval_name']}"

    retrieval_name += f'_{detector_selection_name}'

    if retrieve_mock_observations:
        retrieval_name += '_sim'

    planet = Planet.get(planet_name)

    # Overriding to Rosenthal et al. 2021
    planet.star_radius = 0.78271930158600000 * nc.r_sun
    planet.star_radius_error_upper = +0.01396094224705000 * nc.r_sun
    planet.star_radius_error_lower = -0.01396094224705000 * nc.r_sun

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

    resolving_power = 8.04e4
    retrieval_name = f"{planet_name.replace(' ', '_')}_{mode}{retrieval_name}_{n_live_points}lp"

    retrieval_directory = os.path.join(output_directory, 'retrievals', 'carmenes_retrievals', retrieval_name)

    # Load
    if rank == 0:
        print('Loading data...')

        wavelengths_instrument, observed_spectra, instrument_snr, uncertainties, orbital_phases, airmasses, \
            barycentric_velocities, times, mid_transit_time = load_carmenes_data(
                directory=os.path.join(additional_data_directory, 'carmenes', 'hd_189733_b'),
                mid_transit_jd=58004.425291
            )

        instrumental_deformations, telluric_transmittances_wavelengths, telluric_transmittances, simulated_snr = \
            load_additional_data(
                data_dir=additional_data_directory,
                wavelengths_instrument=wavelengths_instrument,
                airmasses=airmasses,
                times=times,
                resolving_power=resolving_power
            )

        wavelengths_instrument, observed_spectra, uncertainties, instrument_snr, instrumental_deformations, \
            telluric_transmittances_wavelengths, telluric_transmittances, simulated_snr, orbital_phases, airmasses, \
            barycentric_velocities, times = \
            data_selection(
                wavelengths_instrument=wavelengths_instrument,
                observed_spectra=observed_spectra,
                uncertainties=uncertainties,
                instrument_snr=instrument_snr,
                instrumental_deformations=instrumental_deformations,
                telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
                telluric_transmittances=telluric_transmittances,
                simulated_snr=simulated_snr,
                times=times,
                mid_transit_time=mid_transit_time,
                transit_duration=planet_transit_duration,
                orbital_phases=orbital_phases,
                airmasses=airmasses,
                barycentric_velocities=barycentric_velocities,
                detector_selection=detector_selection,
                n_transits=n_transits
            )

        simulated_uncertainties = np.moveaxis(
            np.moveaxis(simulated_snr, 2, 0) / np.mean(simulated_snr, axis=2) * np.mean(uncertainties, axis=2),
            0,
            2
        )

        if use_simulated_uncertainties:
            model_uncertainties = simulated_uncertainties
        else:
            model_uncertainties = copy.deepcopy(uncertainties)

        data_shape = observed_spectra.shape

        if add_noise:
            print('Generating noise...')
            noise_matrix = np.random.default_rng().normal(loc=0, scale=uncertainties, size=data_shape)
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
                'NH3_main_iso'
            ],
            rayleigh_species=['H2', 'He'],
            continuum_opacities=['H2-H2', 'H2-He'],
            opacity_mode='lbl',
            lbl_opacity_sampling=4,
            # Temperature profile parameters
            temperature_profile_mode='isothermal',
            temperature=planet.equilibrium_temperature,  # K
            # Chemical parameters
            use_equilibrium_chemistry=False,
            imposed_mass_mixing_ratios={
                'CH4_hargreaves_main_iso': 3.4e-5,
                'CO_all_iso': 1.8e-2,
                'H2O_main_iso': 5.4e-3,
                'H2S_main_iso': 1.0e-3,
                'NH3_main_iso': 7.9e-6
            },
            fill_atmosphere=True,
            # Transmission spectrum parameters (radtrans.calc_transm)
            planet_radius=planet.radius,  # cm
            planet_surface_gravity=planet.surface_gravity,  # cm.s-2
            reference_pressure=1e-2,  # bar
            cloud_pressure=1e-1,
            # Instrument parameters
            new_resolving_power=8.04e4,
            output_wavelengths=wavelengths_instrument,  # um
            # Scaling parameters
            star_radius=planet.star_radius,  # cm
            # Orbital parameters
            star_mass=planet.star_mass,  # g
            semi_major_axis=planet.orbit_semi_major_axis,  # cm
            orbital_phases=orbital_phases,
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
            constance_tolerance=1e300,  # force constant convolve
            detector_selection=detector_selection
        )

        if 'planet_radial_velocity_amplitude' in retrieved_parameters:
            retrieved_parameters['planet_radial_velocity_amplitude']['prior_parameters'] *= \
                spectral_model.model_parameters['planet_radial_velocity_amplitude']

        if 'planet_radius' in retrieved_parameters:
            retrieved_parameters['planet_radius']['prior_parameters'] *= \
                spectral_model.model_parameters['planet_radius']

        retrieval_velocities = spectral_model.get_retrieval_velocities(
            planet_radial_velocity_amplitude_range=retrieved_parameters[
                'planet_radial_velocity_amplitude']['prior_parameters'],
            planet_rest_frame_velocity_shift_range=retrieved_parameters[
                'planet_rest_frame_velocity_shift']['prior_parameters']
        )

        spectral_model.wavelengths_boundaries = spectral_model.get_optimal_wavelength_boundaries(
            relative_velocities=retrieval_velocities
        )
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

    def calculate_mean_molar_masses(mass_mixing_ratios, mean_molar_masses_offset, **kwargs):
        mmw = SpectralModel.calculate_mean_molar_masses(mass_mixing_ratios, **kwargs)

        return mmw + mean_molar_masses_offset

    spectral_model.calculate_mean_molar_masses = calculate_mean_molar_masses

    if rank == 0:
        print("Broadcasting done!")

    comm.barrier()

    radtrans = spectral_model.get_radtrans()

    comm.barrier()

    if rank == 0 and check:
        print('Validity checks...')
        validity, true_log_l, true_chi2 = validity_checks(
            simulated_data_model=copy.deepcopy(spectral_model),
            radtrans=radtrans,
            telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
            telluric_transmittances=telluric_transmittances,
            # telluric_transmittances=None,
            instrumental_deformations=instrumental_deformations,
            # instrumental_deformations=deformation_matrix,
            noise_matrix=noise_matrix,
            scale=scale,
            shift=shift,
            convolve=convolve,
            rebin=rebin
        )

        spectral_model.model_parameters['pipeline_validity'] = validity
        spectral_model.model_parameters['true_log_l'] = true_log_l
        spectral_model.model_parameters['true_chi2'] = true_chi2

    if retrieve_mock_observations:
        if rank == 0:
            if retrieve_mock_3d:
                print("Reprocessing simulated 3D data...")
            else:
                print("Initializing simulated data...")

        comm.barrier()

        if retrieve_mock_3d:
            wavelengths_instrument, reprocessed_data = get_orange_simulation_model(
                directory=os.path.join(additional_data_directory, 'carmenes', 'hd_189733_b', 'simu_orange'),
                additional_data_directory=additional_data_directory,
                base_name='range',
                mode=mode,
                telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
                telluric_transmittances=telluric_transmittances,
                instrumental_deformations=instrumental_deformations,
                noise_matrix=noise_matrix,
                scale=scale,
                shift=shift,
                convolve=convolve,
                rebin=rebin,
                reduce=True,
                **spectral_model.model_parameters
            )
        else:
            wavelengths_instrument, reprocessed_data = spectral_model.get_spectrum_model(
                radtrans=radtrans,
                mode=mode,
                update_parameters=True,
                telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
                telluric_transmittances=telluric_transmittances,
                # telluric_transmittances=None,
                instrumental_deformations=instrumental_deformations,
                # instrumental_deformations=deformation_matrix,
                noise_matrix=noise_matrix,
                scale=scale,
                shift=shift,
                convolve=convolve,
                rebin=rebin,
                reduce=True
            )

        reprocessed_data_uncertainties = copy.deepcopy(spectral_model.model_parameters['reduced_uncertainties'])
        plot_true_values = True
    else:
        reprocessed_data, reprocessing_matrix, reprocessed_data_uncertainties = spectral_model.pipeline(
            observed_spectra,
            **spectral_model.model_parameters
        )
        plot_true_values = False

    if rank == 0:
        print("Saving model...")
        if not os.path.isdir(retrieval_directory):
            os.mkdir(retrieval_directory)

        spectral_model.model_parameters['n_transits'] = n_transits
        spectral_model.save(os.path.join(retrieval_directory, 'simulated_data_model.h5'))

    if rank == 0:
        print("Initializing retrieval...")

    comm.barrier()

    retrieval_model = copy.deepcopy(spectral_model)

    retrieval = retrieval_model.init_retrieval(
        radtrans=radtrans,
        data=reprocessed_data,
        data_wavelengths=wavelengths_instrument,
        data_uncertainties=reprocessed_data_uncertainties,
        retrieval_directory=retrieval_directory,
        retrieved_parameters=retrieved_parameters,
        retrieval_name=retrieval_name,
        mode=mode,
        update_parameters=True,
        # telluric_transmittances=telluric_transmittance,
        telluric_transmittances=None,
        # instrumental_deformations=variable_throughput,
        instrumental_deformations=None,
        scale=scale,
        shift=shift,
        convolve=convolve,
        rebin=rebin,
        reduce=True,
        run_mode='retrieval',
        scattering=True
    )

    if retrieve:
        if rank == 0:
            save = True
        else:
            save = False

        spectral_model.run_retrieval(
            retrieval=retrieval,
            n_live_points=n_live_points,
            resume=resume,
            save=save
        )

        if rank == 0:
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

            if archive:
                print(f"Creating archive '{retrieval_directory}.tar.gz'...")
                root_dir, base_dir = retrieval_directory.rsplit(os.sep, 1)
                shutil.make_archive(retrieval_directory, 'gztar', root_dir, base_dir)


def _main():
    # Manual initialisation
    use_t23 = True
    use_t1535 = True
    planet_name = 'HD 189733 b'
    output_directory = os.path.abspath(os.path.abspath(os.path.dirname(__file__))
                                       + '../../../../../work/run_outputs/petitRADTRANS')
    additional_data_directory = os.path.join(output_directory, 'data')
    # output_dir = os.path.abspath(os.path.abspath(os.path.dirname(__file__))
    #                              + '../../../../run_outputs/petitRADTRANS/simulation_retrievals/CARMENES')
    # data_dir = os.path.abspath(os.path.abspath(os.path.dirname(__file__))
    #                            + '../../../../run_inputs/petitRADTRANS/additional_data')
    mode = 'transmission'

    retrieval_name = 'test_data'  # 'iso_CO_CO36_CO2_H2O_H2S'
    n_live_points = 15
    resume = False
    tellurics_mask_threshold = 0.5

    retrieve_mock_observations = True
    use_simulated_uncertainties = False
    add_noise = False
    n_transits = 1

    check = False
    retrieve = False
    archive = True

    scale = True
    shift = True
    convolve = True
    rebin = True

    detector_selection_name = 'strictt1535'

    retrieval_parameters = [
        'new_resolving_power',
        'planet_radial_velocity_amplitude',
        'planet_rest_frame_velocity_shift',
        'planet_radius',
        'log10_planet_surface_gravity',
        'temperature',
        'CH4_hargreaves_main_iso',
        'CO_all_iso',
        'H2O_main_iso',
        'H2S_main_iso',
        'NH3_main_iso',
        'log10_cloud_pressure'
    ]

    main(
        planet_name=planet_name,
        output_directory=output_directory,
        additional_data_directory=additional_data_directory,
        mode=mode,
        retrieval_name=retrieval_name,
        retrieval_parameters=retrieval_parameters,
        detector_selection_name=detector_selection_name,
        n_live_points=n_live_points,
        resume=resume,
        tellurics_mask_threshold=tellurics_mask_threshold,
        retrieve_mock_observations=retrieve_mock_observations,
        use_simulated_uncertainties=use_simulated_uncertainties,
        add_noise=add_noise,
        n_transits=n_transits,
        check=check,
        retrieve=retrieve,
        archive=archive,
        scale=scale,
        shift=shift,
        convolve=convolve,
        rebin=rebin,
        use_t23=use_t23,
        use_t1535=use_t1535
    )


if __name__ == '__main__':
    t0 = time.time()

    args = parser.parse_args()

    print(f'rm: {args.retrieve_mock_observations}')
    print(f'a: {args.no_archive}')
    print(f'convolve: {args.no_convolve}')

    main(
        planet_name=args.planet_name,
        output_directory=args.output_directory,
        additional_data_directory=args.additional_data_directory,
        mode=args.mode,
        retrieval_name=args.retrieval_name,
        retrieval_parameters=args.retrieval_parameters,
        detector_selection_name=args.detector_selection_name,
        n_live_points=args.n_live_points,
        resume=args.resume,
        tellurics_mask_threshold=args.tellurics_mask_threshold,
        retrieve_mock_observations=args.retrieve_mock_observations,
        use_simulated_uncertainties=args.use_simulated_uncertainties,
        add_noise=args.add_noise,
        n_transits=args.n_transits,
        check=args.check,
        retrieve=args.no_retrieval,
        archive=args.no_archive,
        scale=args.no_scale,
        shift=args.no_shift,
        convolve=args.no_convolve,
        rebin=args.no_rebin,
        use_t23=args.use_t23,
        use_t1535=args.use_t1535
    )

    print(f"Done in {time.time() - t0} s")
