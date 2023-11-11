"""
Run with:
    mpiexec -n N --use-hwthread-cpus python3 _test_high_resolution.py
N is the number of processes.
Try:
    sudo mpiexec -n N --allow-run-as-root ...
If for some reason the script crashes.
"""
import copy
#import copy
import json

import matplotlib.colors
#import numpy as np
from scipy.special import erf
import scipy.special

from petitRADTRANS.retrieval.preparing import preparing_pipeline
from petitRADTRANS.retrieval.retrieval import Retrieval
from petitRADTRANS.utils import calculate_reduced_chi2

from scripts._hr_retrieval_script_carmenes import *
from scripts.plot.style import *


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
    'strictt142': np.array([3, 6, 7, 9, 25, 26, 46, 47]),
    'strictt143': np.array([3, 6, 7, 9, 13, 25, 26, 29, 46, 54]),
    'strictt1535': np.array([1, 3, 9, 14, 25, 28, 29, 30, 46, 54]),  # using T_mid = 58004.425291
    'strictt15352': np.array([1, 2, 3, 7, 14, 22, 25, 28, 29, 30, 46, 52, 54]),  # using T_mid = 58004.4247
    'strictt15353': np.array([1, 3, 7, 8, 9, 14, 25, 28, 29, 30, 46, 54]),
    # using T_mid = 58004.4247, corrected for V_rest
    'strict23': np.array([1, 2, 7, 12, 13, 17, 20, 21, 24, 25, 26, 28, 29, 30, 46, 48, 52, 54]),
    'nh3d': np.array([1, 6, 7, 9, 10, 12, 13, 25, 26, 28, 29, 30, 46, 47, 49, 52, 54]),  # (nh3d)
    'nh3h2sd': np.array([1, 6, 7, 9, 10, 12, 13, 25, 26, 28, 29, 30, 32, 33, 46, 47, 49, 52, 54]),  # (nh3h2sd)
    'alex': np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 42,
         43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]),
    'alex2': np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 42,
         43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]),
    'alex3': np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 42,
         43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]),
    'alexstart': np.array(
        [0, 1, 3, 5, 6, 7, 9, 10, 13, 15, 17, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 44, 45, 46, 48, 49, 50, 51,
         52]),
    'bad': np.array([4, 5, 8, 13, 21, 23, 24, 25, 29, 44, 54]),
    'bad3': np.array([1, 9, 10, 11, 13, 18, 19, 24, 26, 27, 29, 31, 32, 33, 34]),
    'bad3-2': np.array([1,  2,  4,  7, 14, 17, 20, 21, 27, 29, 31, 32]),
    'pauls': np.array([3, 4, 5, 6, 7, 8, 9, 13, 14, 56, 27, 28, 29, 30, 31, 32, 46, 47, 48, 49, 50, 51, 52, 53, 54]),
    'nosnrnoanorm': np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18, 21,
                              22, 24, 25, 26, 28, 29, 30, 31, 33, 41, 42, 44, 46, 47, 48, 49, 50,
                              51, 52, 53, 54, 55]),
    'testd': np.array([46, 47]),
    'testd3': np.array([0, 55]),
    'testd4': np.array([1])
}


def plot_broken_yaxis(*args, axis_high, axis_low, plot_function_name, y_min, y_low, y_high, y_max,
                      markersize=12, marker_vertical_ratio=0.5, mew=1, linestyle="none", color='k', mec='k',
                      plot_colors=None, **kwargs):
    axis_high_plot_function = getattr(axis_high, plot_function_name)
    axis_low_plot_function = getattr(axis_low, plot_function_name)

    if plot_colors is not None:
        for c in plot_colors:
            axis_high_plot_function(*args, color=c, **kwargs)
            axis_low_plot_function(*args, color=c, **kwargs)
    else:
        axis_high_plot_function(*args, **kwargs)
        axis_low_plot_function(*args, **kwargs)

    axis_high.set_ylim(y_high, y_max)
    axis_low.set_ylim(y_min, y_low)

    axis_high.spines.bottom.set_visible(False)
    axis_low.spines.top.set_visible(False)

    axis_high.xaxis.tick_top()
    axis_high.tick_params(labeltop=False)  # don't put tick labels at the top

    axis_low.xaxis.tick_bottom()

    transform_kwargs = dict(marker=[(-1, -marker_vertical_ratio), (1, marker_vertical_ratio)], markersize=markersize,
                            linestyle=linestyle, color=color, mec=mec, mew=mew, clip_on=False)

    axis_high.plot([0, 1], [0, 0], transform=axis_high.transAxes, **transform_kwargs)
    axis_low.plot([0, 1], [1, 1], transform=axis_low.transAxes, **transform_kwargs)

    return axis_high, axis_low


def plot_model_steps(spectral_model, radtrans, mode, ccd_id,
                     path_outputs, xlim=None, figure_name='model_steps', image_format='pdf'):
    update_figure_font_size(MEDIUM_FIGURE_FONT_SIZE)
    plt.rc('legend', fontsize=14)  # legend fontsize
    orbital_phases = spectral_model.model_parameters['orbital_longitudes'] / 360
    mid_phase = np.argmin(np.abs(orbital_phases))

    t23 = Planet.calculate_full_transit_duration(
        total_transit_duration=spectral_model.model_parameters['transit_duration'],
        planet_radius=spectral_model.model_parameters['planet_radius'],
        star_radius=spectral_model.model_parameters['star_radius'],
        impact_parameter=Planet.calculate_impact_parameter(
            orbit_semi_major_axis=spectral_model.model_parameters['orbit_semi_major_axis'],
            orbital_inclination=spectral_model.model_parameters['orbital_inclination'],
            star_radius=spectral_model.model_parameters['star_radius']
        )
    )

    t1535 = (spectral_model.model_parameters['transit_duration'] + t23) / 2

    t_to_t0 = spectral_model.model_parameters['times'] - spectral_model.model_parameters['mid_transit_time']

    phase_35 = np.argmin(np.abs(t_to_t0 - t1535 / 2))
    phase_15 = np.argmin(np.abs(t_to_t0 + t1535 / 2))

    phase_t4 = np.argmin(np.abs(t_to_t0 - spectral_model.model_parameters['transit_duration'] / 2)) #+ 1  # +1 to get OOT
    phase_t1 = np.argmin(np.abs(t_to_t0 + spectral_model.model_parameters['transit_duration'] / 2)) #- 1  # -1 to get OOT

    # Step 1-3
    true_wavelengths_instrument, true_spectrum_instrument = spectral_model.calculate_spectrum(
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
        use_transit_light_loss=False,
        convolve=False,
        rebin=False,
        prepare=False
    )

    # Step 4
    _, spectra_scale = spectral_model.calculate_spectrum(
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
        use_transit_light_loss=False,
        convolve=False,
        rebin=False,
        prepare=False
    )

    # Step 5
    w_shift, spectra_shift = spectral_model.calculate_spectrum(
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
        use_transit_light_loss=False,
        convolve=False,
        rebin=False,
        prepare=False
    )

    # Step 6
    _, spectra_tlloss = spectral_model.calculate_spectrum(
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
        shift=True,
        use_transit_light_loss=True,
        convolve=False,
        rebin=False,
        prepare=False
    )

    # Step 7
    _, spectra_convolve = spectral_model.calculate_spectrum(
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
        use_transit_light_loss=True,
        convolve=True,
        rebin=False,
        prepare=False
    )

    # Step 8
    wavelengths_instrument, spectra_final = spectral_model.calculate_spectrum(
        radtrans=radtrans,
        mode=mode,
        update_parameters=True,
        # telluric_transmittances=telluric_transmittance,
        telluric_transmittances=None,
        # instrumental_deformations=variable_throughput,
        instrumental_deformations=None,
        noise_matrix=None,
        scale=True,
        use_transit_light_loss=True,
        shift=True,
        convolve=True,
        rebin=True,
        prepare=False
    )

    # Init
    w_shift = w_shift * 1e-6  # um to m
    wavelengths_instrument = wavelengths_instrument[ccd_id] * 1e-6  # um to m
    true_wavelengths_instrument = true_wavelengths_instrument[0] * 1e-6  # um to m
    true_spectrum_instrument = true_spectrum_instrument[0] * 1e-2  # cm to m

    features_amplitude = np.max(spectra_tlloss[mid_phase]) - np.min(spectra_tlloss[mid_phase])
    features_amplitude_factor = 0.05

    # Plots
    default_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    custom_cycle = [default_cycle[0], default_cycle[2], default_cycle[3]]

    try:
        fig = plt.figure(figsize=(6.4, 7 * 1.6))
        matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=custom_cycle)

        # Base grid
        gs0 = fig.add_gridspec(7, 1, top=0.975, bottom=0.055, left=0.15, right=0.98, hspace=0.3)

        # Steps 3 to 5: no broken axis
        gs00 = gs0[:3].subgridspec(3, 1, hspace=0.3)

        ax0 = fig.add_subplot(gs00[0])
        ax0.plot(true_wavelengths_instrument, true_spectrum_instrument, color='C2')
        ax0.tick_params(labelbottom=False)
        ax0.set_title('Step 3: base model')

        ax1 = fig.add_subplot(gs00[1], sharex=ax0)
        ax1.plot(true_wavelengths_instrument, spectra_scale[0], color='C2')
        ax1.tick_params(labelbottom=False)
        ax1.set_title('Step 4: scaling')

        ax2 = fig.add_subplot(gs00[2], sharex=ax0)
        ax2.plot(
            w_shift[phase_15], spectra_shift[phase_15], label=rf'$\Phi$ = {orbital_phases[phase_15]:.3f}', color='C0'
        )
        ax2.plot(
            w_shift[phase_35], spectra_shift[phase_35], label=rf'$\Phi$ = {orbital_phases[phase_35]:.3f}', color='C3'
        )
        ax2.tick_params(labelbottom=False)
        ax2.legend(loc=4)
        ax2.set_title('Step 5: shifting')

        # Step 6: broken axis required
        gs01 = gs0[3].subgridspec(2, 1, hspace=0.1)

        ax3_high = fig.add_subplot(gs01[0], sharex=ax0)
        ax3_low = fig.add_subplot(gs01[1], sharex=ax0)

        xs = np.array([w_shift[phase_15], w_shift[mid_phase], w_shift[phase_35]]).T
        ys = np.array([spectra_tlloss[phase_15], spectra_tlloss[mid_phase], spectra_tlloss[phase_35]]).T
        labels = [
            rf'$\Phi$ = {orbital_phases[phase_15]:.3f}',
            rf'$\Phi$ = {orbital_phases[mid_phase]:.3f}',
            rf'$\Phi$ = {orbital_phases[phase_35]:.3f}'
        ]

        ax3_high, ax3_low = plot_broken_yaxis(
            xs, ys,
            axis_high=ax3_high,
            axis_low=ax3_low,
            plot_function_name='plotlib',
            y_min=None,
            y_low=np.max(spectra_tlloss[mid_phase]) + features_amplitude_factor * features_amplitude,
            y_high=np.min(spectra_tlloss[phase_15]) - features_amplitude_factor * features_amplitude,
            y_max=None,
            label=labels
        )
        ax3_high.set_yticks([0.988, 0.990])
        ax3_low.tick_params(labelbottom=False)
        ax3_low.legend(loc=4)
        ax3_high.set_title('Step 6: adding transit effect')

        # Step 7: broken axis required
        gs02 = gs0[4].subgridspec(2, 1, hspace=0.1)

        ax4_high = fig.add_subplot(gs02[0], sharex=ax0)
        ax4_low = fig.add_subplot(gs02[1], sharex=ax0)

        features_amplitude = np.max(spectra_tlloss[mid_phase]) - np.min(spectra_tlloss[mid_phase])

        ys = np.array([spectra_convolve[phase_15], spectra_convolve[mid_phase], spectra_convolve[phase_35]]).T
        labels = [
            rf'$\Phi$ = {orbital_phases[phase_15]:.3f}',
            rf'$\Phi$ = {orbital_phases[mid_phase]:.3f}',
            rf'$\Phi$ = {orbital_phases[phase_35]:.3f}'
        ]

        ax4_high, ax4_low = plot_broken_yaxis(
            xs, ys,
            axis_high=ax4_high,
            axis_low=ax4_low,
            plot_function_name='plotlib',
            y_min=None,
            y_low=np.max(spectra_convolve[mid_phase]) + features_amplitude_factor * features_amplitude,
            y_high=np.min(spectra_convolve[phase_15]) - features_amplitude_factor * features_amplitude,
            y_max=None,
            label=labels
        )

        ax4_high.set_yticks([0.988, 0.990])
        ax4_low.tick_params(labelbottom=False)
        ax4_low.legend(loc=4)
        ax4_high.set_title('Step 7: convolving')

        # Step 8: no broken axis
        gs03 = gs0[5:].subgridspec(2, 1, hspace=0.3)

        ax5 = fig.add_subplot(gs03[0], sharex=ax0)
        ax5.pcolormesh(
            wavelengths_instrument,
            orbital_phases,
            np.moveaxis(np.moveaxis(spectra_final[ccd_id], -1, 0) / np.max(spectra_final[ccd_id], axis=-1), 0, -1),
            shading='nearest',
            cmap='viridis'
        )
        ax5.tick_params(labelbottom=False)
        ax5.set_title('Step 8: re-binning (normalised)')

        ax6 = fig.add_subplot(gs03[1], sharex=ax0)
        ax6.pcolormesh(
            wavelengths_instrument,
            orbital_phases,
            spectra_final[ccd_id],
            shading='nearest',
            cmap='viridis'
        )
        ax6.set_title('Step 8: re-binning')

        if xlim is None:
            xlim = (wavelengths_instrument[0], wavelengths_instrument[-1])

        ax6.set_xlim(xlim)
        x_ticks = ax6.get_xticks()
        ax6.set_xticks(x_ticks[1::2])
        ax6.set_xlim(xlim)
        ax6.set_ylim((orbital_phases[phase_t1], orbital_phases[phase_t4]))
        ax5.set_ylim((orbital_phases[phase_t1], orbital_phases[phase_t4]))
        ax6.set_xlabel('Wavelength (m)')

        #plt.tight_layout()

        matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=default_cycle)  # back to default

        spectral_axes = fig.add_subplot(gs0[0:1], frameon=False)
        spectral_axes.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        spectral_axes.set_ylabel(r'$m_{\theta,0}$ (m)', labelpad=20)

        spectral_axes = fig.add_subplot(gs0[1:5], frameon=False)
        spectral_axes.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        spectral_axes.set_ylabel('Arbitrary units', labelpad=20)

        spectral_axes = fig.add_subplot(gs0[5:], frameon=False)
        spectral_axes.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        spectral_axes.set_ylabel(r'$\Phi$', labelpad=20)

        plt.savefig(os.path.join(path_outputs, figure_name + '.' + image_format))
    finally:
        matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=default_cycle)  # back to default


def plot_model_steps_model(spectral_model, radtrans, mode, ccd_id,
                           telluric_transmittances_wavelengths, telluric_transmittances, instrumental_deformations,
                           noise_matrix, path_outputs, xlim=None,
                           figure_name='simulated_data_steps', image_format='pdf', noise_factor=100.0):
    update_figure_font_size(MEDIUM_FIGURE_FONT_SIZE)
    plt.rc('legend', fontsize=14)  # legend fontsize
    orbital_phases = spectral_model.model_parameters['orbital_longitudes'] / 360

    t23 = Planet.calculate_full_transit_duration(
        total_transit_duration=spectral_model.model_parameters['transit_duration'],
        planet_radius=spectral_model.model_parameters['planet_radius'],
        star_radius=spectral_model.model_parameters['star_radius'],
        impact_parameter=Planet.calculate_impact_parameter(
            orbit_semi_major_axis=spectral_model.model_parameters['orbit_semi_major_axis'],
            orbital_inclination=spectral_model.model_parameters['orbital_inclination'],
            star_radius=spectral_model.model_parameters['star_radius']
        )
    )

    t1535 = (spectral_model.model_parameters['transit_duration'] + t23) / 2

    t_to_t0 = spectral_model.model_parameters['times'] - spectral_model.model_parameters['mid_transit_time']

    phase_35 = np.argmin(np.abs(t_to_t0 - t1535 / 2))
    phase_15 = np.argmin(np.abs(t_to_t0 + t1535 / 2))

    phase_t4 = np.argmin(np.abs(t_to_t0 - spectral_model.model_parameters['transit_duration'] / 2)) #+ 1  # +1 to get OOT
    phase_t1 = np.argmin(np.abs(t_to_t0 + spectral_model.model_parameters['transit_duration'] / 2)) #- 1  # -1 to get OOT

    # Step 6 bis
    w_shift, spectra_shift = spectral_model.calculate_spectrum(
        radtrans=radtrans,
        mode=mode,
        update_parameters=True,
        telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
        telluric_transmittances=telluric_transmittances,
        instrumental_deformations=None,
        noise_matrix=None,
        scale=True,
        shift=True,
        use_transit_light_loss=True,
        convolve=False,
        rebin=False,
        prepare=False
    )

    # Step 7
    _, spectra_convolve = spectral_model.calculate_spectrum(
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
        use_transit_light_loss=True,
        convolve=True,
        rebin=False,
        prepare=False
    )

    # Step 8
    wavelengths_instrument, spectra_final = spectral_model.calculate_spectrum(
            radtrans=radtrans,
            mode=mode,
            update_parameters=True,
            telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
            telluric_transmittances=telluric_transmittances,
            instrumental_deformations=None,
            noise_matrix=None,
            scale=True,
            shift=True,
            use_transit_light_loss=True,
            convolve=True,
            rebin=True,
            prepare=False
        )

    # Step 9
    _, spectra_tt = spectral_model.calculate_spectrum(
        radtrans=radtrans,
        mode=mode,
        update_parameters=True,
        telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
        telluric_transmittances=telluric_transmittances,
        instrumental_deformations=instrumental_deformations,
        noise_matrix=None,
        scale=True,
        shift=True,
        use_transit_light_loss=True,
        convolve=True,
        rebin=True,
        prepare=False
    )

    # Step 10
    _, spectra_n = spectral_model.calculate_spectrum(
        radtrans=radtrans,
        mode=mode,
        update_parameters=True,
        telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
        telluric_transmittances=telluric_transmittances,
        instrumental_deformations=instrumental_deformations,
        noise_matrix=noise_matrix * noise_factor,
        scale=True,
        shift=True,
        use_transit_light_loss=True,
        convolve=True,
        rebin=True,
        prepare=False
    )

    # Plots
    w_shift = w_shift * 1e-6  # um to m
    wavelengths_instrument = wavelengths_instrument[ccd_id] * 1e-6  # um to m

    fig = plt.figure(figsize=(6.4, 5 * 1.6))

    # Base grid
    gs00 = fig.add_gridspec(5, 1, top=0.965, bottom=0.065, left=0.15, right=0.98, hspace=0.3)

    ax0 = fig.add_subplot(gs00[0])
    ax0.plot(w_shift[0], spectra_shift[0], label=rf'$\Phi$ = {orbital_phases[phase_15]:.3f}', color='C0')
    ax0.plot(w_shift[-1], spectra_shift[-1], label=rf'$\Phi$ = {orbital_phases[phase_35]:.3f}', color='C3', ls=':')
    ax0.tick_params(labelbottom=False)
    ax0.set_title('Step 6 bis: adding telluric transmittance')
    ax0.set_ylim([0.4, 1])
    ax0.legend(loc=4)

    ax1 = fig.add_subplot(gs00[1], sharex=ax0)
    ax1.plot(w_shift[0], spectra_convolve[0], label=rf'$\Phi$ = {orbital_phases[phase_15]:.3f}', color='C0')
    ax1.plot(w_shift[-1], spectra_convolve[-1], label=rf'$\Phi$ = {orbital_phases[phase_35]:.3f}', color='C3', ls=':')
    ax1.tick_params(labelbottom=False)
    ax1.set_title('Step 7: convolving')
    ax1.set_ylim([0.4, 1])
    ax1.legend(loc=4)

    ax2 = fig.add_subplot(gs00[2], sharex=ax0)
    ax2.pcolormesh(
        wavelengths_instrument,
        orbital_phases,
        spectra_final[ccd_id],
        shading='nearest',
        cmap='viridis'
    )
    ax2.tick_params(labelbottom=False)
    ax2.set_title('Step 8: re-binning')
    ax2.set_ylim((orbital_phases[phase_t1], orbital_phases[phase_t4]))

    ax3 = fig.add_subplot(gs00[3], sharex=ax0)
    ax3.pcolormesh(
        wavelengths_instrument,
        orbital_phases,
        spectra_tt[ccd_id],
        shading='nearest',
        cmap='viridis'
    )
    ax3.tick_params(labelbottom=False)
    ax3.set_title('Step 9: adding instrumental deformations')
    ax3.set_ylim((orbital_phases[phase_t1], orbital_phases[phase_t4]))

    ax4 = fig.add_subplot(gs00[4], sharex=ax0)
    ax4.pcolormesh(
        wavelengths_instrument,
        orbital_phases,
        spectra_n[ccd_id],
        shading='nearest',
        cmap='viridis'
    )
    ax4.set_title(f'Step 10: adding noise ({noise_factor:.0f} times increased)')
    ax4.set_ylim((orbital_phases[phase_t1], orbital_phases[phase_t4]))

    ax4.set_xlim(xlim)
    x_ticks = ax4.get_xticks()
    ax4.set_xticks(x_ticks[1::2])
    ax4.set_xlim(xlim)
    ax4.set_xlabel('Wavelength (m)')

    spectral_axes = fig.add_subplot(gs00[:2], frameon=False)
    spectral_axes.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    spectral_axes.set_ylabel(r'Arbitrary units', labelpad=20)

    spectral_axes = fig.add_subplot(gs00[2:], frameon=False)
    spectral_axes.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    spectral_axes.set_ylabel(r'$\Phi$', labelpad=20)

    plt.savefig(os.path.join(path_outputs, figure_name + '.' + image_format))


def plot_reprocessing_effect_1d(spectral_model, radtrans, uncertainties, mode,
                                telluric_transmittances_wavelengths, telluric_transmittances, instrumental_deformations,
                                ccd_id, orbital_phase_id,
                                path_outputs, xlim=None, figure_name='preparing_steps', image_format='pdf'):
    # Ref
    wavelengths_ref, spectra_ref = spectral_model.calculate_spectrum(
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
        use_transit_light_loss=True,
        convolve=True,
        rebin=True,
        prepare=False
    )

    # Start
    _, spectra_start = spectral_model.calculate_spectrum(
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
        use_transit_light_loss=True,
        convolve=True,
        rebin=True,
        prepare=False
    )

    spectra_start = np.ma.masked_where(uncertainties.mask, spectra_start)

    # Step 1
    spectra_vt_corrected, vt_matrix, vt_uncertainties = preparing_pipeline(
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
    spectra_corrected, r_matrix, r_uncertainties = preparing_pipeline(
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
    axes[1].set_title('Preparing step 1')

    axes[2].plot(
        wavelengths_ref[ccd_id], spectra_corrected[ccd_id, orbital_phase_id],
        label='reprocessed spectrum'
    )

    axes[2].set_title('Preparing step 2')
    axes[2].set_xlabel('Wavelength (m)')
    axes[2].set_xlim([wavelengths_ref[ccd_id].min(), wavelengths_ref[ccd_id].max()])
    axes[2].ticklabel_format(useOffset=True)

    #axes[-1].set_xlim([wavelengths_ref[ccd_id][0], wavelengths_ref[ccd_id][-1]])

    if xlim is None:
        xlim = (wavelengths_ref[ccd_id][0], wavelengths_ref[ccd_id][-1])

    axes[-1].set_xlim(xlim)
    x_ticks = axes[-1].get_xticks()
    axes[-1].set_xticks(x_ticks[1::2])
    axes[-1].set_xlim(xlim)

    plt.tight_layout()

    plt.savefig(os.path.join(path_outputs, figure_name + '.' + image_format))


def plot_reprocessing_effect(spectral_model, radtrans, reprocessed_data, mode, simulated_uncertainties, ccd_id,
                             telluric_transmittances_wavelengths, telluric_transmittances, instrumental_deformations,
                             noise_matrix, path_outputs,
                             use_sysrem=False, n_iterations_max=10, n_passes=1,
                             xlim=None, side_by_side=False, add_prepared_model=True,
                             figure_name='preparing_effect', image_format='pdf', save=True):
    update_figure_font_size(MEDIUM_FIGURE_FONT_SIZE)
    spectral_model_ = copy.deepcopy(spectral_model)
    spectral_model_.model_parameters['output_wavelengths'] = \
        np.array([spectral_model_.model_parameters['output_wavelengths'][ccd_id]])
    spectral_model_.model_parameters['uncertainties'] = \
        np.ma.array([spectral_model_.model_parameters['uncertainties'][ccd_id]])
    instrumental_deformations = instrumental_deformations[ccd_id]

    if not hasattr(n_passes, '__iter__'):
        n_passes = [n_passes]

    if use_sysrem:
        print("Using SysRem pipeline")
        spectral_model_.pipeline = pipeline_sys
        spectral_model_.model_parameters['preparing'] = 'SysRem'
        spectral_model_.model_parameters['n_iterations_max'] = n_iterations_max
        spectral_model_.model_parameters['convergence_criterion'] = -1
        spectral_model_.model_parameters['subtract'] = True

    if np.ndim(spectral_model_.model_parameters['uncertainties'].mask) == 0:
        spectral_model_.model_parameters['uncertainties'].mask = np.zeros(
            spectral_model_.model_parameters['uncertainties'].shape, dtype=bool
        )

    orbital_phases = spectral_model_.model_parameters['orbital_longitudes'] / 360

    wavelengths, data_noiseless = spectral_model_.calculate_spectrum(
        radtrans=radtrans,
        mode=mode,
        update_parameters=True,
        telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
        telluric_transmittances=telluric_transmittances,
        instrumental_deformations=instrumental_deformations,
        noise_matrix=None,
        scale=True,
        shift=True,
        use_transit_light_loss=True,
        convolve=True,
        rebin=True,
        prepare=False
    )

    if add_prepared_model:
        _, model = spectral_model_.calculate_spectrum(
            radtrans=radtrans,
            mode=mode,
            update_parameters=True,
            telluric_transmittances_wavelengths=None,
            telluric_transmittances=None,
            instrumental_deformations=None,
            noise_matrix=None,
            scale=True,
            shift=True,
            use_transit_light_loss=True,
            convolve=True,
            rebin=True,
            prepare=True
        )
    else:
        model = None

    # fake_model = copy.deepcopy(spectral_model_)
    # fake_model.model_parameters['uncertainties'] = np.ma.array([simulated_uncertainties[ccd_id]])
    #
    # if np.ndim(fake_model.model_parameters['uncertainties'].mask) == 0:
    #     fake_model.model_parameters['uncertainties'].mask = np.zeros(
    #         fake_model.model_parameters['uncertainties'].shape, dtype=bool
    #     )
    #
    # _, reprocessed_data_noiseless_fake = fake_model.get_spectrum_model(
    #     radtrans=radtrans,
    #     mode=mode,
    #     update_parameters=True,
    #     telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
    #     telluric_transmittances=telluric_transmittances,
    #     instrumental_deformations=instrumental_deformations,
    #     noise_matrix=None,
    #     scale=True,
    #     shift=True,
    #     use_transit_light_loss=True,
    #     convolve=True,
    #     rebin=True,
    #     reduce=True
    # )

    reprocessed_data_noiseless = []
    models = []

    for n_p in n_passes:
        spectral_model_.model_parameters['n_passes'] = n_p
        _, reprocessed_data_noiseless_ = spectral_model_.calculate_spectrum(
            radtrans=radtrans,
            mode=mode,
            update_parameters=True,
            telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
            telluric_transmittances=telluric_transmittances,
            instrumental_deformations=instrumental_deformations,
            noise_matrix=None,
            scale=True,
            shift=True,
            use_transit_light_loss=True,
            convolve=True,
            rebin=True,
            prepare=True
        )
        reprocessed_data_noiseless.append(reprocessed_data_noiseless_)

        if side_by_side:
            _, _model = spectral_model_.calculate_spectrum(
                radtrans=radtrans,
                mode=mode,
                update_parameters=True,
                telluric_transmittances_wavelengths=None,
                telluric_transmittances=None,
                instrumental_deformations=None,
                noise_matrix=None,
                scale=True,
                shift=True,
                use_transit_light_loss=True,
                convolve=True,
                rebin=True,
                prepare=True
            )
            models.append(_model)

    if use_sysrem:
        if 'n_passes' in spectral_model.model_parameters:
            spectral_model_.model_parameters['n_passes'] = copy.deepcopy(spectral_model.model_parameters['n_passes'])
        else:
            spectral_model_.model_parameters['n_passes'] = n_passes[0]

    _, reprocessed_data_noisy = spectral_model_.calculate_spectrum(
        radtrans=radtrans,
        mode=mode,
        update_parameters=True,
        telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
        telluric_transmittances=telluric_transmittances,
        instrumental_deformations=instrumental_deformations,
        noise_matrix=np.array([noise_matrix[ccd_id]]),
        scale=True,
        shift=True,
        use_transit_light_loss=True,
        convolve=True,
        rebin=True,
        prepare=True
    )

    def __get_pass_str(n_passes_):
        p_str = 'pass'

        if n_passes_ > 1:
            p_str += 'es'

        return p_str

    # Plots
    wavelengths = wavelengths[0] * 1e-6  # um to m
    n_figures = 2 + len(n_passes)

    if add_prepared_model:
        n_figures += 1

    if side_by_side:
        data = [reprocessed_data_noiseless, models]
    else:
        data = [reprocessed_data_noiseless]

    if side_by_side:
        n_columns = 2
    else:
        n_columns = 1

    fig, axes = plt.subplots(nrows=n_figures, ncols=n_columns, sharex='all', sharey='all',
                             figsize=(6.4 * n_columns, 1.6 * n_figures))

    if np.ndim(axes) == 1:
        axes = axes[:, np.newaxis]

    row_id_offset = 0

    if add_prepared_model:
        if side_by_side:
            axes[0, 1].axis('off')

        axes[0, 0].pcolormesh(
            wavelengths,
            orbital_phases,
            model[0],
            cmap='viridis'
        )

        pass_str = __get_pass_str(n_passes[0])

        if use_sysrem:
            n_passes_str = f', {n_passes[0]} {pass_str})'
        else:
            n_passes_str = ')'

        axes[0, 0].set_title(r'Prepared model')
        row_id_offset += 1

    titles = [
        r'Prepared noiseless simulated data',
        r'Prepared model'
    ]

    for i, n_p in enumerate(n_passes):
        for j, d in enumerate(data):
            axes[i + row_id_offset, j].pcolormesh(
                wavelengths,
                orbital_phases,
                d[i][0],
                cmap='viridis'
            )

            pass_str = __get_pass_str(n_p)

            if use_sysrem:
                n_passes_str = f' ({n_p} {pass_str})'
            else:
                n_passes_str = ''

            # axes[i + j].set_title(r'Prepared noiseless simulated data ($\sigma_{\mathbf{N}}$' + n_passes_str)
            axes[i + row_id_offset, j].set_title(titles[j] + n_passes_str)

    i = len(n_passes)

    if side_by_side:
        axes[i+row_id_offset, 1].axis('off')

    axes[i+row_id_offset, 0].pcolormesh(
        wavelengths,
        orbital_phases,
        reprocessed_data_noisy[0],
        cmap='viridis'
    )

    pass_str = __get_pass_str(n_passes[0])

    if use_sysrem:
        n_passes_str = f' ({n_passes[0]} {pass_str})'
    else:
        n_passes_str = ''

    #axes[i+j].set_title(r'Prepared noisy simulated data ($\sigma_{\mathbf{N}}$' + n_passes_str)
    axes[i+row_id_offset, 0].set_title(r'Prepared noisy simulated data' + n_passes_str)
    row_id_offset += 1

    if side_by_side:
        axes[i+row_id_offset, 1].axis('off')

    axes[i+row_id_offset, 0].pcolormesh(
        wavelengths,
        orbital_phases,
        reprocessed_data[ccd_id],
        cmap='viridis'
    )

    if use_sysrem:
        if 'n_passes' in spectral_model.model_parameters:
            n_p = copy.deepcopy(spectral_model.model_parameters['n_passes'])
        else:
            n_p = n_passes[0]

        n_passes_str = f" ({n_p} {pass_str})"
    else:
        n_passes_str = ''

    axes[i+row_id_offset, 0].set_title('Prepared CARMENES data' + n_passes_str)
    axes[i+row_id_offset, 0].set_xlabel('Wavelength (m)')

    if side_by_side:
        axes[i + row_id_offset - 2, 1].set_xlabel('Wavelength (m)')

        for j in range(axes[:-2, 1].size):
            axes[j, 1].yaxis.set_tick_params(labelleft=False)

    if xlim is None:
        xlim = (wavelengths[0], wavelengths[-1])

    axes[-1, 0].set_xlim(xlim)
    x_ticks = axes[-1, 0].get_xticks()
    axes[-1, 0].set_xticks(x_ticks[1::2])
    axes[-1, 0].set_xlim(xlim)

    if side_by_side:
        axes[i + row_id_offset - 2, 1].xaxis.set_tick_params(labelbottom=True)

    plt.tight_layout()

    gs = axes[0, 0].get_gridspec()

    spectral_axes = fig.add_subplot(gs[:], frameon=False)
    spectral_axes.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    spectral_axes.set_ylabel('Orbital phase', labelpad=20)

    if save:
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
        sample_dict, parameter_dict = Retrieval.get_samples(ultranest=False, names=[retrieval_name],
                                                            output_dir=f'./petitRADTRANS/__tmp/test_retrieval/'
                                                                        f'{retrieval_name}/',
                                                            name=[retrieval_name]
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
                np.mean(np.log10(sm.model_parameters['imposed_mass_fractions'][p]))
            )
        elif p not in sm.model_parameters and 'log10_' in p:
            p = p.split('log10_', 1)[1]
            true_values.append(np.mean(np.log10(sm.model_parameters[p])))
        else:
            true_values.append(np.mean(sm.model_parameters[p]))

    i = -1

    for key in sd:
        i += 1

        if i >= len(true_values):
            print(f"Retrieved parameter '{key}' not in retrieval '{expected_retrieval_directory}'")
            break

        value = retrieved_parameters[key]

        if 'figure_coefficient' in value:
            sd[key] *= value['figure_coefficient']
            true_values[i] *= value['figure_coefficient']

        if 'figure_offset' in value:
            sd[key] += value['figure_offset']
            true_values[i] += value['figure_offset']

        true_values_dict[key] = true_values[i]

    return sd, true_values_dict


def plot_partial_corners(retrieved_parameters, sd, sm, true_values, figure_directory, image_format, split_at=5):
    update_figure_font_size(11)

    parameter_ranges, _, fig_labels, fig_titles, _, _, sd = get_parameter_range(sd, sm, retrieved_parameters)

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


def prepare_plot_hist(result_directory, retrieved_parameters, true_values, sm=None):
    if np.ndim(result_directory) > 0:
        sd = []
        sample_dict = {}

        for i, directory in enumerate(result_directory):
            sd.append(static_get_sample(directory))
            sample_dict[f'{i}'] = None
    else:
        sd = [static_get_sample(result_directory)]
        sample_dict = {'': None}

    parameter_names_dict = {}
    parameter_plot_indices_dict = {}
    parameter_ranges_dict = {}
    true_values_dict = {}
    fig_titles_dict = {}
    fig_labels_dict = {}
    parameter_names_ref = None

    samples = list(sample_dict.keys())

    for i, sample in enumerate(samples):
        parameter_ranges, parameter_names, fig_labels, fig_titles, coefficients, offsets, sd[i] = \
            get_parameter_range(sd[i], retrieved_parameters, sm)

        sample_dict[sample] = np.array(list(sd[i].values())).T

        sample_dict[sample] = sample_dict[sample] * coefficients + offsets
        parameter_names_dict[sample] = parameter_names
        fig_titles_dict[sample] = fig_titles
        fig_labels_dict[sample] = fig_labels

        if i == 0:
            parameter_plot_indices_dict[sample] = np.arange(0, len(parameter_ranges))
            parameter_names_ref = np.array(parameter_names_dict[sample])
            parameter_ranges_dict[sample] = parameter_ranges
        else:
            parameter_ranges_dict[sample] = copy.deepcopy(parameter_ranges_dict[samples[0]])

            for j, range_ in enumerate(parameter_ranges):
                if parameter_names_dict[sample][j] not in parameter_names_ref:
                    raise KeyError(f"parameter '{parameter_names_dict[sample][j]}' was not in first sample")

                for k, parameter_name in enumerate(parameter_names_dict[samples[0]]):
                    if parameter_name == parameter_names_dict[sample][j]:
                        if range_[0] < parameter_ranges_dict[samples[0]][k][0]:
                            parameter_ranges_dict[samples[0]][k][0] = range_[0]

                        if range_[1] > parameter_ranges_dict[samples[0]][k][1]:
                            parameter_ranges_dict[samples[0]][k][1] = range_[1]

                        break

            parameter_plot_indices_dict[sample] = np.zeros(np.size(parameter_names_dict[sample]), dtype=int)

            for j, parameter_name in enumerate(parameter_names_dict[sample]):
                if parameter_name not in parameter_names_ref:
                    raise KeyError(f"key '{parameter_name}' "
                                   f"of sample '{sample}' not in sample '{parameter_names_ref}'")

                parameter_plot_indices_dict[sample][j] = (parameter_names_ref == parameter_name).nonzero()[0][0]

    for sample in samples[1:]:
        parameter_ranges_dict[sample] = parameter_ranges_dict[samples[0]]

        parameter_ranges_dict_tmp = copy.deepcopy(parameter_ranges_dict[sample])

        for j, plot_indice in enumerate(parameter_plot_indices_dict[sample]):
            parameter_ranges_dict_tmp[plot_indice] = parameter_ranges_dict[sample][j]

        parameter_ranges_dict[sample] = copy.deepcopy(parameter_ranges_dict_tmp)

    if true_values is not None:
        if isinstance(true_values, dict):
            if list(true_values.keys())[0] != '':
                true_values = [true_values[key] for key in sd[0]]

        for sample in sample_dict:
            true_values_dict[sample] = true_values
    else:
        true_values_dict = None

    return sample_dict, parameter_names_dict, parameter_plot_indices_dict, parameter_ranges_dict, true_values_dict, \
        fig_titles_dict, fig_labels_dict


def plot_result_corner(result_directory, retrieved_parameters,
                       figure_directory, figure_name, image_format='pdf', true_values=None, sm=None, figure_font_size=8,
                       save=True, **kwargs):
    sample_dict, parameter_names_dict, parameter_plot_indices_dict, parameter_ranges_dict, true_values_dict, \
        fig_titles, fig_labels = prepare_plot_hist(result_directory, retrieved_parameters, true_values, sm)

    if 'hist2d_kwargs' in kwargs:
        kwargs['hist2d_kwargs']['titles'] = list(fig_titles.values())[0]
    else:
        kwargs['hist2d_kwargs'] = {'titles': list(fig_titles.values())[0]}

    update_figure_font_size(figure_font_size)

    contour_corner(
        sampledict=sample_dict,
        parameter_names=fig_labels,
        output_file=None,
        parameter_plot_indices=parameter_plot_indices_dict,
        parameter_ranges=parameter_ranges_dict,
        true_values=true_values_dict,
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

    parameter_ranges, _, fig_names, fig_titles, _, _, sd = get_parameter_range(sd, retrieved_parameters, sm)

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


def plot_best_fit_comparison(exorem_file, model_directories, data=None, radtrans=None, resolving_power=None, rebin=True,
                             planet_radius_offsets=None, rebin_wavelengths=None, order_selection=None,
                             envelope=None, colors=None, labels=None, linestyles=None, linewidths=None,
                             data_color='r', data_label='', data_linestyle='', data_marker='+',
                             xlim=None, legend=True,
                             save=False, figure_directory='./', figure_name='cmp', image_format='pdf',
                             figure_font_size=18, figsize=(6.4 * 3, 4.8 * 1.5), **kwargs):
    """
    Plot the different contributions in the transmission spectrum.
    :param exorem_file: spectrum file
    :param planet_radius_offset_exorem: (m) altitude offset of the transmission spectrum
    :param cloud_altitude: (m) add an opaque cloud deck at the given altitude
    :param wvn2wvl: convert wavenumbers (cm-1) into wavelengths (m)
    :param xlim: x-axis boundaries
    :param legend: plotlib the legend
    :param exclude: list of label to exclude (e.g. ['H2O', 'clouds'])
    :param kwargs: keyword arguments for plotlib
    """
    update_figure_font_size(figure_font_size)
    fig, axe = plt.subplots(figsize=figsize)

    if planet_radius_offsets is None:
        planet_radius_offsets = [0.0] * (len(model_directories) + 1)

    if colors is None:
        colors = ['k']

        for i in range(len(model_directories)):
            colors.append(f"C{i % 8}")

    if labels is None:
        labels = [''] * (len(model_directories) + 1)

    if linestyles is None:
        linestyles = ['-'] * (len(model_directories) + 1)

    if linewidths is None:
        linewidths = [1] * (len(model_directories) + 1)

    if envelope is None:
        envelope = [False] * len(model_directories)

    exorem_dict = load_result(exorem_file)

    x_axis = np.asarray(exorem_dict['outputs']['spectra']['wavenumber'])
    x_axis = 1e-2 / x_axis  # cm-1 to m

    x_axis_label = rf'Wavelength ({wavelength_units})'

    y_axis = np.asarray(exorem_dict['outputs']['spectra']['transmission']['transit_depth'])

    if planet_radius_offsets[0] != 0:
        star_radius = exorem_dict['model_parameters']['light_source']['radius'][()]
        planet_radius_0 = star_radius * np.sqrt(y_axis)
        y_axis = ((planet_radius_0 + planet_radius_offsets[0] * 1e-2) / star_radius) ** 2

    if data is not None:
        w_min = data[:, 0] * 1e-6
        w_max = data[:, 1] * 1e-6
        spectrum = data[:, 2]
        uncertainties = data[:, 3]
        wavelengths = (w_min + w_max) / 2
        xerr = np.mean((wavelengths - w_min, w_max - wavelengths), axis=0)

        axe.errorbar(wavelengths, spectrum, yerr=uncertainties, xerr=xerr, label=data_label,
                     color=data_color, ls=data_linestyle, marker=data_marker)

    axe.plot(x_axis, y_axis * 1e6, label=labels[0], color=colors[0], ls=linestyles[0], lw=linewidths[0], **kwargs)

    sms_best_fit = []
    wavelengths_best_fits = []
    spectrum_best_fits = []

    for i, directory in enumerate(model_directories):
        print(f"Generating model from '{directory}' ({i+1}/{len(model_directories)})...")
        sm_best_fit, _, _, sd = get_best_fit_model(directory)

        if radtrans is None:
            radtrans = sm_best_fit.get_radtrans()

        if resolving_power is not None:
            sm_best_fit.model_parameters['new_resolving_power'] = resolving_power

        if rebin_wavelengths is not None:
            sm_best_fit.model_parameters['output_wavelengths'] = rebin_wavelengths

        sm_best_fit.model_parameters['planet_radius'] += planet_radius_offsets[i + 1]

        if envelope[i]:
            sigmas = (1, 3)
            parameters_sets = get_envelope_parameter_sets(sd, sigmas=sigmas)

            for sigma in sigmas[::-1]:  # reverse to get lower sigmas on top
                s_min = None
                s_max = None
                w_env = None

                for set_id, parameters_set in enumerate(parameters_sets[sigma]):
                    print(f"Generating models at {sigma} sigma ({set_id + 1}/{len(parameters_sets[sigma])})...")
                    sm_set = get_model_from_parameters(sm_best_fit, parameters_set)

                    if resolving_power is not None:
                        sm_set.model_parameters['new_resolving_power'] = resolving_power

                    w, s = sm_set.calculate_spectrum(
                        radtrans,
                        'transmission',
                        update_parameters=True,
                        scale=True,
                        convolve=True,
                        rebin=rebin,
                        shift=False,
                        use_transit_light_loss=False,
                        prepare=False
                    )

                    if w_env is None:
                        w_env = w

                    if s_min is None:
                        s_min = s

                    if s_max is None:
                        s_max = s

                    s_min = np.minimum(s_min, s)
                    s_max = np.maximum(s_max, s)

                s_min = s_min[:, 0]
                s_max = s_max[:, 0]

                if rebin:
                    for j, order in enumerate(s_min):
                        print(f"Plotting envelope of order {j}/{len(s_min) - 1}")
                        axe.fill_between(
                            w_env[j] * 1e-6, (1 - s_min[j]) * 1e6, (1 - s_max[j]) * 1e6,
                            color=colors[i + 1], label=None,
                            ls='', alpha=0.3,
                            **kwargs
                        )
                else:
                    axe.fill_between(
                        w_env * 1e-6, (1 - s_min) * 1e6, (1 - s_max) * 1e6,
                        color=colors[i + 1], label=None,
                        ls='', alpha=0.3,
                        **kwargs
                    )

        w, s = sm_best_fit.calculate_spectrum(
            radtrans,
            'transmission',
            update_parameters=True,
            scale=True,
            convolve=True,
            rebin=rebin,
            shift=False,
            use_transit_light_loss=False,
            prepare=False
        )

        sms_best_fit.append(sm_best_fit)
        wavelengths_best_fits.append(w)
        spectrum_best_fits.append(s)

        s = s[:, 0]

        if rebin:
            for j, order in enumerate(s):
                print(f"Plotting order {j}/{len(s)-1}")
                if j == 0:
                    axe.plot(w[j] * 1e-6, (1 - s[j]) * 1e6, label=labels[i + 1],
                             color=colors[i + 1], ls=linestyles[i + 1], lw=linewidths[i + 1],
                             **kwargs)
                else:
                    axe.plot(w[j] * 1e-6, (1 - s[j]) * 1e6, label=None,
                             color=colors[i + 1], ls=linestyles[i + 1], lw=linewidths[i + 1],
                             **kwargs)
        else:
            axe.plot(w * 1e-6, (1 - s) * 1e6, label=labels[i + 1],
                     color=colors[i + 1], ls=linestyles[i + 1], lw=linewidths[i + 1], **kwargs)

    axe.ticklabel_format(useMathText=True)

    if xlim is None:
        axe.set_xlim([np.min(x_axis), np.max(x_axis)])
    else:
        axe.set_xlim(xlim)

    axe.set_ylim([None, None])

    if order_selection is not None:
        for i, wvl in enumerate(rebin_wavelengths):
            if i not in order_selection:
                axe.fill_betweenx([axe.get_ylim()[0], axe.get_ylim()[1]], wvl.min() * 1e-6, wvl.max() * 1e-6,
                                  color='grey', alpha=0.3, zorder=1)

    axe.set_xlabel(x_axis_label)
    axe.set_ylabel(f'Transit depth (ppm)')
    fig.tight_layout()

    if legend:
        axe.legend()

    if save:
        plt.savefig(os.path.join(figure_directory, figure_name + '.' + image_format))

    return sms_best_fit, wavelengths_best_fits, spectrum_best_fits, x_axis, y_axis, fig, axe


def plot_validity(sm, radtrans, figure_directory, image_format, noise_matrix, sysrem=False):
    print('Polyfit pipeline...')
    validity, true_log_l, true_chi2, noiseless_validity, true_wavelengths, true_spectrum, deformed_spectrum, \
        reprocessed_spectrum, reprocessed_true_spectrum, reprocessed_deformed_spectrum, reprocessed_noisy_spectrum, \
        reprocessed_matrix_true, reprocessed_matrix_deformed, reprocessed_matrix_noisy = validity_checks(
            simulated_data_model=copy.deepcopy(sm),
            radtrans=radtrans,
            telluric_transmittances_wavelengths=sm.model_parameters['telluric_transmittances_wavelengths'],
            telluric_transmittances=sm.model_parameters['telluric_transmittances'],
            instrumental_deformations=sm.model_parameters['instrumental_deformations'],
            noise_matrix=noise_matrix,
            scale=True,
            shift=True,
            use_transit_light_loss=True,
            convolve=True,
            rebin=True,
            save=True,
            filename=os.path.join(figure_directory, 'bias_pipeline_metric.npz'),
            full=True
        )

    if sysrem:
        print('Sysrem pipeline...')
        sm_sysrem = copy.deepcopy(sm)
        sm_sysrem.model_parameters['verbose'] = True
        sm_sysrem.model_parameters['n_iterations_max'] = 15
        sm_sysrem.model_parameters['convergence_criterion'] = -1
        sm_sysrem.model_parameters['tellurics_mask_threshold'] = 0.8

        sm_sysrem.pipeline = pipeline_sys

        validity_s, true_log_l_s, true_chi2_s, noiseless_validity_s, true_wavelengths_s, true_spectrum_s, \
            deformed_spectrum_s, reprocessed_spectrum_s, reprocessed_true_spectrum_s, reprocessed_deformed_spectrum_s, \
            reprocessed_noisy_spectrum_s, reprocessed_matrix_true_s, reprocessed_matrix_deformed_s, \
            reprocessed_matrix_noisy_s = validity_checks(
                simulated_data_model=sm_sysrem,
                radtrans=radtrans,
                telluric_transmittances_wavelengths=sm.model_parameters['telluric_transmittances_wavelengths'],
                telluric_transmittances=sm.model_parameters['telluric_transmittances'],
                instrumental_deformations=sm.model_parameters['instrumental_deformations'],
                noise_matrix=noise_matrix,
                scale=True,
                shift=True,
                convolve=True,
                use_transit_light_loss=True,
                rebin=True,
                save=True,
                filename=os.path.join(figure_directory, 'bias_pipeline_metric_sysrem.npz'),
                full=True
            )

    print('Plotting figure...')
    update_figure_font_size(MEDIUM_FIGURE_FONT_SIZE)

    plt.figure(figsize=(6.4, 4.8))

    if sysrem:
        axe = plot_hist(
            np.ma.log10(np.abs(validity_s).flatten()),
            color='C3',
            y_max=0.95
        )
        axe = plot_hist(
            np.ma.log10(np.abs(validity).flatten()),
            r'$\log_{10}$(|BPM|)',
            color='C0',
            axe=axe,
            y_max=0.95
        )
    else:
        axe = plot_hist(
            np.ma.log10(np.abs(validity).flatten()),
            r'$\log_{10}$(|BPM|)',
            color='C0',
            y_max=0.95
        )

    no_pipeline = np.ma.log10(
        np.abs(1 - (true_spectrum + noise_matrix) / (deformed_spectrum + noise_matrix))
    )
    data_only_pipeline = np.ma.log10(
        np.abs(1 - (true_spectrum + noise_matrix * reprocessed_matrix_noisy) / reprocessed_noisy_spectrum)
    )

    if sysrem:
        colors = ['k', 'C1', 'C0', 'C3']
        labels = ['No pipeline', 'Data-only pipeline', 'Data+model pipeline', 'Data+model Sys-Rem']
    else:
        colors = ['k', 'C1', 'C0']
        labels = ['No pipeline', 'Data-only pipeline', 'Data+model pipeline']

    for i, d in enumerate([no_pipeline, data_only_pipeline]):
        d = np.ma.masked_invalid(d)
        d = d[~d.mask]
        plt.vlines(np.median(d), 0, 1.1, color=colors[i], ls='--')
        plt.vlines(np.quantile(d, 0.16), 0, 1.1, color=colors[i], ls='--')
        plt.vlines(np.quantile(d, 0.84), 0, 1.1, color=colors[i], ls='--')

    for i, c in enumerate(colors):
        plt.plot([-np.inf, -np.inf], color=c, label=labels[i])

    plt.legend(loc=2)
    plt.savefig(os.path.join(figure_directory, 'bpm' + '.' + image_format))

    plt.figure()
    plt.pcolormesh(
        sm.model_parameters['output_wavelengths'][0],
        sm.model_parameters['times'],
        reprocessed_deformed_spectrum[0]
    )

    plt.figure()
    plt.pcolormesh(
        sm.model_parameters['output_wavelengths'][0],
        sm.model_parameters['times'],
        reprocessed_noisy_spectrum[0]
    )

    return validity, noiseless_validity, reprocessed_deformed_spectrum, reprocessed_noisy_spectrum


def plot_contribution(sm, radtrans, figure_directory, image_format):
    plt.figure()
    plt.imshow(radtrans.transmission_contribution, aspect='auto', origin='upper',
               extent=[np.min(sm.wavelengths) * 1e-6, np.max(sm.wavelengths) * 1e-6, np.log10(sm.table_pressures[-1]) + 5,
                       np.log10(sm.table_pressures[0]) + 5])
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


def plot_3d_model(model_directory, pressure_target=1e4, save=True, figure_directory='./', image_format='pdf'):
    # Load data
    lon = np.load(os.path.join(model_directory, 'long.npy'))
    lat = np.load(os.path.join(model_directory, 'lat.npy'))
    temperatures = np.load(os.path.join(model_directory, 'T.npy'))
    pressures = np.load(os.path.join(model_directory, 'P.npy'))

    # Find pressures corresponding to the desired pressure for interpolation
    p_high_indices = np.argmin(np.abs(pressures - pressure_target), axis=0)

    pressure_high = np.zeros((lat.size, lon.size))
    pressure_low = np.zeros((lat.size, lon.size))
    p_low_indices = np.zeros((lat.size, lon.size), dtype=int)

    for i in range(lat.size):
        for j in range(lon.size):
            pressure_high[i, j] = pressures[p_high_indices[i, j], i, j]

            if pressure_high[i, j] > pressure_target:
                if p_high_indices[i, j] < pressures.shape[0] - 1:
                    p_low_indices[i, j] = p_high_indices[i, j] + 1  # take lower pressure from the level above
                else:
                    # the top of the atmosphere is reached
                    warnings.warn(f"target pressure {pressure_target} is above the top of the atmosphere")
                    p_low_indices[i, j] = p_high_indices[i, j]
                    p_high_indices[i, j] = p_high_indices[i, j] - 1

                pressure_low[i, j] = pressures[p_low_indices[i, j], i, j]
            else:
                if p_high_indices[i, j] > 0:
                    p_low_indices[i, j] = p_high_indices[i, j]  # the lower pressure level is already found
                    p_high_indices[i, j] = p_high_indices[i, j] - 1  # take higher pressure from the level below
                else:
                    # the bottom of the atmosphere is reached
                    warnings.warn(f"target pressure {pressure_target} is below the bottom of the atmosphere")
                    p_low_indices[i, j] = 1
                    p_high_indices[i, j] = 0

                pressure_high[i, j] = pressures[p_high_indices[i, j], i, j]
                pressure_low[i, j] = pressures[p_low_indices[i, j], i, j]

    # Adjust indices for temperatures: pressures start 1 altitude level below temperatures
    t_indices1 = np.zeros(p_high_indices.shape, dtype=int)
    t_indices1[np.greater(p_high_indices, 0)] = p_high_indices[np.greater(p_high_indices, 0)] - 1

    t_indices2 = np.zeros(p_low_indices.shape, dtype=int)
    t_indices2[np.greater(p_low_indices, 0)] = p_low_indices[np.greater(p_low_indices, 0)] - 1

    # Log-interpolate temperatures to the desired pressure
    temp = np.zeros((lat.size, lon.size))

    for i in range(lat.size):
        for j in range(lon.size):
            temp[i, j] = np.interp(
                np.log(pressure_target),
                (np.log(pressure_low[i, j]), np.log(pressure_high[i, j])),
                (temperatures[t_indices2[i, j], i, j], temperatures[t_indices1[i, j], i, j])
            )

    # Plot figure
    update_figure_font_size(MEDIUM_FIGURE_FONT_SIZE)

    plt.figure(figsize=(6.4, 4.8))
    #divnorm = matplotlib.colors.TwoSlopeNorm(vmin=, vcenter=1209)
    plt.pcolormesh(lon, lat, temp, cmap='plasma')
    plt.colorbar(label='Temperature (K)')
    cs = plt.contour(lon, lat, temp, levels=[1000, 1209, 1300], colors=['k'])
    plt.gca().clabel(cs, cs.levels, inline=True, fontsize=10)
    plt.vlines([90, 270], -90, 90, colors='k', ls=':')
    plt.xlabel(r'Longitude ($\circ$)')
    plt.ylabel(r'Latitude ($\circ$)')
    plt.tight_layout()

    if save:
        plt.savefig(
            os.path.join(figure_directory, f'model_3d_temperature_{pressure_target:.1e}pa' + '.' + image_format)
        )


def plot_planet_distribution(data_directory, planet, figure_directory):
    eu = np.genfromtxt(data_directory, delimiter='\t', dtype=None, names=True)
    wh = np.where(eu['pl_controv_flag'] == 0)
    masses = eu['pl_bmasse'][wh]
    masses_err_min = eu['pl_bmasseerr2'][wh]
    masses_err_max = eu['pl_bmasseerr1'][wh]
    radius = eu['pl_rade'][wh]
    radius_err_min = eu['pl_radeerr2'][wh]
    radius_err_max = eu['pl_radeerr1'][wh]

    masses[np.where(masses == 0)] = np.nan
    masses_err_min[np.where(masses_err_min == 0)] = np.nan
    masses_err_max[np.where(masses_err_max == 0)] = np.nan
    radius[np.where(radius == 0)] = np.nan
    radius_err_min[np.where(radius_err_min == 0)] = np.nan
    radius_err_max[np.where(radius_err_max == 0)] = np.nan

    # wh = np.where(
    #    np.logical_not(np.logical_or(np.isnan(masses), np.isnan(radius)))
    # )
    wh = np.where(radius)
    print(f'selected = {np.size(wh)}')
    masses = masses[wh].astype(float)
    masses_err_min = np.abs(masses_err_min[wh].astype(float))
    masses_err_max = np.abs(masses_err_max[wh].astype(float))
    radius = radius[wh].astype(float)
    radius_err_min = np.abs(radius_err_min[wh].astype(float))
    radius_err_max = np.abs(radius_err_max[wh].astype(float))

    wh = np.where(
        np.logical_and(
            np.max((masses_err_min, masses_err_max), axis=0) < 0.15 * masses,
            np.max((radius_err_min, radius_err_max), axis=0) < 0.15 * radius,
        )
    )

    print(f'plotted = {np.size(wh)}')

    update_figure_font_size(MEDIUM_FIGURE_FONT_SIZE)
    fig = plt.figure()
    gs = fig.add_gridspec(4, 4)

    fig_ax1 = fig.add_subplot(gs[:-1, 1:])
    plt.plot([2, 2], np.array([0, 1e10]) * 1e5 / cst.r_earth, ls='--', color='C7', lw=1)
    plt.plot([0.03 * cst.m_jup / cst.m_earth, 0.03 * cst.m_jup / cst.m_earth],
             np.array([0, 1e10]) * 1e5 / cst.r_earth, ls=':', color='lightgray',
             lw=1)
    plt.text(2, 2e5 * 1e5 / cst.r_earth, 'Rocky planets', rotation=90, ha='right', va='top', c='C7', fontsize=12,
             zorder=4)
    plt.text(4.44, 1.6e5 * 1e5 / cst.r_earth, 'Super-earths/\nSub-Neptunes', rotation=90, ha='center', va='top', c='C7',
             fontsize=10, zorder=4)
    plt.text(0.03 * cst.m_jup / cst.m_earth, 1.6e5 * 1e5 / cst.r_earth, '?', rotation=0, ha='left', va='center', c='C7',
             fontsize=10, bbox=dict(fc='w', ec='none'), zorder=4)
    plt.annotate('', (2, 1.6e5 * 1e5 / cst.r_earth), (0.03 * cst.m_jup / cst.m_earth, 1.6e5 * 1e5 / cst.r_earth),
                 arrowprops=dict(arrowstyle='<|-|>', fc='C7', ec='C7'), zorder=4)
    plt.text(0.41 * cst.m_jup / cst.m_earth, 2e5 * 1e5 / cst.r_earth, 'Ice giants ', rotation=0, ha='right', va='top',
             c='C7', fontsize=12, zorder=4)
    plt.text(0.41 * cst.m_jup / cst.m_earth, 2e5 * 1e5 / cst.r_earth, ' Gas giants', rotation=0, ha='left', va='top',
             c='C7', fontsize=12, zorder=4)
    plt.plot([0.41 * cst.m_jup / cst.m_earth, 0.41 * cst.m_jup / cst.m_earth], np.array([0, 1e10]) * 1e5 / cst.r_earth,
             ls='--', color='C7', lw=1, zorder=4)

    plt.errorbar(
        masses[wh], radius[wh],
        xerr=(masses_err_min[wh], masses_err_max[wh]),
        yerr=(radius_err_min[wh], radius_err_max[wh]),
        ls='', marker='+', color='darkgrey'
    )
    plt.errorbar(
        np.array([planet.mass]) / cst.m_earth, np.array([planet.radius]) / cst.r_earth,
        xerr=(np.array([np.abs(planet.mass_error_lower)]) / cst.m_earth, np.array([planet.mass_error_upper]) / cst.m_earth),
        yerr=(np.array([np.abs(planet.radius_error_lower)]) / cst.r_earth, np.array([planet.radius_error_upper]) / cst.r_earth),
        ls='', marker='o', color='r'
    )
    plt.plot(
        np.array([planet.mass, planet.mass]) / cst.m_earth,
        np.array([1e-300, planet.radius]) / cst.r_earth, c='r', ls=':', zorder=3
    )
    plt.plot(
        np.array([1e-300, planet.mass]) / cst.m_earth,
        np.array([planet.radius, planet.radius]) / cst.r_earth, c='r', ls=':', zorder=3
    )
    plt.text(planet.mass / cst.m_earth, planet.radius / cst.r_earth, planet.name,
             ha='left', va='bottom', c='r', fontsize=16)

    plt.errorbar(
        1, 1,
        ls='', marker='o', color='k'
    )
    plt.text(1, 1, 'Earth', ha='left', va='top', c='k', fontsize=16)

    plt.errorbar(
        1.02413e26 * 1e3 / cst.m_earth, 24622 * 1e5 / cst.r_earth,
        ls='', marker='o', color='b'
    )
    plt.text(1.02413e26 * 1e3 / cst.m_earth, 24622 * 1e5 / cst.r_earth, 'Neptune', ha='left', va='bottom', c='b',
             fontsize=16)

    # plt.errorbar(
    #     np.array([8.6810e25]) * 1e3 / cst.m_earth, np.array([25362]) * 1e5 / cst.r_earth,
    #     ls='', marker='o', color='c'
    # )
    # plt.text(8.6810e25 * 1e3 / cst.m_earth, 25362 * 1e5 / cst.r_earth, 'Uranus', ha='right', va='bottom', c='c', fontsize=16)

    plt.errorbar(
        np.array([cst.m_jup]) / cst.m_earth, np.array([cst.r_jup]) / cst.r_earth,
        ls='', marker='o', color='C1'
    )
    plt.text(cst.m_jup / cst.m_earth, cst.r_jup / cst.r_earth, 'Jupiter', ha='left', va='top', c='C1', fontsize=16,
             zorder=4)

    rad_xlim = 749
    mass_ylim = 499
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(np.array([5e24, 1e29]) * 1e3 / cst.m_earth)
    plt.ylim(np.array([5e3, 2e5]) * 1e5 / cst.r_earth)
    plt.setp(fig_ax1.get_xticklabels(), visible=False)
    plt.setp(fig_ax1.get_yticklabels(), visible=False)
    plt.minorticks_on()

    fig.add_subplot(gs[-1, 1:], sharex=fig_ax1)
    plt.hist(masses, bins=10 ** np.linspace(24, 29, 51) * 1e3 / cst.m_earth, edgecolor='black', color='w')
    plt.plot(np.array([planet.mass, planet.mass]) / cst.m_earth, np.array([0, mass_ylim]), c='r', ls=':')
    plt.semilogx()
    plt.xlabel(r'Mass (M$_\oplus$)')
    # plt.ylabel('Count')
    plt.ylim([0, mass_ylim])
    plt.minorticks_on()

    fig.add_subplot(gs[:-1, 0], sharey=fig_ax1)
    plt.hist(radius, bins=10 ** np.linspace(3.5, 5.5, 51) * 1e5 / cst.r_earth,
             orientation='horizontal', edgecolor='black', color='w')
    plt.plot(np.array([0, rad_xlim]),
             np.array([planet.radius, planet.radius]) / cst.r_earth, c='r', ls=':')
    plt.semilogy()
    plt.ylabel(r'Radius (R$_\oplus$)')
    plt.xlabel('Count      ')
    plt.xlim([0, rad_xlim])
    plt.minorticks_on()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(os.path.join(figure_directory, 'exoplanet_distribution.pdf'))


def plot_ccf(v_rest, kps, ccf, title=None, true_vr=None, true_kp=None):
    plt.pcolormesh(v_rest * 1e-5, kps * 1e-5, ccf)

    if true_vr is not None:
        plt.vlines(true_vr * 1e-5, np.min(kps) * 1e-5, np.max(kps) * 1e-5, color='w', ls='--')

    if true_kp is not None:
        plt.hlines(true_kp * 1e-5, np.min(v_rest) * 1e-5, np.max(v_rest) * 1e-5, color='w', ls='--')

    plt.xlabel(r"$V_r$ (km$\cdot$s$^{-1}$)")
    plt.ylabel(r"$K_p$ (km$\cdot$s$^{-1}$)")

    if title:
        plt.title(title)


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
    :param legend: plotlib the legend
    :param exclude: list of label to exclude (e.g. ['H2O', 'clouds'])
    :param kwargs: keyword arguments for plotlib
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

    if 'all' not in exclude:
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
            w, s, _ = sm.retrieval_model_generating_function(
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
                prepare=True
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
              axe=None, y_max=None, y_label='Probability density', tight_layout=True, fmt='.2f'):
    if axe is None:
        fig, axe = plt.subplots(1, 1)

    median = np.median(d)
    sm = np.quantile(d, 0.16)
    sp = np.quantile(d, 0.84)

    c = axe.hist(d, bins=bins, histtype='step', color=color, density=True)

    if y_max is None:
        y_max = np.max(c[0]) * 1.1

    axe.vlines(median, 0, y_max, color=color, ls='--')
    axe.vlines(sm, 0, y_max, color=color, ls='--')
    axe.vlines(sp, 0, y_max, color=color, ls='--')

    if true_value is not None:
        axe.vlines(true_value, 0, y_max, color='r', ls='-')
        ts = f' ({true_value:.2f})'
    else:
        ts = ''

    if cmp is not None:
        plt.errorbar(true_value, y_max * 0.1, xerr=[[cmp[0]], [cmp[1]]], color='C1', capsize=2, marker='o')

    fmt = "{{0:{0}}}".format(fmt).format
    title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
    title = title.format(fmt(median), fmt(median - sm), fmt(sp - median))

    if label is not None:
        axe.set_xlabel(label + ' = ' + title + ts)

    if y_label is not None:
        axe.set_ylabel('Probability density')

    axe.set_ylim([0, y_max])

    if tight_layout:
        plt.tight_layout()

    return axe


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


def plot_multiple_data(x, y, data, **kwargs):
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

            axes[i, j].pcolormesh(
                x[data_i],
                y,
                data[data_i],
                **kwargs
            )

            axes[i, j].set_title(data_i)


def plot_multiple_hists_data(result_directory, retrieved_parameters, log_evidences=None, true_values=None,
                             add_rectangle=None,
                             parameter_names_ref=None, bins=15, color='C0', figure_font_size=11, use_titles=True,
                             save=False, figure_directory='./', figure_name='fig', image_format='png', fig_size=19.2):
    if isinstance(result_directory, dict):
        result_names = list(result_directory.keys())
        result_directory = list(result_directory.values())

        if log_evidences is not None:
            result_names = [result_name + f' ({log_evidences[i]:.2f})' for i, result_name in enumerate(result_names)]
    else:
        result_names = None

    sample_dict, parameter_names_dict, parameter_plot_indices_dict, parameter_ranges_dict, true_values_dict, \
        fig_titles, _ = prepare_plot_hist(result_directory, retrieved_parameters, true_values)

    lengths = [len(names) for names in parameter_names_dict.values()]

    nrows = len(sample_dict)
    ncols = np.max(lengths)

    fig_titles_ref = []
    id_ref = np.zeros(np.max(lengths), dtype=int)

    if parameter_names_ref is None:
        parameter_names_ref = parameter_names_dict[str(np.argmax(lengths))]
        id_ref = np.arange(0, np.max(lengths))

        if use_titles:
            fig_titles_ref = fig_titles[str(np.argmax(lengths))]
        else:
            fig_titles_ref = parameter_names_dict[str(np.argmax(lengths))]
    else:
        for sample_id, parameter_name in enumerate(parameter_names_ref):
            if parameter_name not in parameter_names_dict[str(np.argmax(lengths))]:
                raise ValueError(f"unknown parameter '{parameter_name}'")

            for j, pn in enumerate(parameter_names_dict[str(np.argmax(lengths))]):
                if pn == parameter_name:
                    id_ref[sample_id] = j

                    if use_titles:
                        fig_titles_ref.append(fig_titles[str(np.argmax(lengths))][j])
                    else:
                        fig_titles_ref.append(parameter_names_dict[str(np.argmax(lengths))][j])

                    break

    update_figure_font_size(figure_font_size)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_size, fig_size / ncols * nrows), sharex='col')

    if np.ndim(axes) == 1:
        axes = np.array([axes])

    max_col = 0

    # Put last key at the end so that the largest sample is at the bottom of the figure, showing all x-axis tick labels
    _sample_dict = {key: sample_dict[key] for key in list(sample_dict.keys())[1:]}
    _sample_dict[list(sample_dict.keys())[0]] = sample_dict[list(sample_dict.keys())[0]]
    i = 0

    for sample_id in _sample_dict:
        if isinstance(color, dict):
            if result_names[int(sample_id)] in color:
                c = color[result_names[int(sample_id)]]
            else:
                raise ValueError(f"Title {result_names[int(sample_id)]} not found in color")
        else:
            c = color

        for j in range(ncols):
            axes[i, j].set_xlim(parameter_ranges_dict[list(sample_dict.keys())[0]][id_ref[j]])

            if parameter_names_ref[j] not in parameter_names_dict[sample_id]:
                axes[i, j].axis('off')
            else:
                if int(sample_id) == add_rectangle:
                    max_col = j

                for k, parameter_name in enumerate(parameter_names_dict[sample_id]):
                    if parameter_name not in parameter_names_ref:
                        raise KeyError(f"unknown parameter '{parameter_name}'")
                    elif parameter_name == parameter_names_ref[j]:
                        if parameter_name == 'new_resolving_power':
                            fmt = '.0f'
                        else:
                            fmt = '.2f'

                        plot_hist(
                            _sample_dict[sample_id].T[k],
                            label=fig_titles_ref[j],
                            true_value=true_values,
                            cmp=None,
                            bins=bins,
                            color=c,
                            axe=axes[i, j],
                            y_label=None,
                            tight_layout=False,
                            fmt=fmt
                        )
                        axes[i, j].set_yticks([])

                        if parameter_name == 'new_resolving_power' and i == len(sample_dict) - 1:
                            x_ticks = axes[i, j].get_xticks()

                            if len(x_ticks) > 2:
                                x_ticks = x_ticks[::2]
                                axes[i, j].set_xticks(x_ticks[::2])

                        break

        if result_names is not None:
            axes[i, 0].set_ylabel(result_names[int(sample_id)])

        i += 1

    fig.tight_layout(rect=(0, 0, 0.99, 1))

    if add_rectangle is not None:
        bbox0 = axes[add_rectangle, 0].get_tightbbox(fig.canvas.get_renderer())
        bbox1 = axes[add_rectangle, max_col].get_tightbbox(fig.canvas.get_renderer())
        x0, y0, width0, height = bbox0.transformed(fig.transFigure.inverted()).bounds
        x1, y1, width1, _ = bbox1.transformed(fig.transFigure.inverted()).bounds

        width = x1 - x0 + width1 / 4
        print(width, width0, width1, height, x1, x0)

        # slightly increase the very tight bounds:
        xpad0 = np.min((0.05 * width, x0 * 0.80))
        xpad1 = np.min((0.05 * width, 0.05 / 2))
        ypad = 0.05 * height
        fig.add_artist(
            plt.Rectangle(
                (x0 - xpad0, y0 - ypad),
                width + 2 * xpad1,
                height + 2 * ypad,
                edgecolor='k', linewidth=2, fill=False
            )
        )

    if save:
        plt.savefig(os.path.join(figure_directory, figure_name + '.' + image_format))


def all_best_fit_models(directories, additional_data_directory, resolving_power, planet, detector_selection_ref,
                        uncertainties_correction_factor=1.171, n_transits=1, calculate=False, build_table=True):
    wavelengths = []
    spectra = []
    spectral_models = []
    log_es = []
    log_ls = []
    chi2s = []
    k_sigmas = []

    print('Loading data...')
    wavelengths_instrument, observed_spectra, instrument_snr, uncertainties, orbital_phases, airmasses, \
        barycentric_velocities, times, mid_transit_time = load_carmenes_data(
            directory=os.path.join(additional_data_directory, 'carmenes', 'hd_189733_b'),
            mid_transit_jd=58004.42319302507
        )

    instrumental_deformations, telluric_transmittances_wavelengths, telluric_transmittances, simulated_sn, _ = \
        load_additional_data(
            data_dir=additional_data_directory,
            wavelengths_instrument=wavelengths_instrument,
            airmasses=airmasses,
            resolving_power=resolving_power
        )

    wavelengths_boundaries = np.array([np.inf, -np.inf])
    radtrans = None

    for i, directory in enumerate(directories):
        print(f"\nCalculating for directory '{directory}' ({i + 1}/{len(directories)})...")
        detector_selection = None

        for ds in detector_selection_ref:
            if '_' + ds + '_' in directory:
                print(f"Using detector selection '{ds}'")
                detector_selection = detector_selection_ref[ds]
                break

        if detector_selection is None:
            raise ValueError("Detector selection not in reference detector selections")

        if '_t23_' in directory:
            print(f"Using full transit time (T23), not total transit time (T14)")
            planet_transit_duration = planet.calculate_full_transit_duration(
                total_transit_duration=planet.transit_duration,
                planet_radius=planet.radius,
                star_radius=planet.star_radius,
                impact_parameter=planet.calculate_impact_parameter(
                    orbit_semi_major_axis=planet.orbit_semi_major_axis,
                    orbital_inclination=planet.orbital_inclination,
                    star_radius=planet.star_radius
                )
            )

            if '_t1535_' in directory:
                print(f"Adding exposures of half-eclipses")
                planet_transit_duration += (planet.transit_duration - planet_transit_duration) / 2
        else:
            print(f"Using total transit time (T14)")
            planet_transit_duration = planet.transit_duration

        wavelengths_instrument_, observed_spectra_, uncertainties_, instrument_snr_, instrumental_deformations_, \
            telluric_transmittances_wavelengths_, telluric_transmittances_, simulated_snr_, orbital_phases_, \
            airmasses_, barycentric_velocities_, times_ = \
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

        print('Loading model...')
        spectral_model = SpectralModel.load(os.path.join(directory, 'simulated_data_model.h5'))

        print(spectral_model.model_parameters['uncertainties'].shape, observed_spectra_.shape)

        if not np.all(np.in1d(spectral_model.wavelengths_boundaries, wavelengths_boundaries)) and calculate:
            print(f'Updating radtrans '
                  f'(expected: {wavelengths_boundaries}, got: {spectral_model.wavelengths_boundaries})')
            wavelengths_boundaries = spectral_model.wavelengths_boundaries
            radtrans = spectral_model.get_radtrans()
        else:
            radtrans = None

        print('Reprocessing data...')
        reprocessed_data, _, reprocessed_data_uncertainties = spectral_model.pipeline(
            observed_spectra_,
            wavelengths=wavelengths_instrument_,
            **spectral_model.model_parameters
        )

        w, s, smbf, log_e, log_l, chi2, k_sigma, _ = get_best_fit_model(
            directory,
            radtrans,
            reprocessed_data,
            reprocessed_data_uncertainties,
            uncertainties_correction_factor=uncertainties_correction_factor
        )

        wavelengths.append(w)
        spectra.append(s)
        spectral_models.append(spectral_model)
        log_es.append(log_e)
        log_ls.append(log_l)
        chi2s.append(chi2)
        k_sigmas.append(k_sigma)

    if build_table:
        for i, directory in enumerate(directories):
            print(f"{directory}\t& {log_es[i]:.2f}\t& {log_ls[i]:.2f}\t& {k_sigmas[i]:.3f}\t& {chi2s[i]:.3f}")

    return wavelengths, spectra, spectral_models, log_es, log_ls, chi2s


def all_best_fit_stats(directories, build_table=True, retrieval_name='R-', retrieved_parameters=None):
    spectral_models = []
    log_es = []
    log_ls = []
    chi2s = []
    k_sigmas = []
    sds = []

    for i, directory in enumerate(directories):
        print(f"Loading '{directory}' ({i+1}/{len(directories)})...")
        sm_best_fit, log_evidence, log_l_best_fit, sd = get_best_fit_model(directory, None, None, None, 1)
        rp = np.load(os.path.join(directory, 'retrieved_parameters.npz'))

        data = np.ma.masked_where(rp['data_mask'], rp['data'])
        data_uncertainties = np.ma.masked_where(rp['data_mask'], rp['data_uncertainties'])
        data_mask = rp['data_mask']

        uncertainties_correction_factor = np.ma.mean(
            np.ma.mean(data_uncertainties, axis=2) / np.ma.std(data, axis=2)
        )

        chi2_best_fit = log_l_best_fit * -2 / np.size(data[~data_mask]) * uncertainties_correction_factor ** 2

        log_es.append(log_evidence)
        log_ls.append(log_l_best_fit)
        k_sigmas.append(uncertainties_correction_factor)
        chi2s.append(chi2_best_fit)
        sds.append(sd)
        spectral_models.append(sm_best_fit)

    if build_table:
        log_es = np.array(log_es)
        delta_log_es = log_es - np.min(log_es)
        retrievals_titles = []

        for i, directory in enumerate(directories):
            _, _, _, fig_titles, _, _, sds[i] = get_parameter_range(sds[i], retrieved_parameters, None)
            fig_titles = [fig_title.replace('[', '').replace(']', '') for fig_title in fig_titles]
            retrievals_titles.append(fig_titles)

        sorted_ids = np.argsort(delta_log_es)

        for i, sorted_id in enumerate(sorted_ids):
            print(f"({retrieval_name}{i:02d}): {', '.join(retrievals_titles[sorted_id])}\t& "
                  f"${delta_log_es[sorted_id]:.2f}$\t& "
                  f"${log_ls[sorted_id]:.2f}$\t& "
                  f"${chi2s[sorted_id]:.3f}$\t& "
                  f"${k_sigmas[sorted_id]:.3f}$ "
                  r"\\")

    return spectral_models, log_es, log_ls, chi2s, sds


def all_log_evidences(directories, build_table=True):
    log_es = []

    for i, directory in enumerate(directories):
        print(f"\nCalculating for directory '{directory}' ({i + 1}/{len(directories)})...")

        log_e = get_log_evidence(
            directory,
        )

        log_es.append(log_e)

    if build_table:
        for i, directory in enumerate(directories):
            print(f"{directory}\t& {log_es[i]:.2f}")

    return log_es


def get_envelope_parameter_sets(sd, sigmas=None):
    if sigmas is None:
        sigmas = (1, 3)

    parameters = {}
    parameters_sets = {}

    for i, sigma in enumerate(sigmas):
        quantile_0 = (1 - erf(sigma / np.sqrt(2))) / 2
        quantile_1 = 1 - quantile_0

        parameters[sigma] = {}
        parameters_sets[sigma] = []
        selection = None

        for key, value in sd.items():
            if key == 'log_likelihood' or key == 'stats':
                continue

            parameters[sigma][key] = [np.quantile(value, quantile_0), np.quantile(value, quantile_1)]

            if selection is None:
                selection = np.ones(np.size(value), dtype=bool)

            selection = np.logical_and(
                selection,
                np.logical_and(
                    np.greater_equal(value, parameters[sigma][key][0]),
                    np.less_equal(value, parameters[sigma][key][1])
                )
            )

        selection = np.nonzero(selection)

        for key, value in sd.items():
            if key == 'log_likelihood' or key == 'stats':
                continue

            id_min = np.argmin(value[selection])
            id_max = np.argmax(value[selection])

            for j in (id_min, id_max):
                parameters_sets[sigma].append({})

                for key2, value2 in sd.items():
                    if key2 == 'log_likelihood' or key2 == 'stats':
                        continue

                    parameters_sets[sigma][-1][key2] = value2[selection][j]

    return parameters_sets


# Utils
def get_best_fit_model(directory, radtrans=None, data=None, data_uncertainties=None,
                       uncertainties_correction_factor=1.171):
    spectral_model = SpectralModel.load(os.path.join(directory, 'simulated_data_model.h5'))

    name = directory.rsplit(os.sep, 1)[1]

    if 'bad3-2' in name:
        name = name.replace('bad3-2', 'bad3_2')

    sd = static_get_sample(directory, name=name, add_log_likelihood=True, add_stats=True)
    log_evidence = sd['stats']['global evidence']

    best_fit_id = np.greater(np.equal(sd['log_likelihood'], np.max(sd['log_likelihood'])), 0)

    parameters_best_fit = {
        parameter: value[best_fit_id][0] for parameter, value in sd.items()
        if parameter != 'log_likelihood' and parameter != 'stats'
    }
    log_l_best_fit = sd['log_likelihood'][best_fit_id][0]

    sm_best_fit = get_model_from_parameters(spectral_model, parameters_best_fit)

    if radtrans is not None:
        wavelengths, spectrum, _ = SpectralModel.retrieval_model_generating_function(
            radtrans,
            sm_best_fit.model_parameters,
            spectrum_model=sm_best_fit,
            mode='transmission',
            update_parameters=True,
            scale=True,
            shift=True,
            convolve=True,
            rebin=True,
            prepare=True
        )
    else:
        sm_best_fit.temperatures, sm_best_fit.mass_fractions, sm_best_fit.mean_molar_masses, \
            sm_best_fit.model_parameters = sm_best_fit.get_spectral_calculation_parameters(
                pressures=np.array([1]),
                wavelengths=np.array([1]),
                line_species=sm_best_fit.line_species,
                **sm_best_fit.model_parameters
            )
        wavelengths = None
        spectrum = None

    if data is not None:
        m_ = data.mask.flatten()
        d_ = data.flatten()
        u_ = data_uncertainties.flatten()

        if uncertainties_correction_factor is None:
            uncertainties_correction_factor = np.ma.mean(
                np.ma.mean(data_uncertainties, axis=2) / np.ma.std(data, axis=2)
            )
            print(f"k_sigma = {uncertainties_correction_factor}")

        if spectrum is not None:
            s_ = spectrum.flatten()
            chi2 = calculate_reduced_chi2(d_[~m_], s_[~m_], u_[~m_])
            chi2_corr = calculate_reduced_chi2(d_[~m_], s_[~m_], u_[~m_] / uncertainties_correction_factor)
            print(f"'{directory.rsplit(os.sep, 1)[1]}' best fit chi2: {chi2:.3f} ({chi2_corr:.3f})")

        chi2_best_fit = log_l_best_fit * -2 / np.size(d_[~m_]) * uncertainties_correction_factor ** 2

        print(f"'{directory.rsplit(os.sep, 1)[1]}' best fit logL: "
              f"{log_l_best_fit:.3f} "
              f"(chi2: {log_l_best_fit * -2 / np.size(d_[~m_]) * uncertainties_correction_factor ** 2:.3f})")
        print(f"'{directory.rsplit(os.sep, 1)[1]}' best fit logE: "
              f"{log_evidence:.3f} "
              f"(chi2: {log_evidence * -2 / np.size(d_[~m_]) * uncertainties_correction_factor ** 2:.3f})")

        return wavelengths, spectrum, sm_best_fit, log_evidence, log_l_best_fit, chi2_best_fit, \
            uncertainties_correction_factor, sd

    return sm_best_fit, log_evidence, log_l_best_fit, sd


def get_model_from_parameters(spectral_model, parameters):
    # TODO classmethod
    new_spectral_model = copy.deepcopy(spectral_model)

    for parameter, value in parameters.items():
        if parameter not in new_spectral_model.model_parameters:
            if parameter in new_spectral_model.model_parameters['imposed_mass_fractions']:
                new_spectral_model.model_parameters['imposed_mass_fractions'][parameter] = 10 ** value
            else:
                if parameter.split('log10_', 1)[1] in new_spectral_model.model_parameters:
                    parameter = parameter.split('log10_', 1)[1]
                    value = 10 ** value
                elif 'log10_' + parameter in new_spectral_model.model_parameters:
                    parameter = 'log10_' + parameter
                    value = np.log10(value)
                else:
                    raise ValueError(f"parameter '{parameter}' not found in model")

        new_spectral_model.model_parameters[parameter] = value

    return new_spectral_model


def get_log_evidence(directory):
    name = directory.rsplit(os.sep, 1)[1]

    sd = static_get_sample(directory, name=name, add_log_likelihood=True, add_stats=True)

    if 'stats' not in sd:
        warnings.warn(f"bad sample dictionary, returning inf log-evidence")
        return np.inf

    else:
        log_evidence = sd['stats']['global evidence']

    return log_evidence


def get_contribution_density(spectral_model: SpectralModel, radtrans, wavelengths, resolving_power=8.04e4,
                             contribution=None):
    sm = copy.deepcopy(spectral_model)

    if contribution is None:
        sm.model_parameters['calculate_contribution'] = True

        wavelengths, _ = spectral_model.calculate_spectrum(
            radtrans=radtrans,
            mode='transmission',
            update_parameters=True
        )

        contribution = copy.deepcopy(radtrans.transmission_contribution)

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


def get_median_model(directory, radtrans, data=None, data_uncertainties=None, uncertainties_correction_factor=1.161):
    spectral_model = SpectralModel.load(os.path.join(directory, 'simulated_data_model.h5'))
    sd = static_get_sample(directory, add_log_likelihood=True)
    sm_median = copy.deepcopy(spectral_model)

    parameters_median = {
        parameter: np.median(value) for parameter, value in sd.items() if parameter != 'log_likelihood'
    }

    for parameter, value in parameters_median.items():
        if 'log10_' in parameter:
            parameter = parameter.split('log10', 1)[1]
            value = 10 ** value

        sm_median.model_parameters[parameter] = value

    sm_median.model_parameters['uncertainties'] = np.ma.masked_array(sm_median.model_parameters['uncertainties'])
    sm_median.model_parameters['uncertainties'].mask = data.mask

    wavelengths, spectrum, _ = SpectralModel.retrieval_model_generating_function(
        radtrans,
        sm_median.model_parameters,
        spectrum_model=sm_median,
        mode='transmission',
        update_parameters=True,
        scale=True,
        shift=True,
        convolve=True,
        rebin=True,
        prepare=True
    )

    if data is not None:
        m_ = data.mask.flatten()
        d_ = data.flatten()
        s_ = spectrum.flatten()
        u_ = data_uncertainties.flatten()

        chi2 = calculate_reduced_chi2(d_[~m_], s_[~m_], u_[~m_])
        chi2_corr = calculate_reduced_chi2(d_[~m_], s_[~m_], u_[~m_] / uncertainties_correction_factor)
        print(f"'{directory.rsplit(os.sep, 1)[1]}' best fit chi2: {chi2:.3f} ({chi2_corr:.3f})")

    return wavelengths, spectrum, sm_median


def get_parameter_range(sd, retrieved_parameters, sm=None):
    parameter_ranges = []
    parameter_names = []
    parameter_titles = []
    parameter_labels = []
    coefficients = []
    offsets = []

    ordered_id = []
    sd_keys = list(sd.keys())
    sd_non_parameter_keys = []

    if 'log_likelihood' in sd:
        sd_non_parameter_keys.append('log_likelihood')

    if 'stats' in sd:
        sd_non_parameter_keys.append('stats')

    for key in sd:
        if key not in retrieved_parameters:
            if key not in sd_non_parameter_keys:
                raise KeyError(f"Key '{key}' not in retrieved parameters")
            else:
                continue

    for i, key in enumerate(retrieved_parameters):
        if key not in sd:
            continue

        ordered_id.append(sd_keys.index(key))
        dictionary = retrieved_parameters[key]

        prior_parameters = np.array(dictionary['prior_parameters'])

        # pRT corner range
        mean = np.mean(sd[key])
        std = np.std(sd[key])
        low_ref = mean - 4 * std
        high_ref = mean + 4 * std

        if 'figure_coefficient' in dictionary:
            if sm is not None:
                if key == 'radial_velocity_semi_amplitude':
                    figure_coefficient = sm.model_parameters['radial_velocity_semi_amplitude']
                elif key == 'planet_radius':
                    figure_coefficient = sm.model_parameters['planet_radius']
                else:
                    figure_coefficient = 1
            else:
                figure_coefficient = 1

            figure_coefficient *= dictionary['figure_coefficient']

            coefficients.append(dictionary['figure_coefficient'])
            low_ref *= dictionary['figure_coefficient']
            high_ref *= dictionary['figure_coefficient']
        else:
            figure_coefficient = 1
            coefficients.append(1)

        if 'figure_offset' in dictionary:
            if sm is not None:
                if key == 'mid_transit_time':
                    figure_offset = sm.model_parameters['mid_transit_time']
                    prior_parameters += sm.model_parameters['mid_transit_time']
                else:
                    figure_offset = 0
            else:
                if key == 'mid_transit_time':
                    prior_parameters -= dictionary['figure_offset']

                figure_offset = 0

            figure_offset += dictionary['figure_offset']

            offsets.append(dictionary['figure_offset'])
            low_ref += dictionary['figure_offset']
            high_ref += dictionary['figure_offset']
        else:
            figure_offset = 0
            offsets.append(0)

        low, high = prior_parameters * figure_coefficient + figure_offset
        low = np.max((low_ref, low))
        high = np.min((high_ref, high))

        parameter_ranges.append([low, high])
        parameter_names.append(key)

        if 'figure_label' in dictionary:
            parameter_labels.append(dictionary['figure_label'])
        else:
            parameter_labels.append(key)

        if 'figure_title' in dictionary:
            parameter_titles.append(dictionary['figure_title'])
        else:
            parameter_titles.append(None)

    sd_tmp = {}

    for i in ordered_id:
        sd_tmp[sd_keys[i]] = copy.deepcopy(sd[sd_keys[i]])

    for k in sd_non_parameter_keys:
        sd_tmp[k] = copy.deepcopy(sd[k])

    sd = copy.deepcopy(sd_tmp)

    return parameter_ranges, parameter_names, parameter_labels, parameter_titles, \
        np.array(coefficients), np.array(offsets), sd


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
    simulated_snr = simulated_snr[:, wh[0], :]
    orbital_phases = orbital_phases[wh[0]]
    airmasses = airmasses[wh[0]]
    barycentric_velocities = barycentric_velocities[wh[0]]
    times = times[wh[0]]

    if np.ndim(telluric_transmittances) == 2:
        telluric_transmittances_wavelengths = telluric_transmittances_wavelengths[wh[0], :]
        telluric_transmittances = telluric_transmittances[wh[0], :]

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
        telluric_transmittances_wavelengths, telluric_transmittances, simulated_snr, orbital_phases, airmasses, \
        barycentric_velocities, times


def static_get_sample(output_dir, name=None, add_log_likelihood=False, add_stats=False):
    if name is None:
        name = output_dir.rsplit(os.sep, 1)[1]

    sample_file = os.path.join(output_dir, 'out_PMN', name + '_post_equal_weights.dat')

    if not os.path.isfile(sample_file):
        warnings.warn(f"file '{sample_file}' not found, skipping...")
        return {}

    samples = np.genfromtxt(os.path.join(output_dir, 'out_PMN', name + '_post_equal_weights.dat'))

    with open(os.path.join(output_dir, 'out_PMN', name + '_params.json'), 'r') as f:
        parameters_read = json.load(f)

    samples_dict = {}

    for i, key in enumerate(parameters_read):
        samples_dict[key] = samples[:, i]

    if add_log_likelihood:
        samples_dict['log_likelihood'] = samples[:, -1]

    if add_stats:
        with open(os.path.join(output_dir, 'out_PMN', name + '_stats.json'), 'r') as f:
            parameters_read = json.load(f)

        samples_dict['stats'] = parameters_read

    return samples_dict


def plot_envelope(x, ys, log_likelihoods=None, figure=None, **kwargs):
    y_min = np.min(ys, axis=0)
    y_max = np.max(ys, axis=0)

    if figure is None:
        figure = plt.figure()

    figure.gca().fill_between(x, y_min, y_max, **kwargs)

    if log_likelihoods is not None:
        kwargs_ = copy.deepcopy(kwargs)
        kwargs_['alpha'] = 1
        kwargs_['label'] = None

        best_fit_index = np.argmax(log_likelihoods)

        figure.gca().plot(x, ys[best_fit_index], **kwargs_)

    return figure


def plot_relative_velocities_envelope(directories_dict, mid_transit_time=0, planet_transit_duration=0,
                                      add_best_fit=False, times=None, sigmas=None, planet=None, observation_day=None,
                                      **kwargs):
    figure = plt.figure()
    y_max = -np.inf
    y_min = np.inf

    if sigmas is not None:
        quantile = scipy.special.erf(sigmas / np.sqrt(2))
        quantile_min = 0.5 - quantile / 2
        quantile_max = 0.5 + quantile / 2
    else:
        quantile_min = 0
        quantile_max = 1

    spectral_model = None
    times_ = np.array([np.inf, -np.inf])

    for label, directory in directories_dict.items():
        sd = static_get_sample(directory, add_log_likelihood=True)
        spectral_model = SpectralModel.load(os.path.join(directory, 'simulated_data_model.h5'))

        if times is not None:
            spectral_model.model_parameters['times'] = times
        else:
            if np.min(spectral_model.model_parameters['times']) < times_[0]:
                times_[0] = np.min(spectral_model.model_parameters['times'])

            if np.max(spectral_model.model_parameters['times']) > times_[1]:
                times_[1] = np.max(spectral_model.model_parameters['times'])

        if sigmas is None:
            ys = np.asarray([spectral_model.get_relative_velocities(
                radial_velocity_semi_amplitude=sd['radial_velocity_semi_amplitude'][i],
                rest_frame_velocity_shift=sd['rest_frame_velocity_shift'][i],
                mid_transit_time=sd['mid_transit_time'][i]) * 1e-5 for i in range(sd['log_likelihood'].size)])

            if add_best_fit:
                log_likelihood = sd['log_likelihood']
            else:
                log_likelihood = None
        else:
            kps = sd['radial_velocity_semi_amplitude'][np.nonzero(np.logical_and(
                np.greater_equal(
                    sd['radial_velocity_semi_amplitude'],
                    np.quantile(sd['radial_velocity_semi_amplitude'], quantile_min)
                ),
                np.less_equal(
                    sd['radial_velocity_semi_amplitude'],
                    np.quantile(sd['radial_velocity_semi_amplitude'], quantile_max)
                ),
            ))]

            kps = np.array([
                np.min(kps),
                np.max(kps)
            ])

            vrs = sd['rest_frame_velocity_shift'][np.nonzero(np.logical_and(
                np.greater_equal(
                    sd['rest_frame_velocity_shift'],
                    np.quantile(sd['rest_frame_velocity_shift'], quantile_min)
                ),
                np.less_equal(
                    sd['rest_frame_velocity_shift'],
                    np.quantile(sd['rest_frame_velocity_shift'], quantile_max)
                ),
            ))]

            vrs = np.array([
                np.min(vrs),
                np.max(vrs)
            ])

            t0s = sd['mid_transit_time'][np.nonzero(np.logical_and(
                np.greater_equal(
                    sd['mid_transit_time'],
                    np.quantile(sd['mid_transit_time'], quantile_min)
                ),
                np.less_equal(
                    sd['mid_transit_time'],
                    np.quantile(sd['mid_transit_time'], quantile_max)
                ),
            ))]

            t0s = np.array([
                np.min(t0s),
                np.max(t0s)
            ])

            ys = []
            log_likelihood = []

            for kp in kps:
                for vr in vrs:
                    for t0_ in t0s:
                        ys.append(spectral_model.get_relative_velocities(
                            radial_velocity_semi_amplitude=kp,
                            rest_frame_velocity_shift=vr,
                            mid_transit_time=t0_) * 1e-5
                        )

                        if add_best_fit:
                            log_likelihood.append(sd['log_likelihood'])

            if not add_best_fit:
                log_likelihood = None

        if np.max(ys) > y_max:
            y_max = np.max(ys)

        if np.min(ys) < y_min:
            y_min = np.min(ys)

        figure = plot_envelope(
            spectral_model.model_parameters['times'] - mid_transit_time,
            ys,
            log_likelihood,
            alpha=1/len(directories_dict), label=label, figure=figure, **kwargs
        )

    if planet is not None:
        star_mass = [
            planet.star_mass + planet.star_mass_error_lower,
            planet.star_mass + planet.star_mass_error_upper
        ]

        orbit_semi_major_axis = [
            planet.orbit_semi_major_axis + planet.orbit_semi_major_axis_error_lower,
            planet.orbit_semi_major_axis + planet.orbit_semi_major_axis_error_upper,
        ]

        kps = np.asarray([
            [planet.calculate_orbital_velocity(star_m, sma) for star_m in star_mass]
            for sma in orbit_semi_major_axis
        ]).flatten()

        vrs = np.zeros(1)

        t0_, t0_el, t0_eu, _ = planet.calculate_mid_transit_time_from_source(
            observation_day,
            planet.transit_midpoint_time / cst.s_cst.day,
            -planet.transit_midpoint_time_error_lower / cst.s_cst.day,
            planet.transit_midpoint_time_error_upper / cst.s_cst.day,
            planet.orbital_period/ cst.s_cst.day, planet.orbital_period_error_lower / cst.s_cst.day,
            planet.orbital_period_error_upper/ cst.s_cst.day,
            True
        )

        t0s = np.array([
            t0_ + t0_el,
            t0_ + t0_eu,
        ])

        ys = []
        log_likelihood = []
        spectral_model.model_parameters['times'] = np.linspace(times_[0], times_[1], 100)
        spectral_model.model_parameters['system_observer_radial_velocities'] = np.mean(
            spectral_model.model_parameters['system_observer_radial_velocities']
        )

        for kp in kps:
            for vr in vrs:
                for t0__ in t0s:
                    ys.append(spectral_model.get_relative_velocities(
                        radial_velocity_semi_amplitude=kp,
                        rest_frame_velocity_shift=vr,
                        mid_transit_time=t0__) * 1e-5
                    )

                    log_likelihood.append(-np.inf)

        ys.append(spectral_model.get_relative_velocities(
            radial_velocity_semi_amplitude=planet.calculate_orbital_velocity(
                planet.star_mass,
                planet.orbit_semi_major_axis
            ),
            rest_frame_velocity_shift=0,
            mid_transit_time=t0_) * 1e-5
        )

        log_likelihood.append(0)

        kwargs_ = copy.deepcopy(kwargs)
        kwargs_['color'] = 'k'
        label = 'expected'

        figure = plot_envelope(
            spectral_model.model_parameters['times'] - mid_transit_time,
            ys,
            log_likelihood,
            alpha=1/len(directories_dict), label=label, figure=figure, **kwargs_
        )

    figure.gca().legend()
    figure.gca().set_xlabel('Time (s)')
    figure.gca().set_ylabel(r'Relative velocity (km$\cdot$s$^{-1}$)')
    figure.gca().set_ylim([y_min, y_max])
    figure.gca().vlines(0, -1e5, 1e5, color='k', ls='--')
    figure.gca().vlines(-planet_transit_duration/2, -1e5, 1e5, color='k', ls='--')
    figure.gca().vlines(planet_transit_duration/2, -1e5, 1e5, color='k', ls='--')


def pipeline_sys(spectrum, **kwargs):
    """Interface with simple_pipeline.

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

    return preparing_pipeline_sysrem(spectrum=spectrum, full=True, **kwargs)


def pipeline_rico(observations, wavelengths_observations, uncertainties, times, v_bary,
                  number_of_modes=1, max_iterations_per_mode=300, **kwargs):
    import petitRADTRANS.retrieval.redexo as redexo

    # Load data
    dataset = redexo.Dataset()

    for i, exp in enumerate(np.moveaxis(copy.deepcopy(observations), 1, 0)):
        dataset.add_exposure(
            spectrum=exp,
            wl=wavelengths_observations,
            errors=uncertainties[:, i],
            obstime=times,
            vbar=v_bary
        )

    pipeline = redexo.Pipeline()

    pipeline.add_module(redexo.FlagAbsorptionEmissionModule(
        flux_lower_limit=0.8,
        flux_upper_limit=1.2,
        relative_to_continuum=True,
        savename='flagged0')
    )
    pipeline.add_module(redexo.PolynomialContinuumRemovalModule(poly_order=3, savename='cont_removed'))
    pipeline.add_module(redexo.OutlierFlaggingModule(sigma=4, savename='flagged1'))

    pipeline.add_module(
        redexo.SysRemModule(
            number_of_modes=number_of_modes,
            mode='subtract',
            max_iterations_per_mode=max_iterations_per_mode,
            savename='after_sysrem'
        )
    )

    pipeline.run(dataset, num_workers=None, per_order=None)

    return pipeline.get_results('after_sysrem'), pipeline


def test_pipeline_rico(spectral_model, model_base, wavelengths_instrument, barycentric_velocities,
                       custom_functions=True):
    import petitRADTRANS.redexo as redexo

    class RedexoSysRemModule(redexo.Module):
        '''
        mode: subtract/divide
        '''

        def initialise(self, number_of_modes, max_iterations_per_mode=1000, mode='subtract'):
            self.number_of_modes = number_of_modes
            self.max_iterations_per_mode = max_iterations_per_mode
            self.per_order = True
            self.mode = mode

        def process(self, dataset, debug=False):
            spec = dataset.spec[:, 0, :]
            errors = dataset.errors[:, 0, :]
            spec = (spec.T - np.nanmean(spec, axis=1)).T
            a = np.ones(dataset.num_exposures)
            nans = np.any(np.isnan(spec), axis=0)
            spec = spec[:, ~nans]
            errors = errors[:, ~nans]

            for i in range(self.number_of_modes):
                correction = np.zeros_like(spec)

                for j in range(self.max_iterations_per_mode):
                    print(i, j)
                    prev_correction = correction
                    c = np.nansum(spec.T * a / errors.T ** 2, axis=1) / np.nansum(a ** 2 / errors.T ** 2, axis=1)
                    a = np.nansum(c * spec / errors ** 2, axis=1) / np.nansum(c ** 2 / errors ** 2, axis=1)
                    correction = np.dot(a[:, np.newaxis], c[np.newaxis, :])

                    # spec /= correction  #
                    # print('break')
                    # full_spec = np.tile(np.nan, dataset.spec[:, 0, :].shape)
                    # full_spec[:, ~nans] = spec
                    # dataset.spec = full_spec[:, np.newaxis]  #
                    # dataset.c = np.tile(np.nan, dataset.spec.shape[-1])  #
                    # dataset.c[~nans] = c  #
                    # dataset.correction = np.tile(np.nan, dataset.spec[:, 0, :].shape)  #
                    # dataset.correction[:, ~nans] = correction
                    # dataset.a = a  #
                    # return dataset  #

                    fractional_dcorr = np.sum(np.abs(correction - prev_correction)) / (
                                np.sum(np.abs(prev_correction)) + 1e-5)

                    if j > 1 and fractional_dcorr < 1e-3:
                        break

                if self.mode == 'subtract':
                    print('sub')
                    spec -= correction
                elif self.mode == 'divide':
                    correction[np.less_equal(np.abs(correction), 1e-15)] = np.nan
                    spec /= correction
                    errors /= correction

            full_spec = np.tile(np.nan, dataset.spec[:, 0, :].shape)
            full_spec[:, ~nans] = spec
            dataset.spec = full_spec[:, np.newaxis]
            dataset.c = np.tile(np.nan, dataset.spec.shape[-1])  #
            dataset.c[~nans] = c  #
            dataset.correction = np.tile(np.nan, dataset.spec[:, 0, :].shape)  #
            dataset.correction[:, ~nans] = correction
            dataset.a = a  #

            return dataset

    class RedexoPipeline(object):
        def __init__(self, in_memory=True):
            self.modules = []
            self.database = {}

        def add_module(self, module):
            self.modules.append(module)

        def run(self, dataset, per_order=True, num_workers=None, debug=False):
            processed_dataset = dataset
            self.run_times = []

            for module in self.modules:
                t = time.time()
                if (per_order or module.per_order) and module.per_order_possible:
                    if num_workers is not None:
                        raise NotImplementedError
                    else:
                        results = [module.process(processed_dataset.get_order(order), debug=debug) for order in
                                   range(processed_dataset.num_orders)]

                    # Check if resulting dataset still matches our previous dataset
                    # If not, make sure they get compatible
                    if isinstance(results[0], redexo.CCF_Dataset) and not isinstance(processed_dataset,
                                                                                     redexo.CCF_Dataset):
                        empty_spec = np.zeros(
                            (results[0].spec.shape[0], processed_dataset.num_orders, results[0].spec.shape[-1]))
                        processed_dataset = redexo.CCF_Dataset(empty_spec, rv_grid=empty_spec.copy(), \
                                                        vbar=processed_dataset.vbar,
                                                        obstimes=processed_dataset.obstimes, **dataset.header_info)
                    elif not results[0].spec.shape == processed_dataset.spec.shape:
                        processed_dataset.make_clean(
                            shape=(results[0].spec.shape[0], processed_dataset.num_orders, results[0].spec.shape[-1]))

                    # Write results to dataset
                    a_s = np.zeros((results[0].spec.shape[0], processed_dataset.num_orders))
                    c_s = np.zeros((results[0].spec.shape[-1], processed_dataset.num_orders))
                    corrs = np.zeros((results[0].spec.shape[0], processed_dataset.num_orders, results[0].spec.shape[-1]))
                    for order, res in enumerate(results):
                        processed_dataset.set_results(res, order=order)

                        if hasattr(res, 'a'):
                            print('a, order', order, res.a.shape, res.c.shape)
                            a_s[:, order] = res.a
                            c_s[:, order] = res.c
                            corrs[:, order] = res.correction
                            processed_dataset.a = a_s
                            processed_dataset.c = c_s
                            processed_dataset.correction = corrs
                        else:
                            print('no a')

                else:
                    processed_dataset = module.process(processed_dataset, debug=debug)
                if not module.savename is None:
                    self.database[module.savename] = processed_dataset.copy()
                self.run_times.append(time.time() - t)

        def get_results(self, name):
            return self.database[name]

        def write_results(self, filepath):
            self.hdulist = fits.HDUList()
            self.hdulist.append(fits.PrimaryHDU())
            for savename, dataset in self.database.items():
                self.hdulist.append(fits.ImageHDU(dataset.spec, name=savename + "_spec"))
                self.hdulist.append(fits.ImageHDU(dataset.wavelengths, name=savename + "_wavelengths"))
                self.hdulist.append(fits.ImageHDU(dataset.errors, name=savename + "_errors"))
            self.hdulist.writeto(filepath)

        def summary(self):
            print('----------Summary--------')
            for i, module in enumerate(self.modules):
                print('Running {0} took {1:.2f} seconds'.format(module, self.run_times[i]))
            print('--> Total time: {0:.2f} seconds'.format(np.sum(np.array(self.run_times))))
            print('------------------------')

    if not custom_functions:
        RedexoPipeline = redexo.Pipeline
        RedexoSysRemModule = redexo.SysRemModule
    else:
        print("using custom functions")

    def pipeline_rico_test(observations, wavelengths_observations, uncertainties, times, v_bary,
                           number_of_modes=1, max_iterations_per_mode=1, **kwargs):
        import petitRADTRANS.retrieval.redexo as redexo

        # Load data
        dataset = redexo.Dataset()

        for i, exp in enumerate(np.moveaxis(copy.deepcopy(observations), 1, 0)):
            dataset.add_exposure(
                spectrum=exp,
                wl=wavelengths_observations,
                errors=uncertainties[:, i],
                obstime=times,
                vbar=v_bary
            )

        pipeline = RedexoPipeline()

        # pipeline.add_module(redexo.FlagAbsorptionEmissionModule(
        #     flux_lower_limit=0.8,
        #     flux_upper_limit=1.2,
        #     relative_to_continuum=True,
        #     savename='flagged0')
        # )
        pipeline.add_module(redexo.PolynomialContinuumRemovalModule(poly_order=3, savename='cont_removed'))
        pipeline.add_module(redexo.OutlierFlaggingModule(sigma=4, savename='flagged1'))

        pipeline.add_module(
            RedexoSysRemModule(
                number_of_modes=number_of_modes,
                mode='divide',
                max_iterations_per_mode=max_iterations_per_mode,
                savename='after_sysrem'
            )
        )

        pipeline.run(dataset, num_workers=None, per_order=None)

        return pipeline.get_results('after_sysrem'), pipeline

    spectral_model.pipeline = pipeline_rico_test
    spectral_model.model_parameters['number_of_modes'] = 1
    spectral_model.model_parameters['max_iterations_per_mode'] = 15

    rdsRIC, pipRIC = spectral_model.pipeline(
        model_base,
        wavelengths_observations=wavelengths_instrument,
        v_bary=barycentric_velocities,
        **spectral_model.model_parameters
    )

    return rdsRIC, pipRIC


def test_pipeline_sys(spectral_model, model_base, wavelengths_instrument, custom_functions=True):
    def __sysrem_iterationt(spectrum_uncertainties_squared, uncertainties_squared_inverted, a, shape_a, shape_c):
        """SYSREM iteration.
        For the first iteration, c should be 1.
        The inputs are chosen in order to maximize speed.

        Args:
            spectrum_uncertainties_squared: spectral data to correct over the uncertainties ** 2 (..., exposure, wavelength)
            uncertainties_squared_inverted: invers of the squared uncertainties on the data (..., exposure, wavelength)
            c: 2-D matrix (..., exposures, wavelengths) containing the a-priori "extinction coefficients"
            shape_a: intermediate shape for "airmass" estimation (wavelength, ..., exposure)
            shape_c: intermediate shape for "extinction coefficients" estimation (exposure, ..., wavelength)

        Returns:
            The lower-rank estimation of the spectrum (systematics), and the estimated "extinction coefficients"
        """
        # Recalculate the best fitting "extinction coefficients", not related to the true extinction coefficients
        c = np.sum(a * spectrum_uncertainties_squared, axis=-2) / \
            np.sum(a ** 2 * uncertainties_squared_inverted, axis=-2)

        # Tile into a (..., exposure, wavelength) matrix
        c = np.moveaxis(
            c * np.ones(shape_c),
            0,
            -2
        )

        # Get the "airmass" (time variation of a pixel), not related to the true airmass
        a = np.sum(c * spectrum_uncertainties_squared, axis=-1) / \
            np.sum(c ** 2 * uncertainties_squared_inverted, axis=-1)

        # Tile into a (..., exposure, wavelength) matrix
        a = np.moveaxis(
            a * np.ones(shape_a),
            0,
            -1
        )

        return a * c, a, c

    def preparing_pipeline_sysremt(spectrum, uncertainties, n_iterations_max=10, convergence_criterion=1e-3,
                                   tellurics_mask_threshold=0.8,
                                   apply_throughput_removal=True, apply_telluric_lines_removal=True, subtract=False,
                                   full=False, verbose=False, **kwargs):
        """SYSREM preparing pipeline.
        SYSREM tries to find the coefficients a and c such as:
            S**2 = sum_ij ((spectrum_ij - a_j * c_i) / uncertainties)**2
        is minimized. Several iterations can be performed. This assumes that the spectrum is deformed by a combination of
        linear effects.
        The coefficients a and c can be seen as estimates for any strong (linear) systematic effect in the data, they are
        not necessarily related to the airmass and extinction coefficients.

        Source: Tamuz et al. 2005 (doi:10.1111/j.1365-2966.2004.08585.x).
        Thanks to Alejandro Sanchez-Lopez (26-09-2017) for sharing his version of the algorithm.

        Args:
            spectrum: spectral data to correct
            uncertainties: uncertainties on the data
            n_iterations_max: maximum number of SYSREM iterations
            convergence_criterion: SYSREM convergence criterion
            tellurics_mask_threshold: mask wavelengths where the atmospheric transmittance estimate is below this value
            apply_throughput_removal: if True, divide the spectrum by its mean over wavelengths
            apply_telluric_lines_removal: if True, apply the telluric lines removal correction
            subtract: if True, subtract the fitted systematics to the spectrum instead of dividing them
            full: if True, return the reduced matrix and reduced uncertainties in addition to the reduced spectrum
            verbose: if True, print the convergence status at each iteration

        Returns:
            Reduced spectral data (and reduction matrix and uncertainties after reduction if full is True)
        """
        # Pre-preparation
        print('start a')
        reduction_matrix = np.ones(spectrum.shape)
        reduced_data_uncertainties = copy.deepcopy(uncertainties)
        reduced_data = copy.deepcopy(spectrum)

        if apply_throughput_removal:
            throughput_fit = 1 / np.ma.mean(reduced_data, axis=-1)
            reduced_data = np.moveaxis(np.moveaxis(reduced_data, -1, 0) * throughput_fit, 0, -1)
            reduction_matrix = np.moveaxis(np.moveaxis(reduction_matrix, -1, 0) * throughput_fit, 0, -1)
            reduced_data_uncertainties = np.moveaxis(
                np.moveaxis(reduced_data_uncertainties, -1, 0) * np.abs(throughput_fit),
                0,
                -1
            )

        reduced_data = np.ma.masked_where(reduced_data < tellurics_mask_threshold, reduced_data)
        # print('rm 1')
        # reduced_data -= 1

        if not apply_telluric_lines_removal:
            if full:
                return reduced_data, reduction_matrix, reduced_data_uncertainties
            else:
                return reduced_data

        # Initialize SYSREM meaningful variables
        spectrum_shape = list(reduced_data.shape)
        shape_a = copy.copy(spectrum_shape)
        shape_c = copy.copy(spectrum_shape)

        shape_a.insert(0, shape_a.pop(-1))
        shape_c.insert(0, shape_c.pop(-2))

        uncertainties_squared_inverted = 1 / reduced_data_uncertainties ** 2
        spectrum_uncertainties_squared = reduced_data * uncertainties_squared_inverted

        # Handle masked values
        if isinstance(spectrum_uncertainties_squared, np.ma.core.MaskedArray):
            uncertainties_squared_inverted[spectrum_uncertainties_squared.mask] = 0
            spectrum_uncertainties_squared = spectrum_uncertainties_squared.filled(0)

        # Iterate
        i = 0
        a = 1
        systematics_0 = np.zeros(reduced_data.shape)
        systematics = np.zeros(reduced_data.shape)

        for i in range(n_iterations_max):
            systematics, a, c = __sysrem_iterationt(
                spectrum_uncertainties_squared=spectrum_uncertainties_squared,
                uncertainties_squared_inverted=uncertainties_squared_inverted,
                a=a,
                shape_a=shape_a,
                shape_c=shape_c
            )
            systematics[np.nonzero(np.logical_not(np.isfinite(systematics)))] = 0

            # return reduced_data / systematics, None, None, a, c

            # Check for convergence
            if np.sum(np.abs(systematics_0 - systematics)) <= convergence_criterion * np.sum(np.abs(systematics_0)):
                if verbose:
                    print(f"Iteration {i} (max {n_iterations_max}): "
                          f"{np.sum(np.abs(systematics_0 - systematics)) / np.sum(np.abs(systematics_0))} "
                          f"(> {convergence_criterion})")
                    print("Convergence reached!")

                break
            elif verbose and i > 0:
                print(f"Iteration {i} (max {n_iterations_max}): "
                      f"{np.sum(np.abs(systematics_0 - systematics)) / np.sum(np.abs(systematics_0))} "
                      f"(> {convergence_criterion})")

            systematics_0 = systematics

        if i == n_iterations_max - 1 \
                and np.sum(np.abs(systematics_0 - systematics)) > convergence_criterion * np.sum(np.abs(systematics_0)) \
                and convergence_criterion > 0:
            warnings.warn(
                f"convergence not reached in {n_iterations_max} iterations "
                f"({np.sum(np.abs(systematics_0 - systematics)) > convergence_criterion * np.sum(np.abs(systematics_0))} "
                f"> {convergence_criterion})"
            )

        # Mask where systematics are 0 to prevent division by 0 error
        systematics = np.ma.masked_equal(systematics, 0)

        # Remove the systematics from the spectrum
        '''
        This can also be done by subtracting the systematics from the spectrum, but dividing give almost the same results
        and this way the pipeline can be used in retrievals more effectively.
        '''
        if subtract:
            reduced_data -= systematics
        else:
            reduced_data = reduced_data / systematics

        if full:
            if subtract:
                # With the subtractions, uncertainties should not be affected
                # TODO it can be argued that the uncertainties on the systematics should be taken into account
                reduction_matrix -= systematics
            else:
                reduction_matrix /= systematics
                reduced_data_uncertainties = reduced_data_uncertainties * np.abs(reduction_matrix)

            return reduced_data, reduction_matrix, reduced_data_uncertainties, a, c
        else:
            return reduced_data

    if not custom_functions:
        preparing_pipeline_sysremt = preparing_pipeline_sysrem
    else:
        print("Using custom functions")

    def pipeline_syst(spectrum, **kwargs):
        """Interface with simple_pipeline.

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

        return preparing_pipeline_sysremt(spectrum=spectrum, full=True, **kwargs)

    spectral_model.pipeline = pipeline_syst
    spectral_model.model_parameters['n_iterations_max'] = 15
    spectral_model.model_parameters['convergence_criterion'] = -1
    spectral_model.model_parameters['remove_mean'] = True
    spectral_model.model_parameters['verbose'] = True

    rds, rms, rus, a, c = spectral_model.pipeline(
        model_base,
        wavelengths=wavelengths_instrument,
        **spectral_model.model_parameters
    )

    return rds, rms, rus, a, c


def quick_ccf(rebined_model, wavelengths_instrument, prepared_data, sm):
    from petitRADTRANS.ccf.ccf import ccf_analysis

    model_base = rebined_model
    wvlx = SpectralModel.resolving_space(wavelengths_instrument.min(), wavelengths_instrument.max(), 8.04e4)

    print('Preparing model...')
    rdb, rmb, rub = sm.pipeline(model_base, wavelengths=wavelengths_instrument, **sm.model_parameters)

    rdbx = np.moveaxis(rdb, 0, 1).reshape((model_base.shape[-2], prepared_data.shape[-3] * prepared_data.shape[-1]))
    rdbxi = np.array([np.interp(wvlx, wavelengths_instrument.flatten(), rdbxx) for rdbxx in rdbx])
    rdbxi[np.nonzero(np.less_equal(rdbxi, 0.998))] = 1

    print('CCF...')
    co_added_cross_correlations_snr, co_added_cross_correlations, \
        v_rest, kps, ccf_sum, ccfs, velocities_ccf, ccf_models, ccf_model_wavelengths = ccf_analysis(
            wavelengths_instrument[1:-1], prepared_data[1:-1], wvlx, rdbxi,
            radial_velocity_semi_amplitude=sm.model_parameters['radial_velocity_semi_amplitude'],
            model_velocities=sm.model_parameters['relative_velocities'],
            system_observer_radial_velocities=sm.model_parameters['system_observer_radial_velocities'],
            orbital_longitudes=sm.model_parameters['orbital_longitudes'],
            orbital_inclination=sm.model_parameters['planet_orbital_inclination'],
            line_spread_function_fwhm=2.6e5, velocity_interval_extension_factor=-0.0, ccf_sum_axes=[0])

    return co_added_cross_correlations_snr, co_added_cross_correlations, \
        v_rest, kps, ccf_sum, ccfs, velocities_ccf, ccf_models, ccf_model_wavelengths


def quick_ccf2(shifted_model, shifted_model_wavelengths, wavelengths_instrument, prepared_data, sm):
    from petitRADTRANS.ccf.ccf import ccf_analysis

    model_base = shifted_model
    wvlx = SpectralModel.resolving_space(wavelengths_instrument.min(), wavelengths_instrument.max(), 8.04e4)

    print('Preparing model...')
    rdbxi = np.array([np.interp(wvlx, shifted_model_wavelengths[i], rdbxx) for i, rdbxx in enumerate(model_base)])
    unc = copy.deepcopy(sm.model_parameters['uncertainties'])
    unc = np.moveaxis(unc, 0, 1).reshape((unc.shape[-2], unc.shape[-3] * unc.shape[-1]))
    unci = np.array([np.interp(wvlx, wavelengths_instrument.flatten(), uncx) for uncx in unc])
    unci = np.ma.masked_invalid(unci)
    unci = np.ma.masked_less_equal(unci, 1e-15)

    rdbxi = np.array([rdbxi])
    unci = np.ma.array([unci])
    wvlx = np.tile(wvlx, (rdbxi.shape[0], 1))

    rdbxi, rmb, rub = sm.pipeline(
        rdbxi,
        uncertainties=unci,
        wavelengths=wvlx,
        airmass=sm.model_parameters['airmass'],
        tellurics_mask_threshold=sm.model_parameters['tellurics_mask_threshold'],
        polynomial_fit_degree=sm.model_parameters['polynomial_fit_degree'],
        apply_throughput_removal=sm.model_parameters['apply_throughput_removal'],
        apply_telluric_lines_removal=sm.model_parameters['apply_telluric_lines_removal']
    )

    print('CCF...')
    co_added_cross_correlations_snr, co_added_cross_correlations, \
        v_rest, kps, ccf_sum, ccfs, velocities_ccf, ccf_models, ccf_model_wavelengths = ccf_analysis(
            wavelengths_instrument[1:-1], prepared_data[1:-1], wvlx, rdbxi,
            radial_velocity_semi_amplitude=sm.model_parameters['radial_velocity_semi_amplitude'],
            model_velocities=sm.model_parameters['relative_velocities'],
            system_observer_radial_velocities=sm.model_parameters['system_observer_radial_velocities'],
            orbital_longitudes=sm.model_parameters['orbital_longitudes'],
            orbital_inclination=sm.model_parameters['planet_orbital_inclination'],
            line_spread_function_fwhm=2.6e5, velocity_interval_extension_factor=-0.0, ccf_sum_axes=[0])

    return co_added_cross_correlations_snr, co_added_cross_correlations, \
        v_rest, kps, ccf_sum, ccfs, velocities_ccf, ccf_models, ccf_model_wavelengths


def init_retrieved_parameters(retrieval_parameters, mid_transit_time_jd, mid_transit_time_range,
                              external_parameters_ref=None):
    retrieved_parameters_ref = {
        'temperature': {
            'prior_parameters': [100.0, 4000.0],
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
            'figure_offset': 5.0,
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
        'radial_velocity_semi_amplitude': {
            'prior_parameters': np.array([0.4589, 1.6388]) * 15254766.005394705,  # Kp must be close to the true value to help the retrieval
            'prior_type': 'uniform',
            'figure_title': r'$K_p$',
            'figure_label': r'$K_p$ (km$\cdot$s$^{-1}$)',
            'figure_coefficient': 1e-5,
            'retrieval_name': 'Kp'
        },
        'rest_frame_velocity_shift': {
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
            'prior_parameters': np.array([-mid_transit_time_range, mid_transit_time_range], dtype=float),
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

    # Initialisation
    retrieved_parameters = {}

    for parameter in retrieval_parameters:
        if parameter not in retrieved_parameters_ref:
            raise KeyError(f"parameter '{parameter}' was not initialized")
        else:
            retrieved_parameters[parameter] = retrieved_parameters_ref[parameter]

    if external_parameters_ref is not None:
        for parameter in external_parameters_ref:
            if parameter in retrieved_parameters:
                for prop, value in external_parameters_ref[parameter][()].items():
                    retrieved_parameters[parameter][prop] = value

    return retrieved_parameters_ref, retrieved_parameters


def load_variable_throughput_brogi(file, times_size, wavelengths_size):
    variable_throughput = np.load(file)

    variable_throughput = np.max(variable_throughput[0], axis=1)
    variable_throughput = variable_throughput / np.max(variable_throughput)

    xp = np.linspace(0, 1, np.size(variable_throughput))
    x = np.linspace(0, 1, times_size)
    variable_throughput = np.interp(x, xp, variable_throughput)

    return np.tile(variable_throughput, (wavelengths_size, 1)).T


def save_to_alex(file, wavelengths, data, prepared_data, mask, times, barycentric_velocities, airmass, order_selection):
    np.savez_compressed(
        file=file,
        wavelengths=wavelengths,
        data=data,
        prepared_data=prepared_data,
        mask=mask,
        times=times,
        barycentric_velocities=barycentric_velocities,
        airmass=airmass,
        order_selection=order_selection
    )


def quick_figure_setup(external_parameters_ref=None):
    # Manual initialisation
    use_t23 = False
    use_t1535 = False
    planet_name = 'HD 189733 b'
    output_directory = os.path.abspath(os.path.abspath(os.path.dirname('./scripts/scripts.py'))
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

    detector_selection_name = 'testd'

    retrieval_parameters = [
        'new_resolving_power',
        'radial_velocity_semi_amplitude',
        'rest_frame_velocity_shift',
        'mid_transit_time',
        'planet_radius',
        'log10_planet_surface_gravity',
        'temperature',
        'CH4_hargreaves_main_iso',
        'CO_all_iso',
        'H2O_main_iso',
        'H2S_main_iso',
        'HCN_main_iso',
        'NH3_main_iso',
        'log10_cloud_pressure',
        'log10_scattering_opacity_350nm',
        'scattering_opacity_coefficient'
    ]

    mid_transit_time_jd = 58004.424877  # 58004.42319302507  # 58004.425291
    mid_transit_time_range = 300

    retrieved_parameters_ref, retrieved_parameters = init_retrieved_parameters(
        retrieval_parameters,
        mid_transit_time_jd,
        mid_transit_time_range,
        external_parameters_ref=external_parameters_ref
    )

    figure_directory = r'C:\Users\Doriann\Documents\work\run_outputs\petitRADTRANS\figures\HD_189733_b_CARMENES'
    image_format = 'pdf'

    retrieval_directory = r'C:\Users\Doriann\Documents\work\run_outputs\petitRADTRANS\retrievals\carmenes_retrievals'
    additional_data_directory = 'C:\\Users\\Doriann\\Documents\\work\\run_outputs\\petitRADTRANS\\data'
    sm = SpectralModel.load(
        retrieval_directory +
        r'\HD_189733_b_transmission_'
        r'Kp_V0_tiso_H2O_Pc_k0_gams_tmt0.80_t0r300_alex4_nuaw_c819_100lp\simulated_data_model.h5'
    )


def plot_all_figures(retrieved_parameters,
                     figure_directory=r'C:\Users\Doriann\Documents\work\run_outputs\petitRADTRANS\figures\
                                        HD_189733_b_CARMENES',
                     image_format='pdf'):
    # Init
    retrieval_directory = r'C:\Users\Doriann\Documents\work\run_outputs\petitRADTRANS\retrievals\carmenes_retrievals'
    sm = SpectralModel.load(
        retrieval_directory +
        r'\HD_189733_b_transmission_'
        r'R_T0_Kp_V0_g_tiso_CH4_CO_H2O_H2S_HCN_NH3_Pc_k0_gams_tmt0.80_t0r300_alex4_sim_ccn12345_nuaw_c819_100lp\simulated_data_model.h5'
    )
    radtrans = sm.get_radtrans()

    sm.model_parameters['imposed_mass_fractions'] = {
        'CH4_hargreaves_main_iso': 3.4e-5,
        'CO_all_iso': 1.8e-2,
        'H2O_main_iso': 5.4e-3,
        'H2S_main_iso': 1.0e-3,
        'HCN_main_iso': 2.7e-7,
        'NH3_main_iso': 7.9e-6
    }

    sm.model_parameters['cloud_pressure'] = 1
    sm.model_parameters['scattering_opacity_350nm'] = 1e-3
    sm.model_parameters['scattering_opacity_coefficient'] = -4

    sd, true_values = plot_init(
        retrieved_parameters,
        retrieval_directory +
        r'\HD_189733_b_transmission_'
        r'R_T0_Kp_V0_g_tiso_CH4_CO_H2O_H2S_HCN_NH3_Pc_k0_gams_tmt0.80_t0r300_alex4_sim_ccn12345_nuaw_c819_100lp',
        sm
    )

    wavelengths_instrument, observed_spectra, instrument_snr, uncertainties, orbital_phases, airmasses, \
        barycentric_velocities, times, mid_transit_time = load_carmenes_data(
            directory=os.path.join(
                r'C:\Users\Doriann\Documents\work\run_outputs\petitRADTRANS\data',
                'carmenes',
                'hd_189733_b'
            ),
            mid_transit_jd=58004.424877#58004.425291
        )

    instrumental_deformations, telluric_transmittances_wavelengths, telluric_transmittances, simulated_snr, _ = \
        load_additional_data(
            data_dir=r'C:\Users\Doriann\Documents\work\run_outputs\petitRADTRANS\data',
            wavelengths_instrument=wavelengths_instrument,
            airmasses=airmasses,
            resolving_power=sm.model_parameters['new_resolving_power']
        )

    detector_selection = sm.model_parameters['detector_selection']
    # detector_selection = np.arange(0, 56)  # all
    ccd_id = np.asarray(detector_selection == 46).nonzero()[0][0]
    planet = Planet.get('HD 189733 b')

    # Modify TD
    planet_transit_duration = planet.transit_duration + np.max(
        retrieved_parameters['mid_transit_time']['prior_parameters']) \
        - np.min(retrieved_parameters['mid_transit_time']['prior_parameters'])

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
            use_t23=False,
            use_t1535=False
        )

    simulated_uncertainties = np.moveaxis(
        np.moveaxis(simulated_snr, 2, 0) / np.mean(simulated_snr, axis=2) * np.mean(uncertainties, axis=2),
        0,
        2
    )
    simulated_uncertainties = np.ma.masked_where(uncertainties.mask, simulated_uncertainties)

    if 'noise_matrix' in sm.model_parameters:
        noise_matrix = sm.model_parameters['noise_matrix']
    else:
        noise_matrix = np.random.default_rng(seed=12345).normal(loc=0, scale=uncertainties, size=uncertainties.shape)

    sm.model_parameters['uncertainties'] = np.ma.masked_where(
        copy.deepcopy(uncertainties.mask),
        sm.model_parameters['uncertainties']
    )

    # print(f'Reprocessing data...')
    # reprocessed_data, reprocessing_matrix, reprocessed_data_uncertainties = sm.pipeline(
    #     observed_spectra,
    #     wavelengths=wavelengths_instrument,
    #     **sm.model_parameters
    # )

    xlim = (1.519 * 1e-6, 1.522 * 1e-6)

    print('Starting making figures...')
    # Figure 1 from Alex

    # Model steps
    plot_model_steps(
        spectral_model=sm,
        radtrans=radtrans,
        mode='transmission',
        ccd_id=ccd_id,
        xlim=xlim,
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
        xlim=xlim,
        path_outputs=figure_directory,
        image_format=image_format,
        noise_factor=10
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
        orbital_phase_id=8,
        xlim=xlim,
        path_outputs=figure_directory,
        image_format=image_format
    )

    # Reprocessing effect
    retrieval_data = np.load(
        r'C:\Users\Doriann\Documents\work\run_outputs\petitRADTRANS\retrievals\carmenes_retrievals'
        r'\HD_189733_b_transmission_'
        r'R_T0_Kp_V0_tiso_CH4_CO_H2O_H2S_HCN_NH3_Pc_k0_gams_tmt0.80_t0r300_alex4_nuaw_c819_100lp'
        r'\retrieved_parameters.npz'
    )
    reprocessed_data = np.ma.masked_where(retrieval_data['data_mask'], retrieval_data['data'])

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
        xlim=xlim,
        path_outputs=figure_directory,
        image_format=image_format
    )

    # Reprocessing effect SysRem
    retrieval_data = np.load(
        r'C:\Users\Doriann\Documents\work\run_outputs\petitRADTRANS\retrievals\carmenes_retrievals'
        r'\HD_189733_b_transmission_'
        r'R_T0_Kp_V0_g_tiso_CH4_CO_H2O_H2S_HCN_NH3_Pc_k0_gams_tmt0.80_t0r300_alex4_sys-p1-i10-sub_nuaw_c819_100lp'
        r'\retrieved_parameters.npz'
    )
    reprocessed_data = np.ma.masked_where(retrieval_data['data_mask'], retrieval_data['data'])

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
        xlim=xlim,
        path_outputs=figure_directory,
        image_format=image_format,
        figure_name='preparing_effect_sysrem',
        use_sysrem=True,
        add_prepared_model=False,
        side_by_side=True,
        n_passes=[1, 2, 3, 5, 10],
        n_iterations_max=10
    )

    # Reprocessing effect SysRem p2
    retrieval_data = np.load(
        r'C:\Users\Doriann\Documents\work\run_outputs\petitRADTRANS\retrievals\carmenes_retrievals'
        r'\HD_189733_b_transmission_'
        r'R_T0_Kp_V0_g_tiso_CH4_CO_H2O_H2S_HCN_NH3_Pc_k0_gams_tmt0.80_t0r300_alex4_sys-p1-i10-sub_nuaw_c819_100lp'
        r'\retrieved_parameters.npz'
    )
    reprocessed_data = np.ma.masked_where(retrieval_data['data_mask'], retrieval_data['data'])

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
        xlim=xlim,
        path_outputs=figure_directory,
        image_format=image_format,
        figure_name='preparing_effect_sysrem_p2',
        use_sysrem=True,
        n_passes=2,
        n_iterations_max=10
    )

    # Validity
    validity_, noiseless_validity_, reprocessed_deformed_spectrum_validity, reprocessed_noisy_spectrum_validity = (
        plot_validity(sm, radtrans, figure_directory, image_format, noise_matrix, False))

    # Expected retrieval corner plotlib
    parameter_names_ref = [
        'temperature',
        'CH4_hargreaves_main_iso',
        'CO_all_iso',
        'H2O_main_iso',
        'H2S_main_iso',
        'HCN_main_iso',
        'NH3_main_iso',
        'log10_cloud_pressure',
        'log10_scattering_opacity_350nm',
        'scattering_opacity_coefficient',
        'log10_planet_surface_gravity',
        'radial_velocity_semi_amplitude',
        'rest_frame_velocity_shift',
        'new_resolving_power',
        'mid_transit_time'
    ]

    retrieved_parameters_tmp = {}

    for k in parameter_names_ref:
        retrieved_parameters_tmp[k] = copy.deepcopy(retrieved_parameters[k])

    retrieved_parameters = copy.deepcopy(retrieved_parameters_tmp)
    del retrieved_parameters_tmp

    plot_result_corner(
        result_directory=[retrieval_directory +
                          r'\HD_189733_b_transmission_'
                          r'R_T0_Kp_V0_g_tiso_CH4_CO_H2O_H2S_HCN_NH3_Pc_k0_gams_tmt0.80_t0r300_alex4_sim_cc_nuaw_c819_100lp',
                          retrieval_directory +
                          r'\HD_189733_b_transmission_'
                          r'R_T0_Kp_V0_g_tiso_CH4_CO_H2O_H2S_HCN_NH3_Pc_k0_gams_tmt0.80_t0r300_alex4_sim3d_f2_c_nuaw_c819_100lp',
                          # retrieval_directory +
                          # r'\HD_189733_b_transmission_'
                          # r'R_T0_Kp_V0_g_tiso_CH4_CO_H2O_H2S_HCN_NH3_Pc_k0_gams_tmt0.80_t0r300_alex4_sim_cc_sys-p1-i10-sub_nuaw_c819_100lp',
                          ],
        sm=None,
        retrieved_parameters=retrieved_parameters,
        figure_directory=figure_directory,
        figure_name='corner_simulated_retrievals',
        label_kwargs={'fontsize': 10},
        title_kwargs={'fontsize': 8},
        true_values=true_values,
        figure_font_size=8,
        save=True
    )

    plot_result_corner(
        result_directory=[retrieval_directory +
                          r'\HD_189733_b_transmission_'
                          r'R_T0_Kp_V0_g_tiso_CH4_CO_H2O_H2S_HCN_NH3_Pc_k0_gams_tmt0.80_t0r300_alex4_sim_ccn12345_nuaw_c819_100lp',
                          # retrieval_directory +
                          # r'\HD_189733_b_transmission_'
                          # r'R_T0_Kp_V0_g_tiso_CH4_CO_H2O_H2S_HCN_NH3_Pc_k0_gams_tmt0.80_t0r300_alex4_sim_ccn12345_sys-p1-i10-sub_nuaw_c819_100lp',
                          ],
        sm=None,
        retrieved_parameters=retrieved_parameters,
        figure_directory=figure_directory,
        figure_name='corner_noisy_simulated_retrievals',
        label_kwargs={'fontsize': 10},
        title_kwargs={'fontsize': 8},
        true_values=true_values,
        figure_font_size=8,
        color_list=['C0', 'C2', 'C1'],
        save=True
    )

    # Results
    plot_result_corner(
        result_directory=[retrieval_directory +
                          r'\HD_189733_b_transmission_'
                          r'Kp_V0_tiso_H2O_Pc_tmt0.80_t0r300_alex4_nuaw_c819_100lp',
                          # retrieval_directory +
                          # r'\HD_189733_b_transmission_'
                          # r'Kp_V0_tiso_H2O_Pc_tmt0.80_t0r300_alex4_sys-p1-i10-sub_nuaw_c819_100lp',
                          ],
        sm=None,
        retrieved_parameters=retrieved_parameters,
        figure_directory=figure_directory,
        figure_name='corner_simple_retrieval',
        label_kwargs={'fontsize': 16},
        title_kwargs={'fontsize': 13},
        true_values=None,
        figure_font_size=12,
        color_list=['C0', 'C2', 'C1'],
        save=True
    )

    # plot_result_corner(
    #     result_directory=[retrieval_directory +
    #                       r'\HD_189733_b_transmission_'
    #                       r'R_T0_Kp_V0_tiso_H2O_Pc_tmt0.80_t0r300_alex4_sys-p1-i10-sub_nuaw_c819_100lp',
    #                       retrieval_directory +
    #                       r'\HD_189733_b_transmission_'
    #                       r'R_T0_Kp_V0_tiso_H2O_Pc_tmt0.80_t0r300_alex4_sys-p2-i10-sub_nuaw_c819_100lp',
    #                       ],
    #     sm=None,
    #     retrieved_parameters=retrieved_parameters,
    #     figure_directory=figure_directory,
    #     figure_name='corner_P18_retrieval_sysrem',
    #     label_kwargs={'fontsize': 16},
    #     title_kwargs={'fontsize': 13},
    #     true_values=None,
    #     figure_font_size=12,
    #     color_list=['C2', 'C3', 'C1'],
    #     save=True
    # )

    plot_result_corner(
        result_directory=[
            retrieval_directory +
            r'\HD_189733_b_transmission_'
            r'Kp_V0_tiso_H2O_Pc_tmt0.80_t0r300_all_nuaw_c819_100lp',
            # retrieval_directory +
            # r'\HD_189733_b_transmission_'
            # r'Kp_V0_tiso_H2O_Pc_tmt0.80_t0r300_all_sys-p1-i10-sub_nuaw_c819_100lp',
        ],
        sm=None,
        retrieved_parameters=retrieved_parameters,
        figure_directory=figure_directory,
        figure_name='corner_simple_retrieval_all_orders',
        label_kwargs={'fontsize': 16},
        title_kwargs={'fontsize': 13},
        true_values=None,
        figure_font_size=12,
        color_list=['C0', 'C2', 'C1'],
        save=True
    )

    external_parameters_ref2 = np.load(
        r'C:\Users\Doriann\Documents\work\run_outputs\petitRADTRANS\retrievals\carmenes_retrievals' + '/HD_189733_b_transmission_R_T0_Kp_V0_tiso_H2O_Pc_tmt0.80_t0r2000_all_nuaw_c819_100lp/retrieved_parameters.npz',
        allow_pickle=True)
    retrieved_parameters_ref2, retrieved_parameters2 = init_retrieved_parameters(
        retrieval_parameters,
        mid_transit_time_jd,
        mid_transit_time_range,
        external_parameters_ref=external_parameters_ref2
    )

    retrieved_parameters_tmp = {}

    for k in parameter_names_ref:
        retrieved_parameters_tmp[k] = copy.deepcopy(retrieved_parameters2[k])

    retrieved_parameters2 = copy.deepcopy(retrieved_parameters_tmp)
    del retrieved_parameters_tmp

    plot_result_corner(
        result_directory=[
            retrieval_directory +
            r'\HD_189733_b_transmission_'
            r'R_T0_Kp_V0_tiso_H2O_Pc_tmt0.80_t0r2000_alex4_c817_100lp',
            retrieval_directory +
            r'\HD_189733_b_transmission_'
            r'R_T0_Kp_V0_tiso_H2O_Pc_tmt0.80_t0r2000_alex4_sys-sub_c817_100lp',
        ],
        sm=None,
        retrieved_parameters=retrieved_parameters2,
        figure_directory=figure_directory,
        figure_name='corner_simple_retrieval_t02000s',
        label_kwargs={'fontsize': 16},
        title_kwargs={'fontsize': 13},
        true_values=None,
        color_list=['C0', 'C2', 'C1'],
        figure_font_size=12,
        save=True
    )

    # Polyfit
    directories = []

    for f in os.scandir(retrieval_directory):
        if f.is_dir() and 'HD_189733_b_transmission' in f.path and '_sim' not in f.path and 'sys-' not in f.path \
                and 't0r300' in f.path and 'nuaw_c819' in f.path and 'alex4' in f.path and 'beta' not in f.path \
                and 'opu' not in f.path:
            directories.append(f.path)

    spectral_models, log_es, log_ls, chi2s, sds = all_best_fit_stats(directories, retrieved_parameters=retrieved_parameters)

    # SysRem
    # directories = []
    #
    # for f in os.scandir(retrieval_directory):
    #     if (f.is_dir() and 'HD_189733_b_transmission' in f.path and '_sim' not in f.path and 'sys-p1-i10-sub' in f.path
    #             and 't0r300' in f.path and 'nuaw' in f.path and 'c819' in f.path and 'alex4' in f.path
    #             and 'beta' not in f.path and 'opu' not in f.path):
    #         directories.append(f.path)
    #
    # spectral_models_sys, log_es_sys, log_ls_sys, chi2s_sys, sds_sys = all_best_fit_stats(directories, retrieved_parameters=retrieved_parameters)

    # Simulations
    directories = []

    for f in os.scandir(retrieval_directory):
        if f.is_dir() and 'HD_189733_b_transmission' in f.path and '_sim' in f.path and 'sys-' not in f.path \
                and 't0r300' in f.path and 'nuaw_c819' in f.path and 'alex4' in f.path and 'beta' not in f.path \
                and 'opu' not in f.path and '3d_f2' not in f.path and '_ccn' not in f.path:
            directories.append(f.path)

    spectral_models, log_es, log_ls, chi2s, sds = all_best_fit_stats(directories, retrieved_parameters=retrieved_parameters)

    # A00: exp_CCD_param
    directories_hist = {
        'P-10': 'HD_189733_b_transmission_R_T0_Kp_V0_g_tiso_CH4_CO_H2O_H2S_HCN_NH3_Pc_k0_gams_tmt0.80_t0r300_alex4_nuaw_c819_100lp',
        'P-11': 'HD_189733_b_transmission_R_T0_Kp_V0_tiso_CH4_CO_H2O_H2S_HCN_NH3_Pc_k0_gams_tmt0.80_t0r300_alex4_nuaw_c819_100lp',
        'P-12': 'HD_189733_b_transmission_R_T0_Kp_V0_tiso_CH4_CO_H2O_H2S_HCN_NH3_tmt0.80_t0r300_alex4_nuaw_c819_100lp',
        'P-13': 'HD_189733_b_transmission_R_T0_Kp_V0_tiso_CO_H2O_H2S_tmt0.80_t0r300_alex4_nuaw_c819_100lp',
        'P-14': 'HD_189733_b_transmission_R_T0_Kp_V0_tiso_H2O_Pc_tmt0.80_t0r300_alex4_nuaw_c819_100lp',
        'P-15': 'HD_189733_b_transmission_R_Kp_V0_tiso_H2O_Pc_tmt0.80_t0r300_alex4_nuaw_c819_100lp',
        'P-16': 'HD_189733_b_transmission_T0_Kp_V0_tiso_H2O_Pc_tmt0.80_t0r300_alex4_nuaw_c819_100lp',
        'P-17': 'HD_189733_b_transmission_R_T0_Kp_V0_tiso_H2O_k0_gams_tmt0.80_t0r300_alex4_nuaw_c819_100lp',
        'P-18': 'HD_189733_b_transmission_R_T0_Kp_V0_tiso_H2O_tmt0.80_t0r300_alex4_nuaw_c819_100lp',
        'P-01': 'HD_189733_b_transmission_Kp_V0_tiso_H2O_Pc_tmt0.80_t0r300_alex4_nuaw_c819_100lp',

        # 'S-10': 'HD_189733_b_transmission_R_T0_Kp_V0_g_tiso_CH4_CO_H2O_H2S_HCN_NH3_Pc_k0_gams_tmt0.80_t0r300_alex4_sys-p1-i10-sub_nuaw_c819_100lp',
        # 'S-11': 'HD_189733_b_transmission_R_T0_Kp_V0_tiso_CH4_CO_H2O_H2S_HCN_NH3_Pc_k0_gams_tmt0.80_t0r300_alex4_sys-p1-i10-sub_nuaw_c819_100lp',
        # 'S-13': 'HD_189733_b_transmission_R_Kp_V0_tiso_CO_H2O_H2S_tmt0.80_t0r300_alex4_sys-p1-i10-sub_nuaw_c819_100lp',
        # 'S-14': 'HD_189733_b_transmission_R_T0_Kp_V0_tiso_H2O_Pc_tmt0.80_t0r300_alex4_sys-p1-i10-sub_nuaw_c819_100lp',
        # 'S-16': 'HD_189733_b_transmission_T0_Kp_V0_tiso_H2O_Pc_tmt0.80_t0r300_alex4_sys-p1-i10-sub_nuaw_c819_100lp',
        # 'S-18': 'HD_189733_b_transmission_R_Kp_V0_tiso_H2O_tmt0.80_t0r300_alex4_sys-p1-i10-sub_nuaw_c819_100lp',
        # 'S-01': 'HD_189733_b_transmission_Kp_V0_tiso_H2O_Pc_tmt0.80_t0r300_alex4_sys-p1-i10-sub_nuaw_c819_100lp',
    }

    colors = {'P': 'C0', 'S': 'C2'}
    colors = {key: colors[key.split('-', 1)[0]] for key in directories_hist}

    log_evidences = np.zeros(len(directories_hist))

    for i, d in enumerate(directories_hist):
        directories_hist[d] = os.path.join(retrieval_directory, directories_hist[d])

        # for j, dd in enumerate(directories):
        #     if dd == d:
        #         log_evidences[i] = log_es[j]

    plot_multiple_hists_data(
        result_directory=directories_hist,
        retrieved_parameters=retrieved_parameters,
        true_values=None,
        parameter_names_ref=parameter_names_ref,
        figure_font_size=9.5,
        save=True,
        color=colors,
        add_rectangle=7,
        figure_directory=figure_directory,
        figure_name='retrievals_posteriors',
        image_format=image_format
    )

    # A00: exp_CCD_param
    directories_hist = {
        'S-10-1': 'HD_189733_b_transmission_R_T0_Kp_V0_g_tiso_CH4_CO_H2O_H2S_HCN_NH3_Pc_k0_gams_tmt0.80_t0r300_alex4_sys-p1-i10-sub_nuaw_c819_100lp',
        'S-01-1': 'HD_189733_b_transmission_Kp_V0_tiso_H2O_Pc_tmt0.80_t0r300_alex4_sys-p1-i10-sub_nuaw_c819_100lp',
        'S-18-1': 'HD_189733_b_transmission_R_T0_Kp_V0_tiso_H2O_tmt0.80_t0r300_alex4_sys-p1-i10-sub_nuaw_c819_100lp',
        'S-18-2': 'HD_189733_b_transmission_R_T0_Kp_V0_tiso_H2O_tmt0.80_t0r300_alex4_sys-p2-i10-sub_nuaw_c819_100lp',
    }

    colors = {'S-10-1': 'C2', 'S-01-1': 'C2', 'S-18-1': 'C2', 'S-18-2': 'C3'}
    #colors = {key: colors[key.split('-', 1)[0]] for key in directories_hist}

    log_evidences = np.zeros(len(directories_hist))

    for i, d in enumerate(directories_hist):
        directories_hist[d] = os.path.join(retrieval_directory, directories_hist[d])

        # for j, dd in enumerate(directories):
        #     if dd == d:
        #         log_evidences[i] = log_es[j]

    plot_multiple_hists_data(
        result_directory=directories_hist,
        retrieved_parameters=retrieved_parameters,
        true_values=None,
        parameter_names_ref=parameter_names_ref,
        figure_font_size=9.5,
        save=True,
        color=colors,
        add_rectangle=None,
        figure_directory=figure_directory,
        figure_name='retrievals_posteriors_sysrem',
        image_format=image_format
    )

    # A00: exp_CCD_param
    directories_hist = {
        'P-19': 'HD_189733_b_transmission_R_T0_Kp_V0_tiso_H2O_Pc_beta_tmt0.80_t0r300_alex4_nuaw_c819_100lp',
        'P-29': 'HD_189733_b_transmission_R_T0_Kp_V0_tiso_H2O_Pc_opu_tmt0.80_t0r300_alex4_nuaw_c819_100lp',
    }

    colors = {'P-19': 'C0', 'P-29': 'C4'}
    #colors = {key: colors[key.split('-', 1)[0]] for key in directories_hist}

    log_evidences = np.zeros(len(directories_hist))

    for i, d in enumerate(directories_hist):
        directories_hist[d] = os.path.join(retrieval_directory, directories_hist[d])

    external_parameters_ref2 = np.load(
        r'C:\Users\Doriann\Documents\work\run_outputs\petitRADTRANS\retrievals\carmenes_retrievals' +
        '/HD_189733_b_transmission_R_T0_Kp_V0_tiso_H2O_Pc_beta_tmt0.80_t0r300_alex4_nuaw_c819_100lp/retrieved_parameters.npz',
        allow_pickle=True)
    retrieved_parameters_ref2, retrieved_parameters2 = init_retrieved_parameters(
        retrieval_parameters,
        mid_transit_time_jd,
        mid_transit_time_range,
        external_parameters_ref=external_parameters_ref2
    )

    retrieved_parameters2['mid_transit_time'] = copy.deepcopy(retrieved_parameters['mid_transit_time'])

    parameter_names_ref.append('beta')
    parameter_names_ref2 = copy.deepcopy(parameter_names_ref)
    for key in parameter_names_ref:
        print(key, key in list(external_parameters_ref2.keys()))
        if key not in list(external_parameters_ref2.keys()):
            print(key)
            parameter_names_ref2.remove(key)

    plot_multiple_hists_data(
        result_directory=directories_hist,
        retrieved_parameters=retrieved_parameters2,
        true_values=None,
        parameter_names_ref=parameter_names_ref2,
        figure_font_size=9.5,
        save=True,
        color=colors,
        add_rectangle=None,
        figure_directory=figure_directory,
        figure_name='retrievals_posteriors_beta',
        image_format=image_format
    )

    # A00: exp_CCD_param
    parameter_names_ref = [
        'radial_velocity_semi_amplitude',
        'rest_frame_velocity_shift',
        'temperature',
        'H2O_main_iso',
        'mid_transit_time',
        'new_resolving_power',
        'log10_planet_surface_gravity',
    ]

    directories_hist = {
        'Poly-O:N-B:N-01': 'HD_189733_b_transmission_R_T0_Kp_V0_g_tiso_H2O_tmt0.80_alex4_c813_100lp',
        'Poly-O:N-B:Y-01': 'HD_189733_b_transmission_R_T0_Kp_V0_g_tiso_H2O_tmt0.80_alex4_bxctlloss_c813_100lp',
        'SysS-O:N-B:N-01': 'HD_189733_b_transmission_R_T0_Kp_V0_g_tiso_H2O_tmt0.80_alex4_sys-sub_c813_100lp',
        'SysD-O:N-B:N-01': 'HD_189733_b_transmission_R_T0_Kp_V0_g_tiso_H2O_tmt0.80_alex4_sys-div_c813_100lp',

        'Poly-O:Y-B:N-01': 'HD_189733_b_transmission_R_T0_Kp_V0_g_tiso_H2O_tmt0.80_alex4_adddoff_c813_100lp',
        'Poly-O:Y-B:Y-01': 'HD_189733_b_transmission_R_T0_Kp_V0_g_tiso_H2O_tmt0.80_alex4_adddoff_bxctlloss_c813_100lp',
        # 'SysS-O:Y-B:N-01': 'HD_189733_b_transmission_R_T0_Kp_V0_g_tiso_H2O_tmt0.80_alex4_adddoff_sys-sub_c813_100lp',
        # 'SysD-O:Y-B:N-01': 'HD_189733_b_transmission_R_T0_Kp_V0_g_tiso_H2O_tmt0.80_alex4_adddoff_sys-div_c813_100lp',

        'Poly-O:N-B:N-02': 'HD_189733_b_transmission_R_T0_Kp_V0_tiso_H2O_tmt0.80_alex4_c813_100lp',
        'Poly-O:N-B:Y-02': 'HD_189733_b_transmission_R_T0_Kp_V0_tiso_H2O_tmt0.80_alex4_bxctlloss_c813_100lp',
        'SysS-O:N-B:N-02': 'HD_189733_b_transmission_R_T0_Kp_V0_tiso_H2O_tmt0.80_alex4_sys-sub_c813_100lp',
        'SysD-O:N-B:N-02': 'HD_189733_b_transmission_R_T0_Kp_V0_tiso_H2O_tmt0.80_alex4_sys-div_c813_100lp',

        'Poly-O:Y-B:N-02': 'HD_189733_b_transmission_R_T0_Kp_V0_tiso_H2O_tmt0.80_alex4_adddoff_c813_100lp',
        'Poly-O:Y-B:Y-02': 'HD_189733_b_transmission_R_T0_Kp_V0_tiso_H2O_tmt0.80_alex4_adddoff_bxctlloss_c813_100lp',
        # 'SysS-O:Y-B:N-02': 'HD_189733_b_transmission_T0_Kp_V0_tiso_H2O_tmt0.80_alex4_adddoff_sys-sub_c813_100lp',
        # 'SysD-O:Y-B:N-02': 'HD_189733_b_transmission_T0_Kp_V0_tiso_H2O_tmt0.80_alex4_adddoff_sys-div_c813_100lp',
        #
        'Poly-O:N-B:N-03': 'HD_189733_b_transmission_Kp_V0_g_tiso_H2O_tmt0.80_alex4_c813_100lp',
        'Poly-O:N-B:Y-03': 'HD_189733_b_transmission_Kp_V0_tiso_H2O_tmt0.80_alex4_bxctlloss_c813_100lp',
        'SysS-O:N-B:N-03': 'HD_189733_b_transmission_Kp_V0_tiso_H2O_tmt0.80_alex4_sys-sub_c813_100lp',
        'SysD-O:N-B:N-03': 'HD_189733_b_transmission_Kp_V0_tiso_H2O_tmt0.80_alex4_sys-div_c813_100lp',

        'Poly-O:Y-B:N-03': 'HD_189733_b_transmission_Kp_V0_tiso_H2O_tmt0.80_alex4_adddoff_c813_100lp',
        'Poly-O:Y-B:Y-03': 'HD_189733_b_transmission_Kp_V0_tiso_H2O_tmt0.80_alex4_adddoff_bxctlloss_c813_100lp',
        # 'SysS-O:Y-B:N-01': 'HD_189733_b_transmission_R_T0_Kp_V0_g_tiso_H2O_tmt0.80_alex4_adddoff_sys-sub_c813_100lp',
        # 'SysD-O:Y-B:N-01': 'HD_189733_b_transmission_R_T0_Kp_V0_g_tiso_H2O_tmt0.80_alex4_adddoff_sys-div_c813_100lp',

        'Poly-O:N-B:N-04': 'HD_189733_b_transmission_R_Kp_V0_tiso_H2O_tmt0.80_alex4_c813_100lp',
        'Poly-O:N-B:Y-04': 'HD_189733_b_transmission_R_Kp_V0_tiso_H2O_tmt0.80_alex4_bxctlloss_c813_100lp',
        'SysS-O:N-B:N-04': 'HD_189733_b_transmission_R_Kp_V0_tiso_H2O_tmt0.80_alex4_sys-sub_c813_100lp',
        'SysD-O:N-B:N-04': 'HD_189733_b_transmission_R_Kp_V0_tiso_H2O_tmt0.80_alex4_sys-div_c813_100lp',

        'Poly-O:Y-B:N-04': 'HD_189733_b_transmission_R_Kp_V0_tiso_H2O_tmt0.80_alex4_adddoff_c813_100lp',
        'Poly-O:Y-B:Y-04': 'HD_189733_b_transmission_R_Kp_V0_tiso_H2O_tmt0.80_alex4_adddoff_bxctlloss_c813_100lp',
        # 'SysS-O:Y-B:N-01': 'HD_189733_b_transmission_R_T0_Kp_V0_g_tiso_H2O_tmt0.80_alex4_adddoff_sys-sub_c813_100lp',
        # 'SysD-O:Y-B:N-01': 'HD_189733_b_transmission_R_T0_Kp_V0_g_tiso_H2O_tmt0.80_alex4_adddoff_sys-div_c813_100lp',

        'Poly-G:N-05': 'HD_189733_b_transmission_R_T0_g_Kp_V0_tiso_H2O_tmt0.80_alex4_c813_100lp',
        'Poly-G:Y-05': 'HD_189733_b_transmission_R_T0_g_Kp_V0_tiso_H2O_gp_tmt0.80_alex4_c813_100lp',
        'SysS-G:N-05': 'HD_189733_b_transmission_R_T0_Kp_V0_tiso_H2O_gp_tmt0.80_alex4_sys-sub_c813_100lp',
        'SysS-O:N-B:N-05': 'HD_189733_b_transmission_R_T0_Kp_V0_g_tiso_H2O_tmt0.80_alex4_sys-sub_c813_100lp',
    }

    colors = {
        'Poly-O:N-B:N-01': 'C0',
        'Poly-O:N-B:Y-01': 'C0',
        'SysS-O:N-B:N-01': 'C0',
        'SysD-O:N-B:N-01': 'C0',

        'Poly-O:Y-B:N-01': 'C1',
        'Poly-O:Y-B:Y-01': 'C1',
        'SysS-O:Y-B:N-01': 'C1',
        'SysD-O:Y-B:N-01': 'C1',

        'Poly-O:N-B:N-02': 'C0',
        'Poly-O:N-B:Y-02': 'C0',
        'SysS-O:N-B:N-02': 'C0',
        'SysD-O:N-B:N-02': 'C0',

        'Poly-O:Y-B:N-02': 'C1',
        'Poly-O:Y-B:Y-02': 'C1',
        'SysS-O:Y-B:N-02': 'C1',
        'SysD-O:Y-B:N-02': 'C1',

        'Poly-O:N-B:N-03': 'C0',
        'Poly-O:N-B:Y-03': 'C0',
        'SysS-O:N-B:N-03': 'C0',
        'SysD-O:N-B:N-03': 'C0',

        'Poly-O:Y-B:N-03': 'C1',
        'Poly-O:Y-B:Y-03': 'C1',
        'SysS-O:Y-B:N-03': 'C1',
        'SysD-O:Y-B:N-03': 'C1',

        'Poly-O:Y-B:N-04': 'C0',
        'Poly-O:Y-B:Y-04': 'C0',
        'SysS-O:Y-B:N-04': 'C0',
        'SysD-O:Y-B:N-04': 'C0',
    }

    for d in directories_hist:
        directories_hist[d] = os.path.join(retrieval_directory, directories_hist[d])

    plot_multiple_hists_data(
        result_directory=directories_hist,
        sm=sm,
        retrieved_parameters=retrieved_parameters,
        true_values=None,
        parameter_names_ref=parameter_names_ref,
        figure_font_size=10,
        save=True,
        color=colors,
        add_rectangle=None,
        figure_directory=figure_directory,
        figure_name='retrievals_posteriors',
        image_format=image_format
    )

    # A00: exp_CCD_param
    directories_hist = {
        'TTR-06': 'HD_189733_b_transmission_Kp_V0_tiso_H2O_alex2_t1535_t23_c211_100lp',
        'A': 'HD_189733_b_transmission_Kp_V0_tiso_H2O_strictt15353_t1535_t23_c_1000lp',
        'B': 'HD_189733_b_transmission_Kp_V0_tiso_H2O_bad3_t1535_t23_c_1000lp',
    }

    for i, d in enumerate(directories_hist):
        if i > 0:
            directories_hist[d] = os.path.join(retrieval_directory, 'old', directories_hist[d])
        else:
            directories_hist[d] = os.path.join(retrieval_directory, directories_hist[d])

    plot_multiple_hists_data(
        result_directory=directories_hist,
        sm=sm,
        retrieved_parameters=retrieved_parameters,
        true_values=None,
        parameter_names_ref=parameter_names_ref,
        figure_font_size=11,
        fig_size=6.4,
        save=True,
        color='C0',
        add_rectangle=None,
        figure_directory=figure_directory,
        figure_name='retrievals_posteriors_test',
        image_format=image_format
    )

    # 3D comparison figure
    base_wavelengths, base_spectrum = sm.calculate_spectrum(
        radtrans=radtrans,
        mode='transmission',
        update_parameters=True,
        telluric_transmittances_wavelengths=None,
        telluric_transmittances=None,
        instrumental_deformations=None,
        noise_matrix=None,
        scale=False,
        shift=False,
        convolve=False,
        rebin=False,
        prepare=False
    )

    orange_model_with_rotation = np.load(
        'C:/Users/Doriann/Documents/work/run_outputs/petitRADTRANS/data/carmenes/hd_189733_b/simu_orange/'
        'orange_hd_189733_b_transmission.npz'
    )

    orange_wavelengths_no_rotation, orange_spectrum_no_rotation = load_orange_simulation_dat(
        'C:/Users/Doriann/Documents/work/run_outputs/petitRADTRANS/data/carmenes/hd_189733_b/simu_orange/no_rotation',
        'range', 'dat', 31
    )

    plt.figure(figsize=(6.4, 4.8))
    update_figure_font_size(MEDIUM_FIGURE_FONT_SIZE)
    plt.plot(base_wavelengths[0, 110000:120000] * 1e-6,
             base_spectrum[0, 110000:120000] * 1e-5,
             color='k', label='1-D model')
    plt.plot(orange_wavelengths_no_rotation * 1e-6, orange_spectrum_no_rotation * 1e-5,
             color='C4', label='3-D model')
    plt.plot(orange_model_with_rotation['wavelengths'] * 1e-6, orange_model_with_rotation['data'] * 1e-5,
             color='C1', label='3-D model (rotation + wind)')
    plt.xlim(np.array([1.5205 - 0.0005, 1.5205 + 0.0005]) * 1e-6)
    plt.xlabel('Wavelength (m)')
    plt.ylabel(r'$R_p$ (km)')
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(figure_directory, '3d_model_comparison' + '.' + image_format))

    # Best fit figure
    hst_data = np.loadtxt(r'C:\Users\Doriann\Documents\work\run_outputs\petitRADTRANS\data\hst_wfc3\hd_189733_b\transit_depths_kilpatrick2020.dat')
    model_files = r'\\wsl$\Debian\home\dblain\exorem\outputs\exorem\hd_1899733_b_z10_t100_co0.55_nocloud.h5'
    dir_best_fits = [
        r'C:\Users\Doriann\Documents\work\run_outputs\petitRADTRANS\retrievals\carmenes_retrievals\HD_189733_b_transmission_Kp_V0_tiso_H2O_Pc_tmt0.80_t0r300_alex4_nuaw_c819_100lp',
        # r'C:\Users\Doriann\Documents\work\run_outputs\petitRADTRANS\retrievals\carmenes_retrievals\HD_189733_b_transmission_Kp_V0_tiso_H2O_Pc_tmt0.80_t0r300_alex4_sys-p1-i10-sub_nuaw_c819_100lp',
        r'C:\Users\Doriann\Documents\work\run_outputs\petitRADTRANS\retrievals\carmenes_retrievals\HD_189733_b_transmission_R_Kp_V0_tiso_H2O_tmt0.80_t0r300_alex4_nuaw_c819_100lp',
        # r'C:\Users\Doriann\Documents\work\run_outputs\petitRADTRANS\retrievals\carmenes_retrievals\HD_189733_b_transmission_R_Kp_V0_tiso_CO_H2O_H2S_tmt0.80_t0r300_alex4_sys-p1-i10-sub_nuaw_c819_100lp',
    ]

    colors = [
        'k',
        'C0',
        # 'C2',
        'C0',
        # 'C2'
    ]
    linestyles = [
        '-',
        ':',
        # ':',
        '-',
        # '-'
    ]
    linewidths = [
        1.5,
        3,
        # 3,
        1.5,
        # 1.5
    ]
    labels = [
        r'Exo-REM (Z = 10)',
        'Polyfit (P-01)',
        # 'SysRem (S-01)',
        'Polyfit (P-18)',
        # 'SysRem (S-13)'
    ]
    envelope = [
        False,
        # False,
        True,
        # True
    ]
    planet_radius_offsets = [
        2400e5,
        3600e5,
        # 3800e5,
        3600e5,
        # 0.3800e5
    ]
    rebin_wavelengths = wavelengths_instrument_0

    sms_best_fit, wavelengths_best_fits, spectrum_best_fits, w_er, s_er, fig, axe = plot_best_fit_comparison(
        exorem_file=model_files,
        model_directories=dir_best_fits,
        data=hst_data,
        radtrans=radtrans,
        resolving_power=500,
        rebin=True,
        planet_radius_offsets=planet_radius_offsets,
        rebin_wavelengths=rebin_wavelengths,
        order_selection=detector_selection,
        colors=colors,
        labels=labels,
        linestyles=linestyles,
        linewidths=linewidths,
        envelope=envelope,
        data_color='r',
        data_linestyle='',
        data_marker='+',
        data_label='HST WFC3 data (Kilpatrick et al. 2020)',
        xlim=[np.min(wavelengths_instrument) * 1e-6 - 0.01e-6, np.max(wavelengths_instrument) * 1e-6 + 0.01e-6],
        legend=True,
        save=True,
        figure_directory=figure_directory,
        figure_name='best_fit_comparison',
        image_format='pdf',
        figure_font_size=18,
        figsize=(6.4 * 3, 4.8 * 1.5)
    )


if __name__ == '__main__':
    pass
