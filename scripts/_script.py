"""
Script to launch a CCF analysis on multiple models.
"""
import h5py
import os.path

from scripts._plot_functions import *
from scripts.ccf_utils import *
from scripts.model_containers import *

species_color = {
    'CH4': 'C7',
    'CO': 'C3',
    'CO2': 'C5',
    'H2O': 'C0',
    'H2S': 'C8',
    'HCN': 'C4',
    'K': 'y',
    'Na': 'gold',
    'NH3': 'C9',
    'PH3': 'C1'
}


def main():
    # Base parameters
    planet_name = 'WASP-39 b'
    lbl_opacity_sampling = 1
    do_scat_emis = False
    model_suffix = ''
    wlen_modes = {
        # 'Y': np.array([0.92, 1.15]),
        # 'J': np.array([1.07, 1.4]),
        # 'H': np.array([1.4, 1.88]),
        'K': np.array([1.88, 2.55]),
        # 'L': np.array([2.7, 4.25]),
        # 'M': np.array([3.25, 5.5])
    }

    pressures = np.logspace(-10, 2, 130)
    distances = [213.982 * nc.pc]  # np.logspace(1, 3, 7) * nc.c * 3600 * 24 * 365.25
    # star_apparent_magnitude_v = 12.095
    star_apparent_magnitude_j = 10.663
    # star_apparent_magnitude_j = 10
    star_apparent_magnitudes = [star_apparent_magnitude_j]  # np.linspace(4, 16, 7)

    # Models to be tested
    t_int = [200]
    metallicity = [1]
    co_ratio = [0.55]
    p_cloud = [1e2]  # [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4]
    species_list = ['all', 'H2O', 'CO', 'CO2', 'CO_36']

    co_all_iso_mmr = {
        1: np.array([0.04963663, 0.04963663, 0.04963663, 0.04963663, 0.04963663,
           0.04963663, 0.04963663, 0.04963663, 0.04963663, 0.04963663,
           0.04963663, 0.04963663, 0.04963663, 0.04963663, 0.04963663,
           0.04963663, 0.04963663, 0.04963663, 0.04963663, 0.04963663,
           0.04963663, 0.04963663, 0.04963689, 0.04963688, 0.04963686,
           0.04963685, 0.04963684, 0.04963683, 0.04963681, 0.0496368 ,
           0.04963679, 0.04963678, 0.04963677, 0.04963676, 0.04963674,
           0.04963673, 0.04963671, 0.0496367 , 0.04963669, 0.04963668,
           0.04963667, 0.04963667, 0.04963667, 0.04963667, 0.04963667,
           0.04963668, 0.04963668, 0.0496367 , 0.04963671, 0.04963674,
           0.04963677, 0.0496368 , 0.04963685, 0.04963691, 0.04963698,
           0.04963707, 0.04963718, 0.04963731, 0.04963748, 0.04963769,
           0.04963794, 0.04963825, 0.04963863, 0.0496391 , 0.04963968,
           0.04964039, 0.04964125, 0.04964231, 0.0496436 , 0.04964517,
           0.04964685, 0.04964863, 0.04965078, 0.04965333, 0.04965636,
           0.04965998, 0.0496641 , 0.04966802, 0.04967224, 0.04967694,
           0.04968173, 0.04968604, 0.04969051, 0.04969465, 0.04969811,
           0.04970132, 0.04970404, 0.04970585, 0.04970719, 0.04970803,
           0.04970843, 0.04970844, 0.04970813, 0.04970745, 0.04970635,
           0.04970464, 0.04970205, 0.04969816, 0.0496924 , 0.04968354,
           0.04967023, 0.04965077, 0.04962285, 0.04958384, 0.04953114,
           0.04946033, 0.04934505, 0.04918709, 0.0489805 , 0.04872653,
           0.04844177, 0.04816132, 0.04787711, 0.04746033, 0.04716749,
           0.04701968, 0.04678526, 0.0468406 , 0.04680158, 0.04693091,
           0.04719836, 0.0474274 , 0.04765134, 0.04790435, 0.04814762,
           0.04840958, 0.04869867, 0.04895301, 0.04908404, 0.04914225]),
        2:
            np.array([0.24481139, 0.24481139, 0.24481139, 0.24481139, 0.24481139,
                0.24481139, 0.24481139, 0.24481139, 0.24481139, 0.24481139,
                0.2448114 , 0.2448114 , 0.2448114 , 0.24481141, 0.24481141,
                0.24481142, 0.24481143, 0.24481144, 0.24481145, 0.24481147,
                0.24481149, 0.24481152, 0.244822  , 0.24482142, 0.24482082,
                0.24482019, 0.24481961, 0.24481904, 0.24481849, 0.244818  ,
                0.24481756, 0.2448172 , 0.24481693, 0.24481675, 0.24481658,
                0.24481648, 0.24481656, 0.24481682, 0.24481736, 0.2448182 ,
                0.24481941, 0.24482105, 0.24482322, 0.24482604, 0.24482964,
                0.24483419, 0.24483987, 0.24484695, 0.24485571, 0.24486655,
                0.24487989, 0.24489628, 0.24491638, 0.24494101, 0.24497113,
                0.24500792, 0.24505275, 0.24510729, 0.24517352, 0.24524913,
                0.24532617, 0.24541863, 0.24552893, 0.24565969, 0.2458135 ,
                0.24599272, 0.24617682, 0.24636386, 0.24657459, 0.24680123,
                0.24699758, 0.24720382, 0.24740705, 0.24757238, 0.24772764,
                0.24786619, 0.24796173, 0.24803503, 0.248087  , 0.24812009,
                0.24813886, 0.24814876, 0.24815443, 0.24815883, 0.24816338,
                0.24816864, 0.24817487, 0.2481822 , 0.24819095, 0.24820144,
                0.24821407, 0.24822929, 0.24824771, 0.24826994, 0.24829693,
                0.24832504, 0.24835721, 0.24839613, 0.2484432 , 0.2485    ,
                0.24856866, 0.24865105, 0.24873068, 0.24882622, 0.24894027,
                0.24904875, 0.24917809, 0.2493004 , 0.24943391, 0.24956675,
                0.2496923 , 0.2498149 , 0.24993108, 0.25003896, 0.2501363 ,
                0.25021565, 0.2502555 , 0.25021982, 0.25007373, 0.24997653,
                0.25002683, 0.25009511, 0.25015464, 0.25020642, 0.25025138,
                0.2502903 , 0.25032493, 0.25035653, 0.25038543, 0.25036323])
    }

    line_species_list = [
        # 'CH4_main_iso',
        # 'CO_all_iso',
        'CO_main_iso',
        'CO_36',
        'CO2_main_iso',
        'H2O_main_iso',
        # 'H2S_main_iso',
        # 'HCN_main_iso',
        # 'K',
        # 'Na_allard',
        # 'NH3_main_iso',
        # 'PH3_main_iso'
    ]

    # Observation parameters
    # Actually matter (used to get the CRIRES SNR data from the ETC website)
    exposure_time = 6 * 3600  # 4 * 3600
    integration_time = 60
    airmass = 1.2
    velocity_range = [-1400, 1400]
    instrument_resolving_power = 8e4
    # Old (don't do anything anymore)
    telescope_mirror_radius = 8.2e2 / 2  # cm
    telescope_throughput = 0.1
    pixel_sampling = 3

    # Load settings
    settings = load_wavelength_settings(module_dir + '/crires/wavelength_settings.dat')

    # Load planet
    planet = Planet.get(planet_name)

    # Load signal to noise_matrix ratios
    star_snr = get_multiple_crires_snr_data(settings, star_apparent_magnitude_j,  # TODO apparent mag is really annoying
                                            planet.star_effective_temperature, exposure_time,
                                            integration_time, airmass,  # TODO add FLI and seeing
                                            rewrite=False, star_apparent_magnitude_band='J')

    # Generate parameter dictionaries
    parameter_dicts = get_parameter_dicts(
        t_int, metallicity, co_ratio, p_cloud
    )

    # Load/generate relevant models
    models = {}

    for wlen_mode in wlen_modes:
        print(f"Band {wlen_mode}...")

        # Initialize grid
        models[wlen_mode], all_models_exist = init_model_grid(
            planet_name, lbl_opacity_sampling, do_scat_emis, parameter_dicts, species_list,
            wavelength_boundaries=wlen_modes[wlen_mode],
            model_suffix=model_suffix
        )

        if not all_models_exist:
            # Load or generate atmosphere
            atmosphere, atmosphere_filename = SpectralModelLegacy.get_atmosphere_model(
                wlen_bords_micron=wlen_modes[wlen_mode],
                pressures=pressures,
                line_species_list=line_species_list,
                rayleigh_species=SpectralModelLegacy.default_rayleigh_species,
                continuum_opacities=SpectralModelLegacy.default_continuum_opacities,
                lbl_opacity_sampling=lbl_opacity_sampling,
                do_scat_emis=do_scat_emis,
                model_suffix=model_suffix
            )

            # Load or generate modelsfr
            models[wlen_mode] = generate_model_grid(
                models=models[wlen_mode],
                pressures=pressures,
                line_species_list=line_species_list,
                rayleigh_species='default',
                continuum_opacities='default',
                model_suffix=model_suffix,
                atmosphere=atmosphere,
                mass_fractions={'CO_main_iso': co_all_iso_mmr[1] * 0.97, 'CO_36': co_all_iso_mmr[1] * 0.03},
                calculate_transmission_spectrum=True,
                calculate_eclipse_depth=False,
                rewrite=True,
                save=True
            )
        else:
            # Load existing models
            models[wlen_mode] = load_model_grid(models[wlen_mode])

    snrs = {}
    snrs_error = {}
    tsm = {}

    for meta in metallicity:
        snrs[meta] = {}
        snrs_error[meta] = {}
        tsm[meta] = {}

        for band in wlen_modes:
            snrs[meta][band], snrs_error[meta][band], tsm[meta][band], results = get_tsm_snr_pcloud(
                band=band,
                wavelength_boundaries=wlen_modes[band] * 1e-4,
                star_distances=distances,
                p_clouds=p_cloud,
                models=models,
                species_list=species_list,
                settings=settings,
                planet=planet,
                t_int=t_int[0],
                metallicity=meta,
                co_ratio=co_ratio[0],
                velocity_range=velocity_range,
                exposure_time=exposure_time,
                telescope_mirror_radius=telescope_mirror_radius,
                telescope_throughput=telescope_throughput,
                instrument_resolving_power=instrument_resolving_power,
                pixel_sampling=pixel_sampling,
                noise_correction_coefficient=1.0,
                scale_factor=1.0,
                star_snr=star_snr,
                star_apparent_magnitude=star_apparent_magnitudes,
                star_snr_reference_apparent_magnitude=star_apparent_magnitude_j,
                mock_observation_number=100,
                transit_number=4
            )

    for meta in metallicity:
        print(f'\n [Z/H] = {10 ** meta}:')

        for species in species_list:
            if species == 'all':
                continue

            best_snr, best_band, best_setting = find_best_setting(species, snrs[meta])
            print(f"Species '{species}', best setting: {best_band}{best_setting} (CCF SNR: {best_snr})")

    for species in species_list:
        if species == 'all':
            continue

        plt.figure(figsize=(16, 9))
        plot_snr_settings_bars(
            species, snrs,
            model_labels=[rf"Z/H = {10 ** metallicity[0]:.1f} $\times$ solar"],#,
                          # rf"Z/H = {10 ** metallicity[1]:.1f} $\times$ solar"],
            planet_name=planet_name,
            threshold=5,
            y_err=snrs_error
        )

        plt.savefig(f"./figures/{planet_name.replace(' ', '_')}/{species}_detection.png")

    plot_tsm_pcloud_snr(
        p_cloud, tsm, snrs,
        metallicity=metallicity[0],
        band='J',
        setting=list(settings.keys())[0],
        species=species_list[1],
        planet_name=planet_name,
        exposure_time=exposure_time
    )


def main_tic():
    # Base parameters
    planet_name = 'TOI-1130 c'
    lbl_opacity_sampling = 1
    do_scat_emis = False
    model_suffix = ['', 'neq']
    wlen_modes = {
        # 'Y': np.array([0.92, 1.15]),
        # 'J': np.array([1.07, 1.4]),
        'H': np.array([1.4, 1.88]),
        'K': np.array([1.88, 2.55]),
        # 'L': np.array([2.7, 4.25]),
        # 'M': np.array([3.25, 5.5])
    }

    pressures = np.logspace(-10, 2, 130)
    distances = [213.982 * nc.pc]  # np.logspace(1, 3, 7) * nc.c * 3600 * 24 * 365.25
    # star_apparent_magnitude_v = 12.095

    # Models to be tested
    t_int = [80]
    metallicity = [0, 1]
    co_ratio = [0.55]
    p_cloud = [1e2]  # [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4]
    species_list = ['all', 'H2O', 'CH4', 'NH3', 'CO', 'CO2', 'H2S', 'PH3']

    line_species_list = [
        'CH4_main_iso',
        'CO_all_iso',
        # 'CO_main_iso',
        # 'CO_36',
        'CO2_main_iso',
        'H2O_main_iso',
        'H2S_main_iso',
        # 'HCN_main_iso',
        # 'K',
        # 'Na_allard',
        'NH3_main_iso',
        'PH3_main_iso'
    ]

    molecular_w = {
        'CH4_main_iso': 16,
        'CO_all_iso': 28,
        'CO2_main_iso': 44,
        'H2O_main_iso': 18,
        'H2S_main_iso': 34.1,
        # 'HCN_main_iso': 27,
        'NH3_main_iso': 17,
        'PH3_main_iso': 34
    }

    data = load_hdf5(f"{module_dir}/../../../exorem/outputs/exorem/HD_97858_b_z100_g10_t80_co0.55_nocloud.h5")
    mmmr_neq = data['outputs']['layers']['molar_mass'][()][::-1]
    p_neq = data['outputs']['layers']['pressure'][()][::-1] * 1e-5
    vmr_neq = data['outputs']['layers']['volume_mixing_ratios']['absorbers']
    t_neq = data['outputs']['layers']['temperature'][()][::-1]
    mmr_neq = {}

    mmmr_neq_i = interp1d(p_neq, mmmr_neq, fill_value='extrapolate')
    mmmr_neq_i = mmmr_neq_i(pressures)
    t_neq_i = interp1d(p_neq, t_neq, fill_value='extrapolate')
    t_neq_i = t_neq_i(pressures)

    for species in list(vmr_neq.keys()):
        vmr_neq_i = interp1d(p_neq, vmr_neq[species][()][::-1], fill_value='extrapolate')
        vmr_neq_i = vmr_neq_i(pressures)

        for line_species in molecular_w:
            if species + '_' in line_species:
                mmr_neq[line_species] = molecular_w[line_species] * 1e-3 * vmr_neq_i / mmmr_neq_i

    mmr_neq['MMW'] = mmmr_neq_i * 1e3

    mass_fractions = [
        None,
        mmr_neq
    ]

    temperature_profiles = [
        None,
        t_neq_i
    ]

    # Observation parameters
    # Actually matter (used to get the CRIRES SNR data from the ETC website)
    exposure_time = 4 * 3600  # 4 * 3600
    integration_time = 60
    airmass = 1.2
    velocity_range = [-1400, 1400]
    instrument_resolving_power = 8e4
    # Old (don't do anything anymore)
    telescope_mirror_radius = 8.2e2 / 2  # cm
    telescope_throughput = 0.1
    pixel_sampling = 3

    # Load settings
    settings = load_wavelength_settings(module_dir + '/crires/wavelength_settings.dat')

    # Load planet
    planet = Planet.get(planet_name)
    star_apparent_magnitude_j = planet.system_apparent_magnitude_j
    # star_apparent_magnitude_j = 10
    star_apparent_magnitudes = [star_apparent_magnitude_j]  # np.linspace(4, 16, 7)

    # Load signal to noise_matrix ratios
    star_snr = get_multiple_crires_snr_data(settings, star_apparent_magnitude_j,  # TODO apparent mag is really annoying
                                            planet.star_effective_temperature, exposure_time,
                                            integration_time, airmass,  # TODO add FLI and seeing
                                            rewrite=False, star_apparent_magnitude_band='J')

    # Generate parameter dictionaries
    parameter_dicts = get_parameter_dicts(
        t_int, metallicity, co_ratio, p_cloud
    )

    # Load/generate relevant models
    models = {}
    id_model = 0

    for wlen_mode in wlen_modes:
        print(f"Band {wlen_mode}...")

        # Initialize grid
        models[wlen_mode], all_models_exist = init_model_grid(
            planet_name, lbl_opacity_sampling, do_scat_emis, parameter_dicts, species_list,
            wavelength_boundaries=wlen_modes[wlen_mode],
            model_suffix=model_suffix[id_model]
        )

        if not all_models_exist:
            # Load or generate atmosphere
            atmosphere, atmosphere_filename = SpectralModelLegacy.get_atmosphere_model(
                wlen_bords_micron=wlen_modes[wlen_mode],
                pressures=pressures,
                line_species_list=line_species_list,
                rayleigh_species=SpectralModelLegacy.default_rayleigh_species,
                continuum_opacities=SpectralModelLegacy.default_continuum_opacities,
                lbl_opacity_sampling=lbl_opacity_sampling,
                do_scat_emis=do_scat_emis,
                model_suffix=model_suffix[id_model]
            )

            # Load or generate models
            models[wlen_mode] = generate_model_grid(
                models=models[wlen_mode],
                pressures=pressures,
                line_species_list=line_species_list,
                rayleigh_species='default',
                continuum_opacities='default',
                model_suffix=model_suffix,
                atmosphere=atmosphere,
                mass_fractions=mass_fractions[id_model],
                temperature_profile=temperature_profiles[id_model],
                calculate_transmission_spectrum=True,
                calculate_eclipse_depth=False,
                rewrite=True,
                save=True
            )
        else:
            # Load existing models
            models[wlen_mode] = load_model_grid(models[wlen_mode])

    snrs = {}
    snrs_error = {}
    tsm = {}

    for meta in metallicity:
        snrs[meta] = {}
        snrs_error[meta] = {}
        tsm[meta] = {}

        for band in wlen_modes:
            snrs[meta][band], snrs_error[meta][band], tsm[meta][band], results = get_tsm_snr_pcloud(
                band=band,
                wavelength_boundaries=wlen_modes[band] * 1e-4,
                star_distances=distances,
                p_clouds=p_cloud,
                models=models,
                species_list=species_list,
                settings=settings,
                planet=planet,
                t_int=t_int[0],
                metallicity=meta,
                co_ratio=co_ratio[0],
                velocity_range=velocity_range,
                exposure_time=exposure_time,
                telescope_mirror_radius=telescope_mirror_radius,
                telescope_throughput=telescope_throughput,
                instrument_resolving_power=instrument_resolving_power,
                pixel_sampling=pixel_sampling,
                noise_correction_coefficient=1.0,
                scale_factor=1.0,
                star_snr=star_snr,
                star_apparent_magnitude=star_apparent_magnitudes,
                star_snr_reference_apparent_magnitude=star_apparent_magnitude_j,
                mock_observation_number=100,
                transit_number=1
            )

    for meta in metallicity:
        print(f'\n [Z/H] = {10 ** meta}:')

        for species in species_list:
            if species == 'all':
                continue

            best_snr, best_band, best_setting = find_best_setting(species, snrs[meta])
            print(f"Species '{species}', best setting: {best_band}{best_setting} (CCF SNR: {best_snr})")

    for species in species_list:
        if species == 'all':
            continue

        plt.figure(figsize=(16, 9))
        plot_snr_settings_bars(
            species, snrs,
            model_labels=[rf"Z/H = {10 ** metallicity[0]:.1f} $\times$ solar",
                          rf"Z/H = {10 ** metallicity[1]:.1f} $\times$ solar"],
            planet_name=planet_name,
            threshold=5,
            y_err=snrs_error
        )

        if not os.path.isdir(f"./figures/{planet_name.replace(' ', '_')}"):
            os.mkdir(f"./figures/{planet_name.replace(' ', '_')}")

        plt.savefig(f"./figures/{planet_name.replace(' ', '_')}/{species}_detection.png")

    plot_tsm_pcloud_snr(
        p_cloud, tsm, snrs,
        metallicity=metallicity[0],
        band='J',
        setting=list(settings.keys())[0],
        species=species_list[1],
        planet_name=planet_name,
        exposure_time=exposure_time
    )


def main_wasp():
    # Base parameters
    planet_name = 'WASP-96 b'
    lbl_opacity_sampling = 1
    do_scat_emis = False
    model_suffix = ['', 'neq']
    wlen_modes = {
        # 'Y': np.array([0.92, 1.15]),
        # 'J': np.array([1.07, 1.4]),
        'H': np.array([1.4, 1.88]),
        'K': np.array([1.88, 2.55]),
        'L': np.array([2.7, 4.25]),
        'M': np.array([3.25, 5.5])
    }

    pressures = np.logspace(-10, 2, 130)
    distances = [213.982 * nc.pc]  # np.logspace(1, 3, 7) * nc.c * 3600 * 24 * 365.25
    # star_apparent_magnitude_v = 12.095

    # Models to be tested
    t_int = [80]
    metallicity = [0, 1]
    co_ratio = [0.55]
    p_cloud = [1e2]  # [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4]
    species_list = ['all', 'H2O', 'CH4', 'NH3', 'CO', 'CO2', 'H2S', 'PH3']

    line_species_list = [
        'CH4_main_iso',
        'CO_all_iso',
        # 'CO_main_iso',
        # 'CO_36',
        'CO2_main_iso',
        'H2O_main_iso',
        'H2S_main_iso',
        # 'HCN_main_iso',
        # 'K',
        # 'Na_allard',
        'NH3_main_iso',
        'PH3_main_iso'
    ]

    molecular_w = {
        'CH4_main_iso': 16,
        'CO_all_iso': 28,
        'CO2_main_iso': 44,
        'H2O_main_iso': 18,
        'H2S_main_iso': 34.1,
        # 'HCN_main_iso': 27,
        'NH3_main_iso': 17,
        'PH3_main_iso': 34
    }

    data = load_hdf5(f"{module_dir}/../../../exorem/outputs/exorem/wasp_96_b_z1.5_t200_co0.55_nocloud_radiusdown.h5")
    mmmr_neq = data['outputs']['layers']['molar_mass'][()][::-1]
    p_neq = data['outputs']['layers']['pressure'][()][::-1] * 1e-5
    vmr_neq = data['outputs']['layers']['volume_mixing_ratios']['absorbers']
    t_neq = data['outputs']['layers']['temperature'][()][::-1]
    mmr_neq = {}

    mmmr_neq_i = interp1d(p_neq, mmmr_neq, fill_value='extrapolate')
    mmmr_neq_i = mmmr_neq_i(pressures)
    t_neq_i = interp1d(p_neq, t_neq, fill_value='extrapolate')
    t_neq_i = t_neq_i(pressures)

    for species in list(vmr_neq.keys()):
        vmr_neq_i = interp1d(p_neq, vmr_neq[species][()][::-1], fill_value='extrapolate')
        vmr_neq_i = vmr_neq_i(pressures)

        for line_species in molecular_w:
            if species + '_' in line_species:
                mmr_neq[line_species] = molecular_w[line_species] * 1e-3 * vmr_neq_i / mmmr_neq_i

    mmr_neq['MMW'] = mmmr_neq_i * 1e3

    mass_fractions = [
        None,
        mmr_neq
    ]

    temperature_profiles = [
        None,
        t_neq_i
    ]

    # Observation parameters
    # Actually matter (used to get the CRIRES SNR data from the ETC website)
    exposure_time = 60  # 4 * 3600
    integration_time = 60
    airmass = 1.2
    velocity_range = [-1400, 1400]
    instrument_resolving_power = 8e4
    # Old (don't do anything anymore)
    telescope_mirror_radius = 8.2e2 / 2  # cm
    telescope_throughput = 0.1
    pixel_sampling = 3

    # Load settings
    settings = load_wavelength_settings(module_dir + '/crires/wavelength_settings.dat')

    # Load planet
    planet = Planet.get(planet_name)
    star_apparent_magnitude_j = planet.system_apparent_magnitude_j
    # star_apparent_magnitude_j = 10
    star_apparent_magnitudes = [star_apparent_magnitude_j]  # np.linspace(4, 16, 7)

    # Load signal to noise_matrix ratios
    star_snr = get_multiple_crires_snr_data(settings, star_apparent_magnitude_j,  # TODO apparent mag is really annoying
                                            planet.star_effective_temperature, exposure_time,
                                            integration_time, airmass,  # TODO add FLI and seeing
                                            rewrite=False, star_apparent_magnitude_band='J')

    # Generate parameter dictionaries
    parameter_dicts = get_parameter_dicts(
        t_int, metallicity, co_ratio, p_cloud
    )

    # Load/generate relevant models
    models = {}
    id_model = 0

    for wlen_mode in wlen_modes:
        print(f"Band {wlen_mode}...")

        # Initialize grid
        models[wlen_mode], all_models_exist = init_model_grid(
            planet_name, lbl_opacity_sampling, do_scat_emis, parameter_dicts, species_list,
            wavelength_boundaries=wlen_modes[wlen_mode],
            model_suffix=model_suffix[id_model]
        )

        if not all_models_exist:
            # Load or generate atmosphere
            atmosphere, atmosphere_filename = SpectralModelLegacy.get_atmosphere_model(
                wlen_bords_micron=wlen_modes[wlen_mode],
                pressures=pressures,
                line_species_list=line_species_list,
                rayleigh_species=SpectralModelLegacy.default_rayleigh_species,
                continuum_opacities=SpectralModelLegacy.default_continuum_opacities,
                lbl_opacity_sampling=lbl_opacity_sampling,
                do_scat_emis=do_scat_emis,
                model_suffix=model_suffix[id_model]
            )

            # Load or generate models
            models[wlen_mode] = generate_model_grid(
                models=models[wlen_mode],
                pressures=pressures,
                line_species_list=line_species_list,
                rayleigh_species='default',
                continuum_opacities='default',
                model_suffix=model_suffix,
                atmosphere=atmosphere,
                mass_fractions=mass_fractions[id_model],
                temperature_profile=temperature_profiles[id_model],
                calculate_transmission_spectrum=True,
                calculate_eclipse_depth=False,
                rewrite=True,
                save=True
            )
        else:
            # Load existing models
            models[wlen_mode] = load_model_grid(models[wlen_mode])

    snrs = {}
    snrs_error = {}
    tsm = {}

    for meta in metallicity:
        snrs[meta] = {}
        snrs_error[meta] = {}
        tsm[meta] = {}

        for band in wlen_modes:
            snrs[meta][band], snrs_error[meta][band], tsm[meta][band], results = get_tsm_snr_pcloud(
                band=band,
                wavelength_boundaries=wlen_modes[band] * 1e-4,
                star_distances=distances,
                p_clouds=p_cloud,
                models=models,
                species_list=species_list,
                settings=settings,
                planet=planet,
                t_int=t_int[0],
                metallicity=meta,
                co_ratio=co_ratio[0],
                velocity_range=velocity_range,
                exposure_time=exposure_time,
                telescope_mirror_radius=telescope_mirror_radius,
                telescope_throughput=telescope_throughput,
                instrument_resolving_power=instrument_resolving_power,
                pixel_sampling=pixel_sampling,
                noise_correction_coefficient=1.0,
                scale_factor=1.0,
                star_snr=star_snr,
                star_apparent_magnitude=star_apparent_magnitudes,
                star_snr_reference_apparent_magnitude=star_apparent_magnitude_j,
                mock_observation_number=100,
                transit_number=1
            )

    for meta in metallicity:
        print(f'\n [Z/H] = {10 ** meta}:')

        for species in species_list:
            if species == 'all':
                continue

            best_snr, best_band, best_setting = find_best_setting(species, snrs[meta])
            print(f"Species '{species}', best setting: {best_band}{best_setting} (CCF SNR: {best_snr})")

    for species in species_list:
        if species == 'all':
            continue

        plt.figure(figsize=(16, 9))
        plot_snr_settings_bars(
            species, snrs,
            model_labels=[rf"Z/H = {10 ** metallicity[0]:.1f} $\times$ solar",
                          rf"Z/H = {10 ** metallicity[1]:.1f} $\times$ solar"],
            planet_name=planet_name,
            threshold=5,
            y_err=snrs_error
        )

        if not os.path.isdir(f"./figures/{planet_name.replace(' ', '_')}"):
            os.mkdir(f"./figures/{planet_name.replace(' ', '_')}")

        plt.savefig(f"./figures/{planet_name.replace(' ', '_')}/{species}_detection.png")

    plot_tsm_pcloud_snr(
        p_cloud, tsm, snrs,
        metallicity=metallicity[0],
        band='J',
        setting=list(settings.keys())[0],
        species=species_list[1],
        planet_name=planet_name,
        exposure_time=exposure_time
    )


def main_lp():
    # Base parameters
    planet_name = 'LP 791-18 c'
    lbl_opacity_sampling = 1
    do_scat_emis = True
    model_suffix = ''
    wlen_modes = {
        'Y': np.array([0.92, 1.15]),
        'J': np.array([1.07, 1.4]),
        'H': np.array([1.4, 1.88]),
        'K': np.array([1.88, 2.55]),
        'L': np.array([2.7, 4.25]),
        'M': np.array([3.25, 5.5])
    }

    # Load planet
    planet = Planet.get(planet_name)

    pressures = np.logspace(-10, 2, 100)
    distances = [0 * nc.pc]  # np.logspace(1, 3, 7) * nc.c * 3600 * 24 * 365.25
    # star_apparent_magnitude_v = 12.095
    star_apparent_magnitude_j = planet.system_apparent_magnitude_j
    # star_apparent_magnitude_j = 10
    star_apparent_magnitudes = [star_apparent_magnitude_j]  # np.linspace(4, 16, 7)

    # Models to be tested
    t_int = [150]
    metallicity = [2]
    co_ratio = [0.55]
    p_cloud = [1e2]  # [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4]
    species_list = ['all', 'H2O', 'CH4']

    line_species_list = [
        'CH4_main_iso',
        'CO_all_iso',
        'CO2_main_iso',
        'H2O_main_iso',
        'H2S_main_iso',
        'HCN_main_iso',
        'NH3_main_iso',
        'PH3_main_iso'
    ]

    # Observation parameters
    # Actually matter (used to get the CRIRES SNR data from the ETC website)
    exposure_time = 3 * 3600  # 4 * 3600
    integration_time = 60
    airmass = 1.4
    velocity_range = [-1400, 1400]
    instrument_resolving_power = 8e4
    # Old (don't do anything anymore)
    telescope_mirror_radius = 8.2e2 / 2  # cm
    telescope_throughput = 0.1
    pixel_sampling = 1

    # Load settings
    settings = load_wavelength_settings(module_dir + '/crires/wavelength_settings.dat')

    # Load region species
    with open(module_dir + '/crires/opa_regions_species.pkl', 'rb') as f:
        regions_species = pickle.load(f)

    # Load signal to noise_matrix ratios
    star_snr = get_multiple_crires_snr_data(settings, star_apparent_magnitude_j,  # TODO apparent mag is really annoying
                                            planet.star_effective_temperature, exposure_time,
                                            integration_time, airmass,  # TODO add FLI and seeing
                                            rewrite=False, star_apparent_magnitude_band='J')

    # Generate parameter dictionaries
    parameter_dicts = get_parameter_dicts(
        t_int, metallicity, co_ratio, p_cloud
    )

    # Load/generate relevant models
    models = {}

    for wlen_mode in wlen_modes:
        print(f"Band {wlen_mode}...")

        # Initialize grid
        models[wlen_mode], all_models_exist = init_model_grid(
            planet_name, lbl_opacity_sampling, do_scat_emis, parameter_dicts, species_list,
            wavelength_boundaries=wlen_modes[wlen_mode],
            model_suffix=model_suffix
        )

        if not all_models_exist:
            # Load or generate atmosphere
            atmosphere, atmosphere_filename = SpectralModelLegacy.get_atmosphere_model(
                wlen_bords_micron=wlen_modes[wlen_mode],
                pressures=pressures,
                line_species_list=SpectralModelLegacy.default_line_species,
                rayleigh_species=SpectralModelLegacy.default_rayleigh_species,
                continuum_opacities=SpectralModelLegacy.default_continuum_opacities,
                lbl_opacity_sampling=lbl_opacity_sampling,
                do_scat_emis=do_scat_emis,
                model_suffix=model_suffix
            )

            # Load or generate models
            models[wlen_mode] = generate_model_grid(
                models=models[wlen_mode],
                pressures=pressures,
                line_species_list=line_species_list,
                rayleigh_species='default',
                continuum_opacities='default',
                model_suffix=model_suffix,
                atmosphere=atmosphere,
                calculate_transmission_spectrum=True,
                rewrite=True,
                save=True
            )
        else:
            # Load existing models
            models[wlen_mode] = load_model_grid(models[wlen_mode])

    snrs = {}
    snrs_error = {}
    tsm = {}
    results = {}

    for trans in [3]:
        planet_key_ = f' {trans} transits'
        snrs[planet_key_] = {}
        snrs_error[planet_key_] = {}
        tsm[planet_key_] = {}
        results[planet_key_] = {}

        for meta in metallicity:
            snrs[planet_key_][meta] = {}
            snrs_error[planet_key_][meta] = {}
            tsm[planet_key_][meta] = {}
            results[planet_key_][meta] = {}

            for band in wlen_modes:
                snrs[planet_key_][meta][band], snrs_error[planet_key_][meta][band], tsm[planet_key_][meta][band], \
                    results[planet_key_][meta][band] = get_tsm_snr_pcloud(
                    band=band,
                    wavelength_boundaries=wlen_modes[band] * 1e-4,
                    star_distances=distances,
                    p_clouds=p_cloud,
                    models=models,
                    species_list=species_list,
                    settings=settings,
                    planet=planet,
                    t_int=t_int[0],
                    metallicity=meta,
                    co_ratio=co_ratio[0],
                    velocity_range=velocity_range,
                    exposure_time=exposure_time,
                    telescope_mirror_radius=telescope_mirror_radius,
                    telescope_throughput=telescope_throughput,
                    instrument_resolving_power=instrument_resolving_power,
                    pixel_sampling=pixel_sampling,
                    noise_correction_coefficient=1.0,
                    scale_factor=1.0,
                    star_snr=star_snr,
                    star_apparent_magnitude=star_apparent_magnitudes,
                    star_snr_reference_apparent_magnitude=star_apparent_magnitude_j,
                    mock_observation_number=100,
                    mode='transit',
                    transit_number=trans,
                    regions_species=regions_species
                )

    for planet_key in snrs:
        for meta in metallicity:
            print(f'\n [Z/H] = {10 ** meta}:')

            for species in species_list:
                if species == 'all':
                    continue

                best_snr, best_band, best_setting = find_best_setting(species, snrs[planet_key][meta])
                print(f"Species '{species}', best setting: {best_band}{best_setting} (CCF SNR: {best_snr})")

        for species in species_list:
            if species == 'all':
                continue

            plt.figure(figsize=(16, 9))
            plot_snr_settings_bars(
                species, snrs[planet_key_],
                model_labels=[rf"Z/H = {10 ** metallicity[0]:.1f} $\times$ solar",],
                              # rf"Z/H = {10 ** metallicity[1]:.1f} $\times$ solar"],
                planet_name=planet_name,
                threshold=5,
                y_err=snrs_error[planet_key_]
            )

            if not os.path.isdir(f'{module_dir}/figures'):
                os.mkdir(f'{module_dir}/figures')

            if not os.path.isdir(f"{module_dir}/figures/{planet_name.replace(' ', '_')}"):
                os.mkdir(f"{module_dir}/figures/{planet_name.replace(' ', '_')}")

            plt.savefig(f"{module_dir}/figures/{planet_name.replace(' ', '_')}/{species}_detection_{planet_key_.replace(' ', '_')}.png")


def main_ltt2():
    # Base parameters
    planet_name = 'LTT 3780 c'
    lbl_opacity_sampling = 1
    do_scat_emis = True
    model_suffix = ''
    wlen_modes = {
        # 'Y': np.array([0.92, 1.15]),
        # 'J': np.array([1.07, 1.4]),
        'H': np.array([1.4, 1.88]),
        'K': np.array([1.88, 2.55]),
        # 'L': np.array([2.7, 4.25]),
        # 'M': np.array([3.25, 5.5])
    }

    pressures = np.logspace(-10, 2, 100)
    distances = [0 * nc.pc]  # np.logspace(1, 3, 7) * nc.c * 3600 * 24 * 365.25
    # star_apparent_magnitude_v = 12.095

    # Load planet
    planet = Planet.get(planet_name)

    star_apparent_magnitude_j = planet.system_apparent_magnitude_j
    # star_apparent_magnitude_j = 10
    star_apparent_magnitudes = [star_apparent_magnitude_j]  # np.linspace(4, 16, 7)

    # Models to be tested
    t_int = [50]
    metallicity = [2]
    co_ratio = [0.55]
    p_cloud = [1e2]  # [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4]
    species_list = ['all', 'H2O', 'CH4']

    line_species_list = [
        'CH4_main_iso',
        'CO_all_iso',
        'CO2_main_iso',
        'H2O_main_iso',
        'H2S_main_iso',
        'HCN_main_iso',
        'NH3_main_iso',
        'PH3_main_iso'
    ]

    # Observation parameters
    # Actually matter (used to get the CRIRES SNR data from the ETC website)
    exposure_time = 4 * 3600  # 4 * 3600
    integration_time = 60
    airmass = 1.33
    velocity_range = [-1400, 1400]
    instrument_resolving_power = 8e4
    # Old (don't do anything anymore)
    telescope_mirror_radius = 8.2e2 / 2  # cm
    telescope_throughput = 0.1
    pixel_sampling = 1

    # Load settings
    settings = load_wavelength_settings(module_dir + '/crires/wavelength_settings.dat')

    # Load region species
    with open(module_dir + '/crires/opa_regions_species.pkl', 'rb') as f:
        regions_species = pickle.load(f)

    # Load signal to noise_matrix ratios
    star_snr = get_multiple_crires_snr_data(settings, star_apparent_magnitude_j,  # TODO apparent mag is really annoying
                                            planet.star_effective_temperature, exposure_time,
                                            integration_time, airmass,  # TODO add FLI and seeing
                                            rewrite=False, star_apparent_magnitude_band='J')

    # Generate parameter dictionaries
    parameter_dicts = get_parameter_dicts(
        t_int, metallicity, co_ratio, p_cloud
    )

    # Load/generate relevant models
    models = {}

    for wlen_mode in wlen_modes:
        print(f"Band {wlen_mode}...")

        # Initialize grid
        models[wlen_mode], all_models_exist = init_model_grid(
            planet_name, lbl_opacity_sampling, do_scat_emis, parameter_dicts, species_list,
            wavelength_boundaries=wlen_modes[wlen_mode],
            model_suffix=model_suffix
        )

        if not all_models_exist:
            # Load or generate atmosphere
            atmosphere, atmosphere_filename = SpectralModelLegacy.get_atmosphere_model(
                wlen_bords_micron=wlen_modes[wlen_mode],
                pressures=pressures,
                line_species_list=SpectralModelLegacy.default_line_species,
                rayleigh_species=SpectralModelLegacy.default_rayleigh_species,
                continuum_opacities=SpectralModelLegacy.default_continuum_opacities,
                lbl_opacity_sampling=lbl_opacity_sampling,
                do_scat_emis=do_scat_emis,
                model_suffix=model_suffix
            )

            # Load or generate models
            models[wlen_mode] = generate_model_grid(
                models=models[wlen_mode],
                pressures=pressures,
                line_species_list=line_species_list,
                rayleigh_species='default',
                continuum_opacities='default',
                model_suffix=model_suffix,
                atmosphere=atmosphere,
                calculate_transmission_spectrum=True,
                rewrite=False,
                save=True
            )
        else:
            # Load existing models
            models[wlen_mode] = load_model_grid(models[wlen_mode])

    snrs = {}
    snrs_error = {}
    tsm = {}
    results = {}

    for trans in [1, 4]:
        planet_key_ = f' {trans} transits'
        snrs[planet_key_] = {}
        snrs_error[planet_key_] = {}
        tsm[planet_key_] = {}
        results[planet_key_] = {}

        for meta in metallicity:
            snrs[planet_key_][meta] = {}
            snrs_error[planet_key_][meta] = {}
            tsm[planet_key_][meta] = {}
            results[planet_key_][meta] = {}

            for band in wlen_modes:
                snrs[planet_key_][meta][band], snrs_error[planet_key_][meta][band], tsm[planet_key_][meta][band], \
                    results[planet_key_][meta][band] = get_tsm_snr_pcloud(
                    band=band,
                    wavelength_boundaries=wlen_modes[band] * 1e-4,
                    star_distances=distances,
                    p_clouds=p_cloud,
                    models=models,
                    species_list=species_list,
                    settings=settings,
                    planet=planet,
                    t_int=t_int[0],
                    metallicity=meta,
                    co_ratio=co_ratio[0],
                    velocity_range=velocity_range,
                    exposure_time=exposure_time,
                    telescope_mirror_radius=telescope_mirror_radius,
                    telescope_throughput=telescope_throughput,
                    instrument_resolving_power=instrument_resolving_power,
                    pixel_sampling=pixel_sampling,
                    noise_correction_coefficient=1.0,
                    scale_factor=1.0,
                    star_snr=star_snr,
                    star_apparent_magnitude=star_apparent_magnitudes,
                    star_snr_reference_apparent_magnitude=star_apparent_magnitude_j,
                    mock_observation_number=100,
                    mode='transit',
                    transit_number=trans,
                    regions_species=regions_species
                )

    for species in species_list:
        if species == 'all':
            continue
        snrs_ = {'1 transit': snrs[f' {1} transits'][2], '4 transit': snrs[f' {4} transits'][2]}
        snrs_error_ = {'1 transit': snrs_error[f' {1} transits'][2], '4 transit': snrs_error[f' {4} transits'][2]}

        plt.figure(figsize=(16, 9))
        plot_snr_settings_bars(
            species, snrs_,
            model_labels=[rf"1 transit", "4 transits"],
            # rf"Z/H = {10 ** metallicity[1]:.1f} $\times$ solar"],
            planet_name=planet_name,
            threshold=5,
            y_err=snrs_error_
        )

        if not os.path.isdir(f'{module_dir}/figures'):
            os.mkdir(f'{module_dir}/figures')

        if not os.path.isdir(f"{module_dir}/figures/{planet_name.replace(' ', '_')}"):
            os.mkdir(f"{module_dir}/figures/{planet_name.replace(' ', '_')}")

        plt.savefig(
            f"{module_dir}/figures/{planet_name.replace(' ', '_')}/{species}_detection.png")

    for planet_key in snrs:
        for meta in metallicity:
            print(f'\n [Z/H] = {10 ** meta}:')

            for species in species_list:
                if species == 'all':
                    continue

                best_snr, best_band, best_setting = find_best_setting(species, snrs[planet_key][meta])
                print(f"Species '{species}', best setting: {best_band}{best_setting} (CCF SNR: {best_snr})")

        for species in species_list:
            if species == 'all':
                continue

            plt.figure(figsize=(16, 9))
            plot_snr_settings_bars(
                species, snrs[planet_key_],
                model_labels=[rf"Z/H = {10 ** metallicity[0]:.1f} $\times$ solar",],
                              # rf"Z/H = {10 ** metallicity[1]:.1f} $\times$ solar"],
                planet_name=planet_name,
                threshold=5,
                y_err=snrs_error[planet_key_]
            )

            if not os.path.isdir(f'{module_dir}/figures'):
                os.mkdir(f'{module_dir}/figures')

            if not os.path.isdir(f"{module_dir}/figures/{planet_name.replace(' ', '_')}"):
                os.mkdir(f"{module_dir}/figures/{planet_name.replace(' ', '_')}")

            plt.savefig(f"{module_dir}/figures/{planet_name.replace(' ', '_')}/{species}_detection_{planet_key_.replace(' ', '_')}.png")


def main_toi270():
    # Base parameters
    planet_name = 'TOI-270 c'
    lbl_opacity_sampling = 1
    do_scat_emis = True
    model_suffix = ''
    wlen_modes = {
        # 'Y': np.array([0.92, 1.15]),
        # 'J': np.array([1.07, 1.4]),
        'H': np.array([1.4, 1.88]),
        'K': np.array([1.88, 2.55]),
        # 'L': np.array([2.7, 4.25]),
        # 'M': np.array([3.25, 5.5])
    }

    pressures = np.logspace(-10, 2, 100)
    distances = [0 * nc.pc]  # np.logspace(1, 3, 7) * nc.c * 3600 * 24 * 365.25
    # star_apparent_magnitude_v = 12.095

    # Load planet
    planet = Planet.get(planet_name)

    star_apparent_magnitude_j = planet.system_apparent_magnitude_j
    # star_apparent_magnitude_j = 10
    star_apparent_magnitudes = [star_apparent_magnitude_j]  # np.linspace(4, 16, 7)

    # Models to be tested
    t_int = [50]
    metallicity = [2]
    co_ratio = [0.55]
    p_cloud = [1e2]  # [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4]
    species_list = ['all', 'H2O', 'CH4']

    line_species_list = [
        'CH4_main_iso',
        'CO_all_iso',
        'CO2_main_iso',
        'H2O_main_iso',
        'H2S_main_iso',
        'HCN_main_iso',
        'NH3_main_iso',
        'PH3_main_iso'
    ]

    # Observation parameters
    # Actually matter (used to get the CRIRES SNR data from the ETC website)
    exposure_time = 60 #4 * 3600  # 4 * 3600
    integration_time = 60
    airmass = 1.33
    velocity_range = [-1400, 1400]
    instrument_resolving_power = 8e4
    # Old (don't do anything anymore)
    telescope_mirror_radius = 8.2e2 / 2  # cm
    telescope_throughput = 0.1
    pixel_sampling = 1

    # Load settings
    settings = load_wavelength_settings(module_dir + '/crires/wavelength_settings.dat')

    # Load region species
    with open(module_dir + '/crires/opa_regions_species.pkl', 'rb') as f:
        regions_species = pickle.load(f)

    # Load signal to noise_matrix ratios
    star_snr = get_multiple_crires_snr_data(settings, star_apparent_magnitude_j,  # TODO apparent mag is really annoying
                                            planet.star_effective_temperature, exposure_time,
                                            integration_time, airmass,  # TODO add FLI and seeing
                                            rewrite=False, star_apparent_magnitude_band='J')

    # Generate parameter dictionaries
    parameter_dicts = get_parameter_dicts(
        t_int, metallicity, co_ratio, p_cloud
    )

    # Load/generate relevant models
    models = {}

    for wlen_mode in wlen_modes:
        print(f"Band {wlen_mode}...")

        # Initialize grid
        models[wlen_mode], all_models_exist = init_model_grid(
            planet_name, lbl_opacity_sampling, do_scat_emis, parameter_dicts, species_list,
            wavelength_boundaries=wlen_modes[wlen_mode],
            model_suffix=model_suffix
        )

        if not all_models_exist:
            # Load or generate atmosphere
            atmosphere, atmosphere_filename = SpectralModelLegacy.get_atmosphere_model(
                wlen_bords_micron=wlen_modes[wlen_mode],
                pressures=pressures,
                line_species_list=SpectralModelLegacy.default_line_species,
                rayleigh_species=SpectralModelLegacy.default_rayleigh_species,
                continuum_opacities=SpectralModelLegacy.default_continuum_opacities,
                lbl_opacity_sampling=lbl_opacity_sampling,
                do_scat_emis=do_scat_emis,
                model_suffix=model_suffix
            )

            # Load or generate models
            models[wlen_mode] = generate_model_grid(
                models=models[wlen_mode],
                pressures=pressures,
                line_species_list=line_species_list,
                rayleigh_species='default',
                continuum_opacities='default',
                model_suffix=model_suffix,
                atmosphere=atmosphere,
                calculate_transmission_spectrum=True,
                rewrite=False,
                save=True
            )
        else:
            # Load existing models
            models[wlen_mode] = load_model_grid(models[wlen_mode])

    snrs = {}
    snrs_error = {}
    tsm = {}
    results = {}

    for trans in [1, 4, 6]:
        planet_key_ = f' {trans} transits'
        snrs[planet_key_] = {}
        snrs_error[planet_key_] = {}
        tsm[planet_key_] = {}
        results[planet_key_] = {}

        for meta in metallicity:
            snrs[planet_key_][meta] = {}
            snrs_error[planet_key_][meta] = {}
            tsm[planet_key_][meta] = {}
            results[planet_key_][meta] = {}

            for band in wlen_modes:
                snrs[planet_key_][meta][band], snrs_error[planet_key_][meta][band], tsm[planet_key_][meta][band], \
                    results[planet_key_][meta][band] = get_tsm_snr_pcloud(
                    band=band,
                    wavelength_boundaries=wlen_modes[band] * 1e-4,
                    star_distances=distances,
                    p_clouds=p_cloud,
                    models=models,
                    species_list=species_list,
                    settings=settings,
                    planet=planet,
                    t_int=t_int[0],
                    metallicity=meta,
                    co_ratio=co_ratio[0],
                    velocity_range=velocity_range,
                    exposure_time=exposure_time,
                    telescope_mirror_radius=telescope_mirror_radius,
                    telescope_throughput=telescope_throughput,
                    instrument_resolving_power=instrument_resolving_power,
                    pixel_sampling=pixel_sampling,
                    noise_correction_coefficient=1.0,
                    scale_factor=1.0,
                    star_snr=star_snr,
                    star_apparent_magnitude=star_apparent_magnitudes,
                    star_snr_reference_apparent_magnitude=star_apparent_magnitude_j,
                    mock_observation_number=100,
                    mode='transit',
                    transit_number=trans,
                    regions_species=regions_species
                )

    for species in species_list:
        if species == 'all':
            continue
        snrs_ = {'1 transit': snrs[f' {1} transits'][2], '4 transit': snrs[f' {4} transits'][2], '6 transit': snrs[f' {6} transits'][2]}
        snrs_error_ = {'1 transit': snrs_error[f' {1} transits'][2], '4 transit': snrs_error[f' {4} transits'][2], '6 transit': snrs_error[f' {6} transits'][2]}

        plt.figure(figsize=(16, 9))
        plot_snr_settings_bars(
            species, snrs_,
            model_labels=[rf"1 transit", "4 transits", "6 transits"],
            # rf"Z/H = {10 ** metallicity[1]:.1f} $\times$ solar"],
            planet_name=planet_name,
            threshold=5,
            y_err=snrs_error_
        )

        if not os.path.isdir(f'{module_dir}/figures'):
            os.mkdir(f'{module_dir}/figures')

        if not os.path.isdir(f"{module_dir}/figures/{planet_name.replace(' ', '_')}"):
            os.mkdir(f"{module_dir}/figures/{planet_name.replace(' ', '_')}")

        plt.savefig(
            f"{module_dir}/figures/{planet_name.replace(' ', '_')}/{species}_detection.png")

    for planet_key in snrs:
        for meta in metallicity:
            print(f'\n [Z/H] = {10 ** meta}:')

            for species in species_list:
                if species == 'all':
                    continue

                best_snr, best_band, best_setting = find_best_setting(species, snrs[planet_key][meta])
                print(f"Species '{species}', best setting: {best_band}{best_setting} (CCF SNR: {best_snr})")

        for species in species_list:
            if species == 'all':
                continue

            plt.figure(figsize=(16, 9))
            plot_snr_settings_bars(
                species, snrs[planet_key_],
                model_labels=[rf"Z/H = {10 ** metallicity[0]:.1f} $\times$ solar",],
                              # rf"Z/H = {10 ** metallicity[1]:.1f} $\times$ solar"],
                planet_name=planet_name,
                threshold=5,
                y_err=snrs_error[planet_key_]
            )

            if not os.path.isdir(f'{module_dir}/figures'):
                os.mkdir(f'{module_dir}/figures')

            if not os.path.isdir(f"{module_dir}/figures/{planet_name.replace(' ', '_')}"):
                os.mkdir(f"{module_dir}/figures/{planet_name.replace(' ', '_')}")

            plt.savefig(f"{module_dir}/figures/{planet_name.replace(' ', '_')}/{species}_detection_{planet_key_.replace(' ', '_')}.png")


def main_ltt():
    # Base parameters
    planet_name = 'LTT 3780 c'
    lbl_opacity_sampling = 1
    do_scat_emis = False
    model_suffix = ['', 'neq']
    wlen_modes = {
        'Y': np.array([0.92, 1.15]),
        'J': np.array([1.07, 1.4]),
        'H': np.array([1.4, 1.88]),
        'K': np.array([1.88, 2.55]),
        'L': np.array([2.7, 4.25]),
        'M': np.array([3.25, 5.5])
    }

    # Load planet
    planet_1 = Planet.get(planet_name)

    pressures = np.logspace(-10, 2, 130)
    distances = [213.982 * nc.pc]  # np.logspace(1, 3, 7) * nc.c * 3600 * 24 * 365.25
    # star_apparent_magnitude_v = 12.095
    star_apparent_magnitude_j = planet_1.system_apparent_magnitude_j
    # star_apparent_magnitude_j = 10
    star_apparent_magnitudes = [star_apparent_magnitude_j]
    mock_observations_number = 100

    # Models to be tested
    t_int = [50]
    metallicity = [1, 2]
    co_ratio = [0.55]
    p_cloud = [1e2]
    species_list = ['all', 'H2O', 'CO', 'CO2', 'H2S', 'NH3', 'CH4']

    line_species_list = [
        'CH4_main_iso',
        'CO_all_iso',
        'CO2_main_iso',
        'H2O_main_iso',
        'H2S_main_iso',
        'HCN_main_iso',
        'K',
        'Na_allard',
        'NH3_main_iso',
        'PH3_main_iso'
    ]

    molecular_w = {
        'CH4_main_iso': 16,
        'CO_all_iso': 28,
        'CO2_main_iso': 44,
        'H2O_main_iso': 18,
        'H2S_main_iso': 34.1,
        'HCN_main_iso': 27,
        'NH3_main_iso': 17,
        'PH3_main_iso': 34
    }

    data = load_hdf5(f"{module_dir}/../../../exorem/outputs/exorem/ltt-3780_c_z100_g10_t50_co0.55_nocloud.h5")
    mmmr_neq = data['outputs']['layers']['molar_mass'][()][::-1]
    p_neq = data['outputs']['layers']['pressure'][()][::-1] * 1e-5
    vmr_neq = data['outputs']['layers']['volume_mixing_ratios']['absorbers']
    t_neq = data['outputs']['layers']['temperature'][()][::-1]
    mmr_neq = {}

    mmmr_neq_i = interp1d(p_neq, mmmr_neq, fill_value='extrapolate')
    mmmr_neq_i = mmmr_neq_i(pressures)
    t_neq_i = interp1d(p_neq, t_neq, fill_value='extrapolate')
    t_neq_i = t_neq_i(pressures)

    for species in list(vmr_neq.keys()):
        vmr_neq_i = interp1d(p_neq, vmr_neq[species][()][::-1], fill_value='extrapolate')
        vmr_neq_i = vmr_neq_i(pressures)

        for line_species in molecular_w:
            if species + '_' in line_species:
                mmr_neq[line_species] = molecular_w[line_species] * 1e-3 * vmr_neq_i / mmmr_neq_i

    mmr_neq['MMW'] = mmmr_neq_i * 1e3

    mass_fractions = [
        None,
        mmr_neq
    ]

    temperature_profiles = [
        None,
        t_neq_i
    ]

    # Observation parameters
    # Actually matter (used to get the CRIRES SNR data from the ETC website)

    exposure_time = 6 * 3600  # 4 * 3600
    integration_time = 60
    airmass = 1.2
    velocity_range = [-1400, 1400]
    instrument_resolving_power = 8e4
    # Old (don't do anything anymore)
    telescope_mirror_radius = 8.2e2 / 2  # cm
    telescope_throughput = 0.1
    pixel_sampling = 3

    # Load settings
    settings = load_wavelength_settings(module_dir + '/crires/wavelength_settings.dat')

    # Load region species
    with open(module_dir + '/crires/opa_regions_species.pkl', 'rb') as f:
        regions_species = pickle.load(f)

    planet_2 = copy.copy(planet_1)
    planets = [planet_1, planet_2]
    planet_keys = ['eq', 'neq']

    # Load signal to noise_matrix ratios
    star_snr = get_multiple_crires_snr_data(settings, star_apparent_magnitude_j,  # TODO apparent mag is really annoying
                                            planet_1.star_effective_temperature, exposure_time,
                                            integration_time, airmass,  # TODO add FLI and seeing
                                            rewrite=False, star_apparent_magnitude_band='J')

    # Generate parameter dictionaries
    parameter_dicts = get_parameter_dicts(
        t_int, metallicity, co_ratio, p_cloud
    )

    # Load/generate relevant models
    models = {}

    for wlen_mode in wlen_modes:
        print(f"Band {wlen_mode}...")

        atmosphere = None

        for i, planet in enumerate(planets):
            planet_key = planet_keys[i]

            if planet_key not in models:
                models[planet_key] = {}

            # Initialize grid
            models[planet_key][wlen_mode], all_models_exist = init_model_grid(
                planet.name, lbl_opacity_sampling, do_scat_emis, parameter_dicts, species_list,
                wavelength_boundaries=wlen_modes[wlen_mode],
                model_suffix=model_suffix[i]
            )

            if not all_models_exist:
                if atmosphere is None:
                    # Load or generate atmosphere
                    atmosphere, atmosphere_filename = SpectralModelLegacy.get_atmosphere_model(
                        wlen_bords_micron=wlen_modes[wlen_mode],
                        pressures=pressures,
                        line_species_list=line_species_list,
                        rayleigh_species=SpectralModelLegacy.default_rayleigh_species,
                        continuum_opacities=SpectralModelLegacy.default_continuum_opacities,
                        lbl_opacity_sampling=lbl_opacity_sampling,
                        do_scat_emis=do_scat_emis,
                        model_suffix=model_suffix[0]
                    )

                # Load or generate models
                models[planet_key][wlen_mode] = generate_model_grid(
                    models=models[planet_key][wlen_mode],
                    pressures=pressures,
                    line_species_list=line_species_list,
                    rayleigh_species='default',
                    continuum_opacities='default',
                    model_suffix=model_suffix[i],
                    atmosphere=atmosphere,
                    mass_fractions=mass_fractions[i],
                    temperature_profile=temperature_profiles[i],
                    calculate_transmission_spectrum=True,
                    rewrite=False,
                    save=True
                )
            else:
                # Load existing models
                models[planet_key][wlen_mode] = load_model_grid(models[planet_key][wlen_mode])

    snrs = {}
    snrs_error = {}
    results = {}
    tsm = 0

    for trans in [1]:
        for i, planet in enumerate(planets):
            planet_key = planet_keys[i]
            planet_key_ = planet_keys[i] + f' {trans} transits'
            snrs[planet_key_] = {}
            snrs_error[planet_key_] = {}
            results[planet_key_] = {}

            for meta in metallicity:
                snrs[planet_key_][meta] = {}
                snrs_error[planet_key_][meta] = {}
                results[planet_key_][meta] = {}

                for band in wlen_modes:
                    snrs[planet_key_][meta][band], snrs_error[planet_key_][meta][band], tsm, \
                        results[planet_key_][meta][band] = \
                        get_tsm_snr_pcloud(
                            band=band,
                            wavelength_boundaries=wlen_modes[band] * 1e-4,
                            star_distances=distances,
                            p_clouds=p_cloud,
                            models=models[planet_key],
                            species_list=species_list,
                            settings=settings,
                            planet=planet,
                            t_int=t_int[0],
                            metallicity=meta,
                            co_ratio=co_ratio[0],
                            velocity_range=velocity_range,
                            exposure_time=exposure_time,
                            telescope_mirror_radius=telescope_mirror_radius,
                            telescope_throughput=telescope_throughput,
                            instrument_resolving_power=instrument_resolving_power,
                            pixel_sampling=pixel_sampling,
                            noise_correction_coefficient=1.0,
                            scale_factor=1.0,
                            star_snr=star_snr,
                            star_apparent_magnitude=star_apparent_magnitudes,
                            star_snr_reference_apparent_magnitude=star_apparent_magnitude_j,
                            mock_observation_number=100,
                            transit_number=trans,
                            regions_species=regions_species
                        )

    with open(f'{planet_name}_ccf.pkl', 'wb') as f:
        pickle.dump([snrs, snrs_error, results, tsm], f)

    snr = {}
    snr_err = {}

    for key in snrs:
        if '2' in key:
            continue

        snr[key] = snrs[key][1]
        snr_err[key] = snrs_error[key][1]

    for species in species_list:
        if species == 'all':
            continue

        plt.figure(figsize=(16, 9))
        plot_snr_settings_bars(
            species, snr,
            planet_name=planets[0].name,
            threshold=5,
            y_err=snr_err
        )

        plt.savefig(f"./figures/{planet_name.replace(' ', '_')}/{species}_detection_"
                    f"meta{metallicity[0]}.png")
        plt.close("all")

    snr_x = {}
    velocity = np.linspace(velocity_range[0] + 100, velocity_range[1] - 100, 5000)
    ccf_size = np.size(velocity)
    best_setting = {}
    max_snr = {}

    for model in snrs:
        print(f"\nModel {model}")
        snr_x[model] = {}
        best_setting[model] = {}
        max_snr[model] = {}

        for meta in snrs[model]:
            print(f"[Z/H] = {meta}")
            snr_x[model][meta] = {}
            best_setting[model][meta] = {}
            max_snr[model][meta] = {}

            for species in species_list:
                if species == 'all':
                    continue

                print(f'Species {species}')
                snr_x[model][meta][species] = np.zeros(mock_observations_number)
                best_setting[model][meta][species] = {}
                max_snr[model][meta][species] = {}
                ccf = np.zeros((mock_observations_number, ccf_size))

                for band in snrs[model][meta]:
                    model_key = list(results[model][meta][band].keys())[1]

                    best_setting[model][meta][species][band] = None
                    max_snr[model][meta][species][band] = -np.inf
                    ccf_max = 0
                    vel_max = 0

                    for setting in snrs[model][meta][band]:
                        if snrs[model][meta][band][setting][species] > max_snr[model][meta][species][band]:
                            max_snr[model][meta][species][band] = snrs[model][meta][band][setting][species][0, 0]
                            ccf_max = results[model][meta][band][model_key][band][setting][species]['CCF']
                            vel_max = results[model][meta][band][model_key][band][setting][species]['velocity']
                            best_setting[model][meta][species][band] = setting

                    print(f"Band {band}: "
                          f"best setting {best_setting[model][meta][species][band] } "
                          f"(SNR = {max_snr[model][meta][species][band] })")
                    if max_snr[model][meta][species][band] <= 0:
                        #print(f"Bad SNR, skipping band {band}")

                        continue

                    ccf_max = interp1d(vel_max, np.transpose(ccf_max))
                    ccf += ccf_max(velocity)

                for i in range(mock_observations_number):
                    snr_x[model][meta][species][i], mu, std = calculate_ccf_snr(velocity, ccf[i, :])

    with open(f'{planet_name}_sum_band_cff.pkl', 'wb') as f:
        pickle.dump({'Sum SNR bands best setting (100 takes)': snr_x, 'Best setting': best_setting}, f)

    for i in range(len(planets)):
        for species in species_list:
            if species == 'all':
                continue

            plt.figure(figsize=(16, 9))
            plot_snr_settings_bars_from_gaussian(
                species, snrs[planets[i].equilibrium_temperature],
                planet_name=planets[i].name,
                threshold=5
            )

            plt.savefig(f"./figures/{planet_name.replace(' ', '_')}/{species}_detection_"
                        f"Teq{planets[i].equilibrium_temperature}K_tobs{exposure_time/3600}h_"
                        f"{len(snrs[planets[i].equilibrium_temperature])}takes.png")

    plot_tsm_pcloud_snr(
        p_cloud, tsm, snrs,
        metallicity=metallicity[2],
        band='J',
        setting=list(settings.keys())[0],
        species=species_list[1],
        planet_name=planet_name,
        exposure_time=exposure_time
    )


def main_aumicc():
    # Base parameters
    planet_name = 'AU Mic c'
    lbl_opacity_sampling = 1
    do_scat_emis = False
    model_suffix = ''
    wlen_modes = {
        'Y': np.array([0.92, 1.15]),
        'J': np.array([1.07, 1.4]),
        'H': np.array([1.4, 1.88]),
        'K': np.array([1.88, 2.55]),
        'L': np.array([2.7, 4.25]),
        'M': np.array([3.25, 5.5])
    }

    # Load planet
    planet = Planet.get(planet_name)

    # Parameters
    pressures = np.logspace(-10, 2, 130)
    distances = [213.982 * nc.pc]  # np.logspace(1, 3, 7) * nc.c * 3600 * 24 * 365.25
    # star_apparent_magnitude_v = 12.095
    star_apparent_magnitude_j = planet.system_apparent_magnitude_j
    # star_apparent_magnitude_j = 10
    star_apparent_magnitudes = [star_apparent_magnitude_j]

    # Models to be tested
    t_int = [200, 1500]
    metallicity = [1, 2]
    co_ratio = [0.55]
    p_cloud = [1e2]
    species_list = ['all', 'H2O', 'CO', 'CO2', 'H2S', 'NH3', 'PH3', 'HCN', 'CH4']

    line_species_list = [
        'CH4_main_iso',
        'CO_all_iso',
        'CO2_main_iso',
        'H2O_main_iso',
        'H2S_main_iso',
        'HCN_main_iso',
        'K',
        'Na_allard',
        'NH3_main_iso',
        'PH3_main_iso'
    ]

    molecular_w = {
        'CH4_main_iso': 16,
        'CO_all_iso': 28,
        'CO2_main_iso': 44,
        'H2O_main_iso': 18,
        'H2S_main_iso': 34.1,
        'HCN_main_iso': 27,
        'NH3_main_iso': 17,
        'PH3_main_iso': 34
    }

    data = load_hdf5(f"{module_dir}/../../../exorem/outputs/exorem/hd_136352_c_z100_g10_t50_co0.55_nocloud.h5")
    mmmr_neq = data['outputs']['layers']['molar_mass'][()][::-1]
    p_neq = data['outputs']['layers']['pressure'][()][::-1] * 1e-5
    vmr_neq = data['outputs']['layers']['volume_mixing_ratios']['absorbers']
    mmr_neq = {}

    mmmr_neq_i = interp1d(p_neq, mmmr_neq, fill_value='extrapolate')
    mmmr_neq_i = mmmr_neq_i(pressures)

    for species in list(vmr_neq.keys()):
        vmr_neq_i = interp1d(p_neq, vmr_neq[species][()][::-1], fill_value='extrapolate')
        vmr_neq_i = vmr_neq_i(pressures)

        for line_species in molecular_w:
            if species + '_' in line_species:
                mmr_neq[line_species] = molecular_w[line_species] * 1e-3 * vmr_neq_i / mmmr_neq_i

    # Observation parameters
    # Actually matter (used to get the CRIRES SNR data from the ETC website)
    exposure_time = 6 * 3600
    integration_time = 60
    airmass = 1.2
    velocity_range = [-1400, 1400]
    instrument_resolving_power = 8e4
    # Old (don't do anything anymore)
    telescope_mirror_radius = 8.2e2 / 2  # cm
    telescope_throughput = 0.1
    pixel_sampling = 3

    # Load settings
    settings = load_wavelength_settings(module_dir + '/crires/wavelength_settings.dat')

    # Load signal to noise_matrix ratios
    star_snr = get_multiple_crires_snr_data(settings, star_apparent_magnitude_j,  # TODO apparent mag is really annoying
                                            planet.star_effective_temperature, exposure_time,
                                            integration_time, airmass,  # TODO add FLI and seeing
                                            rewrite=False, star_apparent_magnitude_band='J')

    # Generate parameter dictionaries
    parameter_dicts = get_parameter_dicts(
        t_int, metallicity, co_ratio, p_cloud
    )

    # Load/generate relevant models
    models = {}

    for wlen_mode in wlen_modes:
        print(f"Band {wlen_mode}...")

        # Initialize grid
        models[wlen_mode], all_models_exist = init_model_grid(
            planet_name, lbl_opacity_sampling, do_scat_emis, parameter_dicts, species_list,
            wavelength_boundaries=wlen_modes[wlen_mode],
            model_suffix=model_suffix
        )

        if not all_models_exist:
            # Load or generate atmosphere
            atmosphere, atmosphere_filename = SpectralModelLegacy.get_atmosphere_model(
                wlen_bords_micron=wlen_modes[wlen_mode],
                pressures=pressures,
                line_species_list=SpectralModelLegacy.default_line_species,
                rayleigh_species=SpectralModelLegacy.default_rayleigh_species,
                continuum_opacities=SpectralModelLegacy.default_continuum_opacities,
                lbl_opacity_sampling=lbl_opacity_sampling,
                do_scat_emis=do_scat_emis,
                model_suffix=model_suffix
            )

            # Load or generate models
            models[wlen_mode] = generate_model_grid(
                models=models[wlen_mode],
                pressures=pressures,
                line_species_list=line_species_list,
                rayleigh_species='default',
                continuum_opacities='default',
                model_suffix=model_suffix,
                atmosphere=atmosphere,
                calculate_transmission_spectrum=True,
                rewrite=False,
                save=True
            )
        else:
            # Load existing models
            models[wlen_mode] = load_model_grid(models[wlen_mode])

    snrs = {}
    snrs_error = {}
    results = {}
    tsm = 0

    for meta in metallicity:
        snrs[meta] = {}
        snrs_error[meta] = {}
        results[meta] = {}

        for band in wlen_modes:
            snrs[meta][band], snrs_error[meta][band], tsm, results[meta][band] = get_tsm_snr_pcloud(
                band=band,
                wavelength_boundaries=wlen_modes[band] * 1e-4,
                star_distances=distances,
                p_clouds=p_cloud,
                models=models,
                species_list=species_list,
                settings=settings,
                planet=planet,
                t_int=t_int[0],
                metallicity=meta,
                co_ratio=co_ratio[0],
                velocity_range=velocity_range,
                exposure_time=exposure_time,
                telescope_mirror_radius=telescope_mirror_radius,
                telescope_throughput=telescope_throughput,
                instrument_resolving_power=instrument_resolving_power,
                pixel_sampling=pixel_sampling,
                noise_correction_coefficient=1.0,
                scale_factor=1.0,
                star_snr=star_snr,
                star_apparent_magnitude=star_apparent_magnitudes,
                star_snr_reference_apparent_magnitude=star_apparent_magnitude_j,
                mock_observation_number=100
            )

    with open('au_mic_c_ccf.pkl', 'wb') as f:
        pickle.dump([snrs, snrs_error, results, tsm], f)

    for meta in metallicity:
        print(f'\n [Z/H] = {10 ** meta}:')

        for species in species_list:
            if species == 'all':
                continue

            best_snr, best_band, best_setting = find_best_setting(species, snrs[meta])
            print(f"Species '{species}', best setting: {best_band}{best_setting} (CCF SNR: {best_snr})")

    for species in species_list:
        if species == 'all':
            continue

        plt.figure(figsize=(16, 9))
        plot_snr_settings_bars(
            species, snrs,
            model_labels=[rf"Z/H = {10 ** metallicity[0]:.1f} $\times$ solar",
                          rf"Z/H = {10 ** metallicity[1]:.1f} $\times$ solar"],
            planet_name=planet_name,
            threshold=5,
            y_err=snrs_error
        )

        plt.savefig(f"./figures/{planet_name.replace(' ', '_')}/{species}_detection.png")

    plot_tsm_pcloud_snr(
        p_cloud, tsm, snrs,
        metallicity=metallicity[2],
        band='J',
        setting=list(settings.keys())[0],
        species=species_list[1],
        planet_name=planet_name,
        exposure_time=exposure_time
    )

    mmr = models['K']['T_int = 50, [Fe/H] = 1, C/O = 0.55, P_cloud = 100.0']['all'].mass_mixing_ratios

    for species in species_list:
        if species == 'all':
            continue

        for line_species in line_species_list:
            if species + '_' in line_species:
                plot_mmr(mmr[line_species], pressures*1e5, color=species_color[species])

                break


def main_hd():
    # Base parameters
    planet_name = 'HD 136352 c'
    lbl_opacity_sampling = 1
    do_scat_emis = False
    model_suffix = ['', 'neq']
    wlen_modes = {
        'Y': np.array([0.92, 1.15]),
        'J': np.array([1.07, 1.4]),
        'H': np.array([1.4, 1.88]),
        'K': np.array([1.88, 2.55]),
        'L': np.array([2.7, 4.25]),
        'M': np.array([3.25, 5.5])
    }

    # Load planet
    planet_1 = Planet.get(planet_name)

    pressures = np.logspace(-10, 2, 130)
    distances = [213.982 * nc.pc]  # np.logspace(1, 3, 7) * nc.c * 3600 * 24 * 365.25
    # star_apparent_magnitude_v = 12.095
    star_apparent_magnitude_j = planet_1.system_apparent_magnitude_j
    # star_apparent_magnitude_j = 10
    star_apparent_magnitudes = [star_apparent_magnitude_j]
    mock_observations_number = 100
    transit_number = 2

    # Models to be tested
    t_int = [50]
    metallicity = [2]#[1, 2]
    co_ratio = [0.55]
    p_cloud = [1e2]
    species_list = ['all', 'CH4']#['all', 'H2O', 'CO', 'CO2', 'H2S', 'NH3', 'CH4']

    line_species_list = [
        'CH4_main_iso',
        'CO_all_iso',
        'CO2_main_iso',
        'H2O_main_iso',
        'H2S_main_iso',
        'HCN_main_iso',
        'K',
        'Na_allard',
        'NH3_main_iso',
        'PH3_main_iso'
    ]

    molecular_w = {
        'CH4_main_iso': 16,
        'CO_all_iso': 28,
        'CO2_main_iso': 44,
        'H2O_main_iso': 18,
        'H2S_main_iso': 34.1,
        'HCN_main_iso': 27,
        'NH3_main_iso': 17,
        'PH3_main_iso': 34
    }

    data = load_hdf5(f"{module_dir}/../../../exorem/outputs/exorem/hd_136352_c_z10_g10_t50_co0.55_nocloud.h5")
    mmmr_neq = data['outputs']['layers']['molar_mass'][()][::-1]
    p_neq = data['outputs']['layers']['pressure'][()][::-1] * 1e-5
    vmr_neq = data['outputs']['layers']['volume_mixing_ratios']['absorbers']
    t_neq = data['outputs']['layers']['temperature'][()][::-1]
    mmr_neq = {}

    mmmr_neq_i = interp1d(p_neq, mmmr_neq, fill_value='extrapolate')
    mmmr_neq_i = mmmr_neq_i(pressures)
    t_neq_i = interp1d(p_neq, t_neq, fill_value='extrapolate')
    t_neq_i = t_neq_i(pressures)

    for species in list(vmr_neq.keys()):
        vmr_neq_i = interp1d(p_neq, vmr_neq[species][()][::-1], fill_value='extrapolate')
        vmr_neq_i = vmr_neq_i(pressures)

        for line_species in molecular_w:
            if species + '_' in line_species:
                mmr_neq[line_species] = molecular_w[line_species] * 1e-3 * vmr_neq_i / mmmr_neq_i

    mmr_neq['MMW'] = mmmr_neq_i

    mass_fractions = [
        None,
        mmr_neq
    ]

    temperature_profiles = [
        None,
        t_neq_i
    ]

    # Observation parameters
    # Actually matter (used to get the CRIRES SNR data from the ETC website)

    exposure_time = 6 * 3600  # 4 * 3600
    integration_time = 60
    airmass = 1.2
    velocity_range = [-1400, 1400]
    instrument_resolving_power = 8e4
    # Old (don't do anything anymore)
    telescope_mirror_radius = 8.2e2 / 2  # cm
    telescope_throughput = 0.1
    pixel_sampling = 3

    # Load settings
    settings = load_wavelength_settings(module_dir + '/crires/wavelength_settings.dat')

    # Load region species
    with open(module_dir + '/crires/opa_regions_species.pkl', 'rb') as f:
        regions_species = pickle.load(f)

    planet_2 = copy.copy(planet_1)
    planets = [planet_1, planet_2]
    planet_keys = ['eq', 'neq']

    # Load signal to noise_matrix ratios
    star_snr = get_multiple_crires_snr_data(settings, star_apparent_magnitude_j,  # TODO apparent mag is really annoying
                                            planet_1.star_effective_temperature, exposure_time,
                                            integration_time, airmass,  # TODO add FLI and seeing
                                            rewrite=False, star_apparent_magnitude_band='J')

    # Generate parameter dictionaries
    parameter_dicts = get_parameter_dicts(
        t_int, metallicity, co_ratio, p_cloud
    )

    # Load/generate relevant models
    models = {}

    for wlen_mode in wlen_modes:
        print(f"Band {wlen_mode}...")

        atmosphere = None

        for i, planet in enumerate(planets):
            planet_key = planet_keys[i]

            if planet_key not in models:
                models[planet_key] = {}

            # Initialize grid
            models[planet_key][wlen_mode], all_models_exist = init_model_grid(
                planet.name, lbl_opacity_sampling, do_scat_emis, parameter_dicts, species_list,
                wavelength_boundaries=wlen_modes[wlen_mode],
                model_suffix=model_suffix[i]
            )

            if not all_models_exist:
                if atmosphere is None:
                    # Load or generate atmosphere
                    atmosphere, atmosphere_filename = SpectralModelLegacy.get_atmosphere_model(
                        wlen_bords_micron=wlen_modes[wlen_mode],
                        pressures=pressures,
                        line_species_list=line_species_list,
                        rayleigh_species=SpectralModelLegacy.default_rayleigh_species,
                        continuum_opacities=SpectralModelLegacy.default_continuum_opacities,
                        lbl_opacity_sampling=lbl_opacity_sampling,
                        do_scat_emis=do_scat_emis,
                        model_suffix=model_suffix[0]
                    )

                # Load or generate models
                models[planet_key][wlen_mode] = generate_model_grid(
                    models=models[planet_key][wlen_mode],
                    pressures=pressures,
                    line_species_list=line_species_list,
                    rayleigh_species='default',
                    continuum_opacities='default',
                    model_suffix=model_suffix[i],
                    atmosphere=atmosphere,
                    mass_fractions=mass_fractions[i],
                    temperature_profile=temperature_profiles[i],
                    calculate_transmission_spectrum=True,
                    rewrite=False,
                    save=True
                )
            else:
                # Load existing models
                models[planet_key][wlen_mode] = load_model_grid(models[planet_key][wlen_mode])

    snrs = {}
    snrs_error = {}
    results = {}

    for trans in [2]:
        for i, planet in enumerate(planets):
            planet_key = planet_keys[i]
            planet_key_ = planet_keys[i] + f' {trans} transits'
            snrs[planet_key_] = {}
            snrs_error[planet_key_] = {}
            results[planet_key_] = {}

            for meta in metallicity:
                snrs[planet_key_][meta] = {}
                snrs_error[planet_key_][meta] = {}
                results[planet_key_][meta] = {}

                for band in wlen_modes:
                    snrs[planet_key_][meta][band], snrs_error[planet_key_][meta][band], tsm, \
                        results[planet_key_][meta][band] = \
                        get_tsm_snr_pcloud(
                            band=band,
                            wavelength_boundaries=wlen_modes[band] * 1e-4,
                            star_distances=distances,
                            p_clouds=p_cloud,
                            models=models[planet_key],
                            species_list=species_list,
                            settings=settings,
                            planet=planet,
                            t_int=t_int[0],
                            metallicity=meta,
                            co_ratio=co_ratio[0],
                            velocity_range=velocity_range,
                            exposure_time=exposure_time,
                            telescope_mirror_radius=telescope_mirror_radius,
                            telescope_throughput=telescope_throughput,
                            instrument_resolving_power=instrument_resolving_power,
                            pixel_sampling=pixel_sampling,
                            noise_correction_coefficient=1.0,
                            scale_factor=1.0,
                            star_snr=star_snr,
                            star_apparent_magnitude=star_apparent_magnitudes,
                            star_snr_reference_apparent_magnitude=star_apparent_magnitude_j,
                            mock_observation_number=mock_observations_number,
                            transit_number=transit_number,
                            regions_species=regions_species
                        )

    with open(f'{planet_name}_ccf.pkl', 'wb') as f:
        pickle.dump({'mean signal-to-noise_ratios': snrs, 'std signal-to-noise_ratios': snrs_error,
                     'results (100 takes)': results}, f)

    snr = {}
    snr_err = {}

    for key in snrs:
        snr[key] = snrs[key][2]
        snr_err[key] = snrs_error[key][2]

    for species in species_list:
        if species == 'all':
            continue

        plt.figure(figsize=(16, 9))
        plot_snr_settings_bars(
            species, snr,
            planet_name=planets[0].name,
            threshold=5,
            y_err=snr_err
        )

        plt.savefig(f"./figures/{planet_name.replace(' ', '_')}/{species}_detection_"
                    f"Teq{planets[i].equilibrium_temperature}K_meta{metallicity[1]}.png")
        plt.close("all")

    snr_x = {}
    velocity = np.linspace(velocity_range[0] + 100, velocity_range[1] - 100, 5000)
    ccf_size = np.size(velocity)
    best_setting = {}
    max_snr = {}

    for model in snrs:
        print(f"\nModel {model}")
        snr_x[model] = {}
        best_setting[model] = {}
        max_snr[model] = {}

        for meta in snrs[model]:
            print(f"[Z/H] = {meta}")
            snr_x[model][meta] = {}
            best_setting[model][meta] = {}
            max_snr[model][meta] = {}

            for species in species_list:
                if species == 'all':
                    continue

                print(f'Species {species}')
                snr_x[model][meta][species] = np.zeros(mock_observations_number)
                best_setting[model][meta][species] = {}
                max_snr[model][meta][species] = {}
                ccf = np.zeros((mock_observations_number, ccf_size))

                for band in snrs[model][meta]:
                    model_key = list(results[model][meta][band].keys())[1]

                    best_setting[model][meta][species][band] = None
                    max_snr[model][meta][species][band] = -np.inf
                    ccf_max = 0
                    vel_max = 0

                    for setting in snrs[model][meta][band]:
                        if snrs[model][meta][band][setting][species] > max_snr[model][meta][species][band]:
                            max_snr[model][meta][species][band] = snrs[model][meta][band][setting][species][0, 0]
                            ccf_max = results[model][meta][band][model_key][band][setting][species]['CCF']
                            vel_max = results[model][meta][band][model_key][band][setting][species]['velocity']
                            best_setting[model][meta][species][band] = setting

                    print(f"Band {band}: "
                          f"best setting {best_setting[model][meta][species][band] } "
                          f"(SNR = {max_snr[model][meta][species][band] })")
                    if max_snr[model][meta][species][band] <= 0:
                        #print(f"Bad SNR, skipping band {band}")

                        continue

                    ccf_max = interp1d(vel_max, np.transpose(ccf_max))
                    ccf += ccf_max(velocity)

                for i in range(mock_observations_number):
                    snr_x[model][meta][species][i], mu, std = calculate_ccf_snr(velocity, ccf[i, :])

    # Print results
    for model in snr_x:
        print(f'Model {model}')
        for meta in snr_x[model]:
            print(f'[Z/H] = {meta}')
            for species in snr_x[model][meta]:
                print(
                    f'{species}: SNR = {np.mean(snr_x[model][meta][species]):.3f} +/- {np.std(snr_x[model][meta][species]):.3f}')

    with open(f'{planet_name}_sum_band_cff.pkl', 'wb') as f:
        pickle.dump({'Sum SNR bands best setting (100 takes)': snr_x, 'Best setting': best_setting}, f)


def main_toi776():
    # Base parameters
    planet_name = 'TOI-776 c'
    lbl_opacity_sampling = 1
    do_scat_emis = False
    model_suffix = ''
    wlen_modes = {
        'Y': np.array([0.92, 1.15]),
        'J': np.array([1.07, 1.4]),
        'H': np.array([1.4, 1.88]),
        'K': np.array([1.88, 2.55]),
        'L': np.array([2.7, 4.25]),
        'M': np.array([3.25, 5.5])
    }

    # Load planet
    planet = Planet.get(planet_name)
    planet.transit_duration *= 0.8

    # Load region species
    with open('./petitRADTRANS/ccf/crires/opa_regions_species.pkl', 'rb') as f:
        regions_species = pickle.load(f)

    # Parameters
    pressures = np.logspace(-10, 2, 130)
    distances = [213.982 * nc.pc]  # np.logspace(1, 3, 7) * nc.c * 3600 * 24 * 365.25
    # star_apparent_magnitude_v = 12.095
    star_apparent_magnitude_j = planet.system_apparent_magnitude_j
    # star_apparent_magnitude_j = 10
    star_apparent_magnitudes = [star_apparent_magnitude_j]

    # Models to be tested
    t_int = [50]
    metallicity = [1, 2]
    co_ratio = [0.55]
    p_cloud = [1e2]
    species_list = ['all', 'H2O', 'CO', 'CO2', 'H2S', 'NH3', 'PH3', 'HCN', 'CH4']

    line_species_list = [
        'CH4_main_iso',
        'CO_all_iso',
        'CO2_main_iso',
        'H2O_main_iso',
        'H2S_main_iso',
        'HCN_main_iso',
        'K',
        'Na_allard',
        'NH3_main_iso',
        'PH3_main_iso'
    ]

    # Observation parameters
    # Actually matter (used to get the CRIRES SNR data from the ETC website)
    exposure_time = 6 * 3600
    integration_time = 60
    airmass = 1.2
    velocity_range = [-1400, 1400]
    instrument_resolving_power = 8e4
    # Old (don't do anything anymore)
    telescope_mirror_radius = 8.2e2 / 2  # cm
    telescope_throughput = 0.1
    pixel_sampling = 3

    # Load settings
    settings = load_wavelength_settings(module_dir + '/crires/wavelength_settings.dat')

    # Load signal to noise_matrix ratios
    star_snr = get_multiple_crires_snr_data(settings, star_apparent_magnitude_j,  # TODO apparent mag is really annoying
                                            planet.star_effective_temperature, exposure_time,
                                            integration_time, airmass,  # TODO add FLI and seeing
                                            rewrite=False, star_apparent_magnitude_band='J')

    # Generate parameter dictionaries
    parameter_dicts = get_parameter_dicts(
        t_int, metallicity, co_ratio, p_cloud
    )

    # Load/generate relevant models
    models = {}

    for wlen_mode in wlen_modes:
        print(f"Band {wlen_mode}...")

        # Initialize grid
        models[wlen_mode], all_models_exist = init_model_grid(
            planet_name, lbl_opacity_sampling, do_scat_emis, parameter_dicts, species_list,
            wavelength_boundaries=wlen_modes[wlen_mode],
            model_suffix=model_suffix
        )

        if not all_models_exist:
            # Load or generate atmosphere
            atmosphere, atmosphere_filename = SpectralModelLegacy.get_atmosphere_model(
                wlen_bords_micron=wlen_modes[wlen_mode],
                pressures=pressures,
                line_species_list=SpectralModelLegacy.default_line_species,
                rayleigh_species=SpectralModelLegacy.default_rayleigh_species,
                continuum_opacities=SpectralModelLegacy.default_continuum_opacities,
                lbl_opacity_sampling=lbl_opacity_sampling,
                do_scat_emis=do_scat_emis,
                model_suffix=model_suffix
            )

            # Load or generate models
            models[wlen_mode] = generate_model_grid(
                models=models[wlen_mode],
                pressures=pressures,
                line_species_list=line_species_list,
                rayleigh_species='default',
                continuum_opacities='default',
                model_suffix=model_suffix,
                atmosphere=atmosphere,
                calculate_transmission_spectrum=True,
                rewrite=False,
                save=True
            )
        else:
            # Load existing models
            models[wlen_mode] = load_model_grid(models[wlen_mode])

    snrs = {}
    snrs_error = {}
    results = {}

    for meta in metallicity:
        snrs[meta] = {}
        snrs_error[meta] = {}
        results[meta] = {}

        for band in wlen_modes:
            snrs[meta][band], snrs_error[meta][band], tsm, results[meta][band] = get_tsm_snr_pcloud(
                band=band,
                wavelength_boundaries=wlen_modes[band] * 1e-4,
                star_distances=distances,
                p_clouds=p_cloud,
                models=models,
                species_list=species_list,
                settings=settings,
                planet=planet,
                t_int=t_int[0],
                metallicity=meta,
                co_ratio=co_ratio[0],
                velocity_range=velocity_range,
                exposure_time=exposure_time,
                telescope_mirror_radius=telescope_mirror_radius,
                telescope_throughput=telescope_throughput,
                instrument_resolving_power=instrument_resolving_power,
                pixel_sampling=pixel_sampling,
                noise_correction_coefficient=1.0,
                scale_factor=1.0,
                star_snr=star_snr,
                star_apparent_magnitude=star_apparent_magnitudes,
                star_snr_reference_apparent_magnitude=star_apparent_magnitude_j,
                mock_observation_number=100,
                transit_number=4,
                regions_species=regions_species
            )

    for meta in metallicity:
        print(f'\n [Z/H] = {10 ** meta}:')

        for species in species_list:
            if species == 'all':
                continue

            best_snr, best_band, best_setting = find_best_setting(species, snrs[meta])
            print(f"Species '{species}', best setting: {best_band}{best_setting} (CCF SNR: {best_snr})")

    for species in species_list:
        if species == 'all':
            continue

        plt.figure(figsize=(16, 9))
        plot_snr_settings_bars(
            species, snrs,
            model_labels=[rf"Z/H = {10 ** metallicity[0]:.1f} $\times$ solar",
                          rf"Z/H = {10 ** metallicity[1]:.1f} $\times$ solar"],
            planet_name=planet_name,
            threshold=5,
            y_err=snrs_error
        )

        plt.savefig(f"./figures/{planet_name.replace(' ', '_')}/{species}_detection.png")

    mmr = models['K']['T_int = 50, [Fe/H] = 1, C/O = 0.55, P_cloud = 100.0']['all'].mass_mixing_ratios

    for species in species_list:
        if species == 'all':
            continue

        for line_species in line_species_list:
            if species + '_' in line_species:
                plot_mmr(mmr[line_species], pressures*1e5, color=species_color[species])

                break


def main_toi():
    # Base parameters
    planet_name = 'TOI-269 b'
    lbl_opacity_sampling = 1
    do_scat_emis = False
    model_suffix = ''
    wlen_modes = {
        'Y': np.array([0.92, 1.15]),
        'J': np.array([1.07, 1.4]),
        'H': np.array([1.4, 1.88]),
        'K': np.array([1.88, 2.55]),
        'L': np.array([2.7, 4.25]),
        'M': np.array([3.25, 5.5])
    }

    # Load planet
    planet = Planet.get(planet_name)

    pressures = np.logspace(-10, 2, 130)
    distances = [213.982 * nc.pc]  # np.logspace(1, 3, 7) * nc.c * 3600 * 24 * 365.25
    # star_apparent_magnitude_v = 12.095
    star_apparent_magnitude_j = planet.system_apparent_magnitude_j
    # star_apparent_magnitude_j = 10
    star_apparent_magnitudes = [star_apparent_magnitude_j]

    # Models to be tested
    t_int = [50]
    metallicity = [0, 2]
    co_ratio = [0.55]
    p_cloud = [1e2]
    species_list = ['all', 'H2O', 'CO', 'CO2', 'H2S', 'NH3', 'PH3', 'HCN', 'CH4']

    line_species_list = [
        'CH4_main_iso',
        'CO_all_iso',
        'CO2_main_iso',
        'H2O_main_iso',
        'H2S_main_iso',
        'HCN_main_iso',
        'K',
        'Na_allard',
        'NH3_main_iso',
        'PH3_main_iso'
    ]

    # Observation parameters
    # Actually matter (used to get the CRIRES SNR data from the ETC website)
    exposure_time = 6 * 3600  # 4 * 3600
    integration_time = 60
    airmass = 1.2
    velocity_range = [-1400, 1400]
    instrument_resolving_power = 8e4
    # Old (don't do anything anymore)
    telescope_mirror_radius = 8.2e2 / 2  # cm
    telescope_throughput = 0.1
    pixel_sampling = 3

    # Load settings
    settings = load_wavelength_settings(module_dir + '/crires/wavelength_settings.dat')

    # Periapsis and Apoapsis
    planet_apo = copy.copy(planet)
    planet_peri = copy.copy(planet)

    planet_apo.name = planet.name + '_apoapsis'
    planet_peri.name = planet.name + '_periapsis'

    planet_apo.orbit_semi_major_axis = planet.orbit_semi_major_axis * (1 + planet.orbital_eccentricity)
    planet_peri.orbit_semi_major_axis = planet.orbit_semi_major_axis * (1 - planet.orbital_eccentricity)

    planet_apo.equilibrium_temperature, \
        planet_apo.equilibrium_temperature_error_upper, \
        planet_apo.equilibrium_temperature_error_lower = planet_apo.calculate_planetary_equilibrium_temperature()

    planet_peri.equilibrium_temperature, \
        planet_peri.equilibrium_temperature_error_upper, \
        planet_peri.equilibrium_temperature_error_lower = planet_peri.calculate_planetary_equilibrium_temperature()

    planet_peri.save()
    planet_apo.save()

    planet_1 = copy.copy(planet)
    planet_pero_1 = copy.copy(planet_peri)

    planets = [planet_peri, planet, planet_1, planet_pero_1]

    for i in range(2):
        planets[i].transit_duration = 2 * planets[i].transit_duration

    # Load signal to noise_matrix ratios
    star_snr = get_multiple_crires_snr_data(settings, star_apparent_magnitude_j,  # TODO apparent mag is really annoying
                                            planet.star_effective_temperature, exposure_time,
                                            integration_time, airmass,  # TODO add FLI and seeing
                                            rewrite=False, star_apparent_magnitude_band='J')

    # Generate parameter dictionaries
    parameter_dicts = get_parameter_dicts(
        t_int, metallicity, co_ratio, p_cloud
    )

    # Load/generate relevant models
    models = {}

    for wlen_mode in wlen_modes:
        print(f"Band {wlen_mode}...")

        atmosphere = None

        for planet in planets:
            planet_key = f"T_eq {planet.equilibrium_temperature}, " \
                         f"{planet.transit_duration/planet_1.transit_duration} transits"

            if planet_key not in models:
                models[planet_key] = {}

            # Initialize grid
            models[planet_key][wlen_mode], all_models_exist = init_model_grid(
                planet.name, lbl_opacity_sampling, do_scat_emis, parameter_dicts, species_list,
                wavelength_boundaries=wlen_modes[wlen_mode],
                model_suffix=model_suffix
            )

            if not all_models_exist:
                if atmosphere is None:
                    # Load or generate atmosphere
                    atmosphere, atmosphere_filename = SpectralModelLegacy.get_atmosphere_model(
                        wlen_bords_micron=wlen_modes[wlen_mode],
                        pressures=pressures,
                        line_species_list=line_species_list,
                        rayleigh_species=SpectralModelLegacy.default_rayleigh_species,
                        continuum_opacities=SpectralModelLegacy.default_continuum_opacities,
                        lbl_opacity_sampling=lbl_opacity_sampling,
                        do_scat_emis=do_scat_emis,
                        model_suffix=model_suffix
                    )

                # Load or generate models
                models[planet_key][wlen_mode] = generate_model_grid(
                    models=models[planet_key][wlen_mode],
                    pressures=pressures,
                    line_species_list=line_species_list,
                    rayleigh_species='default',
                    continuum_opacities='default',
                    model_suffix=model_suffix,
                    atmosphere=atmosphere,
                    calculate_transmission_spectrum=True,
                    rewrite=False,
                    save=True
                )
            else:
                # Load existing models
                models[planet_key][wlen_mode] = load_model_grid(models[planet_key][wlen_mode])

    snrs = {}
    snrs_error = {}
    tsm = {}

    for planet in planets:
        planet_key = f"T_eq {planet.equilibrium_temperature}, " \
                     f"{planet.transit_duration / planet_1.transit_duration} transits"
        snrs[planet_key] = {}
        snrs_error[planet_key] = {}
        tsm[planet_key] = {}

        for meta in metallicity:
            snrs[planet_key][meta] = {}
            snrs_error[planet_key][meta] = {}
            tsm[planet_key][meta] = {}

            for band in wlen_modes:
                snrs[planet_key][meta][band], snrs_error[planet_key][meta], tsm[planet_key][meta][band], results = \
                    get_tsm_snr_pcloud(
                        band=band,
                        wavelength_boundaries=wlen_modes[band] * 1e-4,
                        star_distances=distances,
                        p_clouds=p_cloud,
                        models=models[planet_key],
                        species_list=species_list,
                        settings=settings,
                        planet=planet,
                        t_int=t_int[0],
                        metallicity=meta,
                        co_ratio=co_ratio[0],
                        velocity_range=velocity_range,
                        exposure_time=exposure_time,
                        telescope_mirror_radius=telescope_mirror_radius,
                        telescope_throughput=telescope_throughput,
                        instrument_resolving_power=instrument_resolving_power,
                        pixel_sampling=pixel_sampling,
                        noise_correction_coefficient=1.0,
                        scale_factor=1.0,
                        star_snr=star_snr,
                        star_apparent_magnitude=star_apparent_magnitudes,
                        star_snr_reference_apparent_magnitude=star_apparent_magnitude_j,
                        mock_observation_number=100
                )

    for i in range(2):
        planet_key1 = f"T_eq {planets[i].equilibrium_temperature}, " \
                     f"{planets[i].transit_duration / planet_1.transit_duration} transits"
        planet_key2 = f"T_eq {planets[i + 2].equilibrium_temperature}, " \
                      f"{planets[i + 2].transit_duration / planet_1.transit_duration} transits"

        for species in species_list:
            if species == 'all':
                continue

            snr = {
                f"{planets[i + 2].transit_duration / planet_1.transit_duration} "
                f"transits": snrs[planet_key1][metallicity[1]],
                f"{planets[i].transit_duration / planet_1.transit_duration} "
                f"transits": snrs[planet_key2][metallicity[1]],
            }

            plt.figure(figsize=(16, 9))
            plot_snr_settings_bars(
                species, snr,
                model_labels=[f"{planets[i + 2].transit_duration / planet_1.transit_duration} transits",
                              f"{planets[i].transit_duration / planet_1.transit_duration} transits"],
                planet_name=planets[i].name,
                threshold=5,
                y_err=2
            )

            plt.savefig(f"./figures/{planet_name.replace(' ', '_')}/{species}_detection_"
                        f"Teq{planets[i].equilibrium_temperature}K_meta{metallicity[1]}.png")
            plt.close("all")

    for i in range(len(planets)):
        for species in species_list:
            if species == 'all':
                continue

            plt.figure(figsize=(16, 9))
            plot_snr_settings_bars_from_gaussian(
                species, snrs[planets[i].equilibrium_temperature],
                planet_name=planets[i].name,
                threshold=5
            )

            plt.savefig(f"./figures/{planet_name.replace(' ', '_')}/{species}_detection_"
                        f"Teq{planets[i].equilibrium_temperature}K_tobs{exposure_time/3600}h_"
                        f"{len(snrs[planets[i].equilibrium_temperature])}takes.png")

    plot_tsm_pcloud_snr(
        p_cloud, tsm, snrs,
        metallicity=metallicity[2],
        band='J',
        setting=list(settings.keys())[0],
        species=species_list[1],
        planet_name=planet_name,
        exposure_time=exposure_time
    )


def main_teff():
    # Base parameters
    lbl_opacity_sampling = 1
    do_scat_emis = False
    model_suffix = ''
    wlen_modes = {
        'Y': np.array([0.92, 1.15]),
        'J': np.array([1.07, 1.4]),
        'H': np.array([1.4, 1.88]),
        'K': np.array([1.88, 2.55]),
        'L': np.array([2.7, 4.25]),
        'M': np.array([3.25, 5.5])
    }

    pressures = np.logspace(-10, 2, 130)
    distances = [213.982 * nc.pc]  # np.logspace(1, 3, 7) * nc.c * 3600 * 24 * 365.25
    # star_apparent_magnitude_v = 12.095
    # star_apparent_magnitude_j = 10.663
    star_apparent_magnitude_j = 10
    star_apparent_magnitudes = np.linspace(4, 16, 7)

    # Models to be tested
    equilibrium_temperatures = [800, 1200, 1600, 2000]
    t_int = [200]
    metallicity = [1]
    co_ratio = [0.55]
    p_cloud = [1e2]
    surface_gravities = [1000]
    species_list = ['all', 'H2O', 'CO']

    # Observation parameters
    # Actually matter (used to get the CRIRES SNR data from the ETC website)
    exposure_time = 6 * 3600
    integration_time = 60
    airmass = 1
    velocity_range = [-1750, 1750]
    instrument_resolving_power = 8e4
    # Old (don't do anything anymore)
    telescope_mirror_radius = 8.2e2 / 2  # cm
    telescope_throughput = 0.1
    pixel_sampling = 3

    # Load settings
    settings = load_wavelength_settings(module_dir + '/crires/wavelength_settings.dat')

    # Load signal to noise_matrix ratios
    star_snr = get_multiple_crires_snr_data(settings, star_apparent_magnitude_j,  # TODO apparent mag is really annoying
                                            5500, exposure_time,
                                            integration_time, airmass,  # TODO add FLI and seeing
                                            rewrite=False, star_apparent_magnitude_band='J')

    # Generate parameter dictionaries
    parameter_dicts = get_parameter_dicts(
        t_int, metallicity, co_ratio, p_cloud
    )

    # Line species list
    line_species_list = [
        'CH4_main_iso',
        'CO_all_iso',
        'CO2_main_iso',
        'H2O_main_iso',
        'H2S_main_iso',
        'HCN_main_iso',
        'K',
        'Na_allard',
        'NH3_main_iso',
        'PH3_main_iso'
    ]

    # Generate planets
    planets = []

    for equilibrium_temperature in equilibrium_temperatures:
        for g in surface_gravities:
            planets.append(make_generic_planet(nc.r_jup, g, equilibrium_temperature))
            planets[-1].save()

    # Load/generate relevant models
    models = {}

    for wlen_mode in wlen_modes:
        print(f"Band {wlen_mode}...")

        atmosphere = None

        for planet in planets:
            planet_key = f"T_eq {planet.equilibrium_temperature}, logg {np.log10(planet.surface_gravity)}"

            if planet_key not in models:
                models[planet_key] = {}

            # Initialize grid
            models[planet_key][wlen_mode], all_models_exist = init_model_grid(
                planet.name, lbl_opacity_sampling, do_scat_emis, parameter_dicts, species_list,
                wavelength_boundaries=wlen_modes[wlen_mode],
                model_suffix=model_suffix
            )

            if not all_models_exist:
                if atmosphere is None:
                    # Load or generate atmosphere
                    atmosphere, atmosphere_filename = SpectralModelLegacy.get_atmosphere_model(
                        wlen_bords_micron=wlen_modes[wlen_mode],
                        pressures=pressures,
                        line_species_list=line_species_list,
                        rayleigh_species=SpectralModelLegacy.default_rayleigh_species,
                        continuum_opacities=SpectralModelLegacy.default_continuum_opacities,
                        lbl_opacity_sampling=lbl_opacity_sampling,
                        do_scat_emis=do_scat_emis,
                        model_suffix=model_suffix
                    )

                # Load or generate models
                models[planet_key][wlen_mode] = generate_model_grid(
                    models=models[planet_key][wlen_mode],
                    pressures=pressures,
                    line_species_list=line_species_list,
                    rayleigh_species='default',
                    continuum_opacities='default',
                    model_suffix=model_suffix,
                    atmosphere=atmosphere,
                    calculate_transmission_spectrum=True,
                    rewrite=False,
                    save=True
                )
            else:
                # Load existing models
                models[planet_key][wlen_mode] = load_model_grid(models[planet_key][wlen_mode])

    snrs = {}
    snrs_error = {}
    tsm = {}

    for planet in planets:
        t_eq = planet.equilibrium_temperature
        snrs[t_eq] = {}
        snrs_error[t_eq] = {}
        tsm[t_eq] = {}

        for meta in metallicity:
            snrs[t_eq][meta] = {}
            snrs_error[t_eq][meta] = {}
            tsm[t_eq][meta] = {}

            for band in wlen_modes:
                snrs[t_eq][meta][band], snrs_error[t_eq][meta], tsm[t_eq][meta][band], results = \
                    get_tsm_snr_pcloud(
                        band=band,
                        wavelength_boundaries=wlen_modes[band] * 1e-4,
                        star_distances=distances,
                        p_clouds=p_cloud,
                        models=models[f'T_eq {t_eq}, logg 3.0'],
                        species_list=species_list,
                        settings=settings,
                        planet=planet,
                        t_int=t_int[0],
                        metallicity=meta,
                        co_ratio=co_ratio[0],
                        velocity_range=velocity_range,
                        exposure_time=exposure_time,
                        telescope_mirror_radius=telescope_mirror_radius,
                        telescope_throughput=telescope_throughput,
                        instrument_resolving_power=instrument_resolving_power,
                        pixel_sampling=pixel_sampling,
                        noise_correction_coefficient=1.0,
                        scale_factor=1.0,
                        star_snr=star_snr,
                        star_apparent_magnitude=star_apparent_magnitudes,
                        star_snr_reference_apparent_magnitude=star_apparent_magnitude_j,
                        mock_observation_number=100
                )

    # TODO fix TSM (it depends only on magnitude)
    snrs_2 = {}
    tsm_2 = tsm[planets[0].equilibrium_temperature][metallicity[0]][list(wlen_modes.keys())[0]]

    for j, t_eq in enumerate(snrs):
        for meta in snrs[t_eq]:
            if meta not in snrs_2:
                snrs_2[meta] = {}

            for band in snrs[t_eq][meta]:
                if band not in snrs_2[meta]:
                    snrs_2[meta][band] = {}

                for setting in snrs[t_eq][meta][band]:
                    if setting not in snrs_2[meta][band]:
                        snrs_2[meta][band][setting] = {}

                    for species in snrs[t_eq][meta][band][setting]:
                        if species not in snrs_2[meta][band][setting]:
                            snrs_2[meta][band][setting][species] = np.zeros(
                                (np.size(star_apparent_magnitudes), np.size(list(snrs.keys())))
                            )

                        snrs_2[meta][band][setting][species][:, j] = snrs[t_eq][meta][band][setting][species][:, 0]

    plt.figure(figsize=(10, 9/16 * 10))
    plot_tsm_x_snr(equilibrium_temperatures, tsm_2, snrs_2[1]['K']['2217']['H2O'], 'K', '2217', 'H2O',
                   '1 R_jup logg=3 [Z/H]=1',
                   exposure_time, x_label='Equilibrium temperature (K)')


def main_tiso():
    # Base parameters
    planet_name = 'WASP-39 b'
    lbl_opacity_sampling = 1
    do_scat_emis = False
    model_suffix = ''
    wlen_modes = {
        # 'Y': np.array([0.92, 1.15]),
        # 'J': np.array([1.07, 1.4]),
        # 'H': np.array([1.4, 1.88]),
        'K': np.array([1.88, 2.55]),
        # 'L': np.array([2.7, 4.25]),
        # 'M': np.array([3.25, 5.5])
    }

    # Load planet
    planet = Planet.get(planet_name)

    pressures = np.logspace(-10, 2, 130)
    distances = [213.982 * nc.pc]  # np.logspace(1, 3, 7) * nc.c * 3600 * 24 * 365.25
    # star_apparent_magnitude_v = 12.095
    # star_apparent_magnitude_j = 10.663
    star_apparent_magnitude_j = 10
    star_apparent_magnitudes = [star_apparent_magnitude_j]  # np.linspace(4, 16, 7)

    # Models to be tested
    equilibrium_temperatures = [800, 2000]
    t_int = [200]
    metallicity = [1]
    co_ratio = [0.55]
    p_cloud = [1e2]
    species_list = ['all', 'H2O']
    mass_fractions = {  # species not included here will be initialized with equilibrium chemistry
        'HCN_main_iso': np.ones_like(pressures) * 1e-7,
        'C2H2,acetylene': np.ones_like(pressures) * 1e-8
    }

    # Observation parameters
    # Actually matter (used to get the CRIRES SNR data from the ETC website)
    exposure_time = 6 * 3600
    integration_time = 60
    airmass = 1
    velocity_range = [-1400, 1400]
    instrument_resolving_power = 8e4
    # Old (don't do anything anymore)
    telescope_mirror_radius = 8.2e2 / 2  # cm
    telescope_throughput = 0.1
    pixel_sampling = 3

    # Load settings
    settings = load_wavelength_settings(module_dir + '/crires/wavelength_settings.dat')

    # Load signal to noise_matrix ratios
    star_snr = get_multiple_crires_snr_data(settings, star_apparent_magnitude_j,  # TODO apparent mag is really annoying
                                            5500, exposure_time,
                                            integration_time, airmass,  # TODO add FLI and seeing
                                            rewrite=False, star_apparent_magnitude_band='J')

    # Generate parameter dictionaries
    parameter_dicts = get_parameter_dicts(
        t_int, metallicity, co_ratio, p_cloud
    )

    # Line species list
    line_species_list = [
        'CH4_main_iso',
        'CO_all_iso',
        'CO2_main_iso',
        'H2O_main_iso',
        'H2S_main_iso',
        'HCN_main_iso',
        'K',
        'Na_allard',
        'NH3_main_iso',
        'PH3_main_iso'
    ]

    # Generate planets
    planets = []

    for equilibrium_temperature in equilibrium_temperatures:
        planets.append(copy.copy(planet))
        planets[-1].name += f'_teq{equilibrium_temperature}K'
        planets[-1].equilibrium_temperature = equilibrium_temperature
        planets[-1].save()  # should check if exists before saving

    # Load/generate relevant models
    models = {}

    for wlen_mode in wlen_modes:
        print(f"Band {wlen_mode}...")

        atmosphere = None

        for planet in planets:
            planet_key = f"T_eq {planet.equilibrium_temperature}"

            if planet_key not in models:
                models[planet_key] = {}

            # Initialize grid
            models[planet_key][wlen_mode], all_models_exist = init_model_grid(
                planet.name, lbl_opacity_sampling, do_scat_emis, parameter_dicts, species_list,
                wavelength_boundaries=wlen_modes[wlen_mode],
                model_suffix=model_suffix
            )

            if not all_models_exist:
                if atmosphere is None:
                    # Load or generate atmosphere
                    atmosphere, atmosphere_filename = SpectralModelLegacy.get_atmosphere_model(
                        wlen_bords_micron=wlen_modes[wlen_mode],
                        pressures=pressures,
                        line_species_list=line_species_list,
                        rayleigh_species=SpectralModelLegacy.default_rayleigh_species,
                        continuum_opacities=SpectralModelLegacy.default_continuum_opacities,
                        lbl_opacity_sampling=lbl_opacity_sampling,
                        do_scat_emis=do_scat_emis,
                        model_suffix=model_suffix
                    )

                # Load or generate models
                models[planet_key][wlen_mode] = generate_model_grid(
                    models=models[planet_key][wlen_mode],
                    pressures=pressures,
                    line_species_list=line_species_list,
                    rayleigh_species='default',
                    continuum_opacities='default',
                    model_suffix=model_suffix,
                    atmosphere=atmosphere,
                    temperature_profile=planet.equilibrium_temperature,
                    mass_fractions=mass_fractions,
                    calculate_transmission_spectrum=True,
                    rewrite=False,
                    save=True
                )
            else:
                # Load existing models
                models[planet_key][wlen_mode] = load_model_grid(models[planet_key][wlen_mode])

    snrs = {}
    snrs_error = {}
    tsm = {}

    for planet in planets:
        t_eq = planet.equilibrium_temperature
        snrs[t_eq] = {}
        snrs_error[t_eq] = {}
        tsm[t_eq] = {}

        for meta in metallicity:
            snrs[t_eq][meta] = {}
            snrs_error[t_eq][meta] = {}
            tsm[t_eq][meta] = {}

            for band in wlen_modes:
                snrs[t_eq][meta][band], snrs_error[t_eq][meta][band], tsm[t_eq][meta][band], results = \
                    get_tsm_snr_pcloud(
                        band=band,
                        wavelength_boundaries=wlen_modes[band] * 1e-4,
                        star_distances=distances,
                        p_clouds=p_cloud,
                        models=models[f'T_eq {t_eq}, logg 3.0'],
                        species_list=species_list,
                        settings=settings,
                        planet=planet,
                        t_int=t_int[0],
                        metallicity=meta,
                        co_ratio=co_ratio[0],
                        velocity_range=velocity_range,
                        exposure_time=exposure_time,
                        telescope_mirror_radius=telescope_mirror_radius,
                        telescope_throughput=telescope_throughput,
                        instrument_resolving_power=instrument_resolving_power,
                        pixel_sampling=pixel_sampling,
                        noise_correction_coefficient=1.0,
                        scale_factor=1.0,
                        star_snr=star_snr,
                        star_apparent_magnitude=star_apparent_magnitudes,
                        star_snr_reference_apparent_magnitude=star_apparent_magnitude_j
                )

    # TODO fix TSM (it depends only on magnitude)
    snrs_2 = {}
    tsm_2 = tsm[planets[0].equilibrium_temperature][metallicity[0]][list(wlen_modes.keys())[0]]

    for j, t_eq in enumerate(snrs):
        for meta in snrs[t_eq]:
            if meta not in snrs_2:
                snrs_2[meta] = {}

            for band in snrs[t_eq][meta]:
                if band not in snrs_2[meta]:
                    snrs_2[meta][band] = {}

                for setting in snrs[t_eq][meta][band]:
                    if setting not in snrs_2[meta][band]:
                        snrs_2[meta][band][setting] = {}

                    for species in snrs[t_eq][meta][band][setting]:
                        if species not in snrs_2[meta][band][setting]:
                            snrs_2[meta][band][setting][species] = np.zeros(
                                (np.size(star_apparent_magnitudes), np.size(list(snrs.keys())))
                            )

                        snrs_2[meta][band][setting][species][:, j] = snrs[t_eq][meta][band][setting][species][:, 0]

    plt.figure(figsize=(10, 9/16 * 10))
    plot_tsm_x_snr(equilibrium_temperatures, tsm_2, snrs_2[1]['K']['2217']['H2O'], 'K', '2217', 'H2O',
                   '1 R_jup logg=3 [Z/H]=1',
                   exposure_time, x_label='Equilibrium temperature (K)')


def _test():
    # Base parameters
    planet_name = 'WASP-39 b'
    lbl_opacity_sampling = 1
    do_scat_emis = False
    model_suffix = ''
    wlen_modes = {
        'K': np.array([1.88, 2.55])
    }

    planet = Planet.get(planet_name)
    pressures = np.logspace(-10, 2, 130)
    distances = [213.982 * nc.pc]
    star_apparent_magnitude_j = planet.system_apparent_magnitude_j
    star_apparent_magnitudes = [star_apparent_magnitude_j]

    # Models to be tested
    t_int = [50]
    metallicity = [0]
    co_ratio = [0.55]
    p_cloud = [1e2]
    species_list = ['all', 'H2O']

    line_species_list = [
        'CH4_main_iso',
        'CO_all_iso',
        'CO2_main_iso',
        'H2O_main_iso',
        'H2S_main_iso',
        'HCN_main_iso',
        'K',
        'Na_allard',
        'NH3_main_iso',
        'PH3_main_iso'
    ]

    # Observation parameters
    # Actually matter (used to get the CRIRES SNR data from the ETC website)
    integration_time = 60
    airmass = 1.2
    velocity_range = [-1400, 1400]
    instrument_resolving_power = 8e4
    # Old (don't do anything anymore)
    telescope_mirror_radius = 8.2e2 / 2  # cm
    telescope_throughput = 0.1
    pixel_sampling = 3

    # Load settings
    settings = load_wavelength_settings(module_dir + '/crires/wavelength_settings.dat')
    settings = {'K': {'2148': settings['K']['2148']}}

    # Load planet
    exposure_time = 1000 * 3600
    planet.transit_duration = 500 * 3600

    # Load signal to noise_matrix ratios
    star_snr = get_multiple_crires_snr_data(settings, star_apparent_magnitude_j,  # TODO apparent mag is really annoying
                                            planet.star_effective_temperature, exposure_time,
                                            integration_time, airmass,  # TODO add FLI and seeing
                                            rewrite=False, star_apparent_magnitude_band='J')

    # Generate parameter dictionaries
    parameter_dicts = get_parameter_dicts(
        t_int, metallicity, co_ratio, p_cloud
    )

    # Load/generate relevant models
    models = {}

    for wlen_mode in wlen_modes:
        print(f"Band {wlen_mode}...")

        # Initialize grid
        models[wlen_mode], all_models_exist = init_model_grid(
            planet_name, lbl_opacity_sampling, do_scat_emis, parameter_dicts, species_list,
            wavelength_boundaries=wlen_modes[wlen_mode],
            model_suffix=model_suffix
        )

        if not all_models_exist:
            # Load or generate atmosphere
            atmosphere, atmosphere_filename = SpectralModelLegacy.get_atmosphere_model(
                wlen_bords_micron=wlen_modes[wlen_mode],
                pressures=pressures,
                line_species_list=SpectralModelLegacy.default_line_species,
                rayleigh_species=SpectralModelLegacy.default_rayleigh_species,
                continuum_opacities=SpectralModelLegacy.default_continuum_opacities,
                lbl_opacity_sampling=lbl_opacity_sampling,
                do_scat_emis=do_scat_emis,
                model_suffix=model_suffix
            )

            # Load or generate models
            models[wlen_mode] = generate_model_grid(
                models=models[wlen_mode],
                pressures=pressures,
                line_species_list=line_species_list,
                rayleigh_species='default',
                continuum_opacities='default',
                model_suffix=model_suffix,
                atmosphere=atmosphere,
                calculate_transmission_spectrum=True,
                rewrite=False,
                save=True
            )
        else:
            # Load existing models
            models[wlen_mode] = load_model_grid(models[wlen_mode])

    snrs, snrs_error, tsm, results = get_tsm_snr_pcloud(
        band='K',
        wavelength_boundaries=wlen_modes['K'] * 1e-4,
        star_distances=distances,
        p_clouds=p_cloud,
        models=models,
        species_list=species_list,
        settings=settings,
        planet=planet,
        t_int=t_int[0],
        metallicity=metallicity[0],
        co_ratio=co_ratio[0],
        velocity_range=velocity_range,
        exposure_time=exposure_time,
        telescope_mirror_radius=telescope_mirror_radius,
        telescope_throughput=telescope_throughput,
        instrument_resolving_power=instrument_resolving_power,
        pixel_sampling=pixel_sampling,
        noise_correction_coefficient=1.0,
        scale_factor=1.0,
        star_snr=star_snr,
        star_apparent_magnitude=star_apparent_magnitudes,
        star_snr_reference_apparent_magnitude=star_apparent_magnitude_j,
        mock_observation_number=100,
        transit_number=1
    )

    snrs_ = {}

    for i in range(10):
        print(i)
        snrs_[i], snrs_error_, tsm_, results_ = get_tsm_snr_pcloud(
            band='K',
            wavelength_boundaries=wlen_modes['K'] * 1e-4,
            star_distances=distances,
            p_clouds=p_cloud,
            models=models,
            species_list=species_list,
            settings=settings,
            planet=planet,
            t_int=t_int[0],
            metallicity=metallicity[0],
            co_ratio=co_ratio[0],
            velocity_range=velocity_range,
            exposure_time=exposure_time,
            telescope_mirror_radius=telescope_mirror_radius,
            telescope_throughput=telescope_throughput,
            instrument_resolving_power=instrument_resolving_power,
            pixel_sampling=pixel_sampling,
            noise_correction_coefficient=1.0,
            scale_factor=1.0,
            star_snr=star_snr,
            star_apparent_magnitude=star_apparent_magnitudes,
            star_snr_reference_apparent_magnitude=star_apparent_magnitude_j,
            mock_observation_number=1
        )

    snr = []

    for i in snrs_:
        snr.append(snrs_[i]['2148']['H2O'][0, 0])

    b, x, ax = plt.hist(snr, density=True)
    import scipy.stats
    plt.plot(x, scipy.stats.norm.pdf(x, snrs['2148']['H2O'][0, 0], snrs_error['2148']['H2O'][0, 0]))
    plt.xlabel('CCF S/N')
    plt.title('WASP-39 b, H2O detection, 100x1 (blue) vs 1x100 (orange) obs')


def _test_emission():
    # Base parameters
    planet_name = 'WASP-39 b'
    lbl_opacity_sampling = 1
    do_scat_emis = False
    model_suffix = ''
    wlen_modes = {
        'K': np.array([1.88, 2.55])
    }

    planet = Planet.get(planet_name)
    pressures = np.logspace(-10, 2, 130)
    distances = [213.982 * nc.pc]
    star_apparent_magnitude_j = planet.system_apparent_magnitude_j
    star_apparent_magnitudes = [star_apparent_magnitude_j]

    # Models to be tested
    t_int = [50]
    metallicity = [0]
    co_ratio = [0.55]
    p_cloud = [1e2]
    species_list = ['all', 'H2O']

    mass_fractions = {
        'H2O_main_iso': np.ones_like(pressures) * 1e-2,
        'CO_all_iso': np.ones_like(pressures) * 2e-2,
    }

    line_species_list = [
        'CH4_main_iso',
        'CO_all_iso',
        'CO2_main_iso',
        'H2O_main_iso',
        'H2S_main_iso',
        'HCN_main_iso',
        'K',
        'Na_allard',
        'NH3_main_iso',
        'PH3_main_iso'
    ]

    # Observation parameters
    # Actually matter (used to get the CRIRES SNR data from the ETC website)
    integration_time = 60
    airmass = 1.2
    velocity_range = [-1400, 1400]
    instrument_resolving_power = 8e4
    # Old (don't do anything anymore)
    telescope_mirror_radius = 8.2e2 / 2  # cm
    telescope_throughput = 0.1
    pixel_sampling = 3

    # Load settings
    settings = load_wavelength_settings(module_dir + '/crires/wavelength_settings.dat')
    settings = {'K': {'2148': settings['K']['2148']}}

    # Load planet
    exposure_time = 1000 * 3600
    planet.transit_duration = 500 * 3600

    # Load signal to noise_matrix ratios
    star_snr = get_multiple_crires_snr_data(settings, star_apparent_magnitude_j,  # TODO apparent mag is really annoying
                                            planet.star_effective_temperature, exposure_time,
                                            integration_time, airmass,  # TODO add FLI and seeing
                                            rewrite=False, star_apparent_magnitude_band='J')

    # Generate parameter dictionaries
    parameter_dicts = get_parameter_dicts(
        t_int, metallicity, co_ratio, p_cloud
    )

    # Load/generate relevant models
    models = {}

    for wlen_mode in wlen_modes:
        print(f"Band {wlen_mode}...")

        # Initialize grid
        models[wlen_mode], all_models_exist = init_model_grid(
            planet_name, lbl_opacity_sampling, do_scat_emis, parameter_dicts, species_list,
            wavelength_boundaries=wlen_modes[wlen_mode],
            model_suffix=model_suffix
        )

        if not all_models_exist:
            # Load or generate atmosphere
            atmosphere, atmosphere_filename = SpectralModelLegacy.get_atmosphere_model(
                wlen_bords_micron=wlen_modes[wlen_mode],
                pressures=pressures,
                line_species_list=SpectralModelLegacy.default_line_species,
                rayleigh_species=SpectralModelLegacy.default_rayleigh_species,
                continuum_opacities=SpectralModelLegacy.default_continuum_opacities,
                lbl_opacity_sampling=lbl_opacity_sampling,
                do_scat_emis=do_scat_emis,
                model_suffix=model_suffix
            )

            # Load or generate models
            models[wlen_mode] = generate_model_grid(
                models=models[wlen_mode],
                pressures=pressures,
                line_species_list=line_species_list,
                rayleigh_species='default',
                continuum_opacities='default',
                model_suffix=model_suffix,
                atmosphere=atmosphere,
                mass_fractions=mass_fractions,
                calculate_transmission_spectrum=False,
                calculate_eclipse_depth=True,
                rewrite=True,
                save=True
            )
        else:
            # Load existing models
            models[wlen_mode] = load_model_grid(models[wlen_mode])

    snrs, snrs_error, tsm, results = get_tsm_snr_pcloud(
        band='K',
        wavelength_boundaries=wlen_modes['K'] * 1e-4,
        star_distances=distances,
        p_clouds=p_cloud,
        models=models,
        species_list=species_list,
        settings=settings,
        planet=planet,
        t_int=t_int[0],
        metallicity=metallicity[0],
        co_ratio=co_ratio[0],
        velocity_range=velocity_range,
        exposure_time=exposure_time,
        telescope_mirror_radius=telescope_mirror_radius,
        telescope_throughput=telescope_throughput,
        instrument_resolving_power=instrument_resolving_power,
        pixel_sampling=pixel_sampling,
        noise_correction_coefficient=1.0,
        scale_factor=1.0,
        star_snr=star_snr,
        star_apparent_magnitude=star_apparent_magnitudes,
        star_snr_reference_apparent_magnitude=star_apparent_magnitude_j,
        mock_observation_number=10,
        mode='eclipse'
    )


def load_hdf5(file, **kwargs):
    """
    Load an HDF5 file file.
    :param file: HDF5 file
    :param kwargs: keyword arguments for h5py.file
    :return: the data
    """
    data_dict = h5py.File(file, mode='r', **kwargs)

    return data_dict


def find_best_setting(species, snrs):
    best_snr = - np.inf
    best_band = None
    best_setting = None

    for band in snrs:
        for setting in snrs[band]:
            snr = snrs[band][setting][species]

            if snr > best_snr:
                best_band = band
                best_setting = setting
                best_snr = snr

    return best_snr, best_band, best_setting


if __name__ == '__main__':
    # main_tic()
    main_wasp()