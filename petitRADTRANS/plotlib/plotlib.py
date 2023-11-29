import copy
import os

import corner
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import binned_statistic

import petitRADTRANS.physical_constants as cst
from petitRADTRANS.chemistry.clouds import (
    return_t_cond_fe, return_t_cond_fe_l, return_t_cond_fe_comb, return_t_cond_kcl, return_t_cond_mgsio3,
    return_t_cond_na2s, simple_cdf_fe, simple_cdf_kcl, simple_cdf_mgsio3, simple_cdf_na2s
)
from petitRADTRANS.chemistry.pre_calculated_chemistry import pre_calculated_equilibrium_chemistry_table
from petitRADTRANS.plotlib.style import update_figure_font_size
from petitRADTRANS.retrieval.utils import get_pymultinest_sample_dict


def _corner_wrap(data_list, title_kwargs, labels_list, label_kwargs, range_list, color_list,
                 truths_list, contour_kwargs, hist_kwargs, quantiles=None, hist2d_levels=None, fig=None,
                 smooth=True, show_titles=True, title_fmt=None, truth_color='r', plot_contours=True, **hist2d_kwargs):
    if quantiles is None:
        quantiles = (0.16, 0.5, 0.84)  # using default title_quantiles, the 1-sigma quantile is actually erf(1)

    if hist2d_levels is None:
        hist2d_levels = (1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-4.5))  # 1, 2 and 3 sigma
    else:
        hist2d_levels = [1 - np.exp(-sigma ** 2 / 2) for sigma in hist2d_levels]

    fig = corner.corner(
        data=np.array(data_list).T,
        range=range_list,
        color=color_list,
        smooth=smooth,
        labels=labels_list,
        label_kwargs=label_kwargs,
        show_titles=show_titles,
        title_fmt=title_fmt,
        title_kwargs=title_kwargs,
        truths=truths_list,
        truth_color=truth_color,
        quantiles=quantiles,
        fig=fig,
        hist_kwargs=hist_kwargs,
        **hist2d_kwargs,
        plot_contours=plot_contours,
        contour_kwargs=contour_kwargs,
        levels=hist2d_levels
    )

    return fig


def _get_parameter_range(sample_dict, retrieved_parameters, spectral_model=None):
    parameter_ranges = []
    parameter_names = []
    parameter_titles = []
    parameter_labels = []
    coefficients = []
    offsets = []

    ordered_id = []
    sd_keys = list(sample_dict.keys())
    sd_non_parameter_keys = []

    if 'log_likelihood' in sample_dict:
        sd_non_parameter_keys.append('log_likelihood')

    if 'stats' in sample_dict:
        sd_non_parameter_keys.append('stats')

    for key in sample_dict:
        if key not in retrieved_parameters:
            if key not in sd_non_parameter_keys:
                raise KeyError(f"Key '{key}' not in retrieved parameters")
            else:
                continue

    for i, key in enumerate(retrieved_parameters):
        if key not in sample_dict:
            continue

        ordered_id.append(sd_keys.index(key))
        dictionary = retrieved_parameters[key]

        prior_parameters = np.array(dictionary['prior_parameters'])

        # pRT corner range
        mean = np.mean(sample_dict[key])
        std = np.std(sample_dict[key])
        low_ref = mean - 4 * std
        high_ref = mean + 4 * std

        if 'figure_coefficient' in dictionary:
            if spectral_model is not None:
                if key == 'radial_velocity_semi_amplitude':
                    figure_coefficient = spectral_model.model_parameters['radial_velocity_semi_amplitude']
                elif key == 'planet_radius':
                    figure_coefficient = spectral_model.model_parameters['planet_radius']
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
            if spectral_model is not None:
                if key == 'mid_transit_time':
                    figure_offset = spectral_model.model_parameters['mid_transit_time']
                    prior_parameters += spectral_model.model_parameters['mid_transit_time']
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
        sd_tmp[sd_keys[i]] = copy.deepcopy(sample_dict[sd_keys[i]])

    for k in sd_non_parameter_keys:
        sd_tmp[k] = copy.deepcopy(sample_dict[k])

    sample_dict = copy.deepcopy(sd_tmp)

    return parameter_ranges, parameter_names, parameter_labels, parameter_titles, \
        np.array(coefficients), np.array(offsets), sample_dict


def _prepare_multiple_retrievals_plot(result_directory, retrieved_parameters, true_values, spectral_model=None):
    # Initialisation
    parameter_names_dict = {}
    parameter_plot_indices_dict = {}
    parameter_ranges_dict = {}
    true_values_dict = {}
    fig_titles_dict = {}
    fig_labels_dict = {}
    parameter_names_ref = None

    # Get all samples
    if np.ndim(result_directory) > 0:
        sd = []
        sample_dict = {}

        for i, directory in enumerate(result_directory):
            sd.append(get_pymultinest_sample_dict(directory))
            sample_dict[f'{i}'] = None
    else:
        sd = [get_pymultinest_sample_dict(result_directory)]
        sample_dict = {'': None}

    samples = list(sample_dict.keys())

    # Get all useful figure information from all retrievals
    for i, sample in enumerate(samples):
        parameter_ranges, parameter_names, fig_labels, fig_titles, coefficients, offsets, sd[i] = \
            _get_parameter_range(sd[i], retrieved_parameters, spectral_model)

        sample_dict[sample] = np.array(list(sd[i].values())).T

        sample_dict[sample] = sample_dict[sample] * coefficients + offsets
        parameter_names_dict[sample] = parameter_names
        fig_titles_dict[sample] = fig_titles
        fig_labels_dict[sample] = fig_labels

        # Ensure that all parameter ranges share the widest ranges among all retrievals
        if i == 0:
            parameter_plot_indices_dict[sample] = np.arange(0, len(parameter_ranges))
            parameter_names_ref = np.array(parameter_names_dict[sample])
            parameter_ranges_dict[sample] = parameter_ranges
        else:
            parameter_ranges_dict[sample] = copy.deepcopy(parameter_ranges_dict[samples[0]])

            for j, range_ in enumerate(parameter_ranges):
                if parameter_names_dict[sample][j] not in parameter_names_ref:
                    raise KeyError(f"parameter '{parameter_names_dict[sample][j]}' was not in first sample")

                # Store the widest parameter ranges in the first sample, which contains all possible parameters
                for k, parameter_name in enumerate(parameter_names_dict[samples[0]]):
                    if parameter_name == parameter_names_dict[sample][j]:
                        if range_[0] < parameter_ranges_dict[samples[0]][k][0]:
                            parameter_ranges_dict[samples[0]][k][0] = range_[0]

                        if range_[1] > parameter_ranges_dict[samples[0]][k][1]:
                            parameter_ranges_dict[samples[0]][k][1] = range_[1]

                        break

            # Ensure that all parameters with the same name across retrievals have the same plot index
            parameter_plot_indices_dict[sample] = np.zeros(np.size(parameter_names_dict[sample]), dtype=int)

            for j, parameter_name in enumerate(parameter_names_dict[sample]):
                if parameter_name not in parameter_names_ref:
                    raise KeyError(f"key '{parameter_name}' "
                                   f"of sample '{sample}' not in sample '{parameter_names_ref}'")

                parameter_plot_indices_dict[sample][j] = np.array(parameter_names_ref == parameter_name).nonzero()[0][0]

    # Broadcast the widest parameter ranges stored in the first sample to all the other samples
    for sample in samples[1:]:
        parameter_ranges_dict[sample] = parameter_ranges_dict[samples[0]]

        parameter_ranges_dict_tmp = copy.deepcopy(parameter_ranges_dict[sample])

        for j, plot_indice in enumerate(parameter_plot_indices_dict[sample]):
            parameter_ranges_dict_tmp[plot_indice] = parameter_ranges_dict[sample][j]

        parameter_ranges_dict[sample] = copy.deepcopy(parameter_ranges_dict_tmp)

    # Add true values
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


def contour_corner(sampledict,
                   parameter_names,
                   output_file=None,
                   parameter_ranges=None,
                   parameter_plot_indices=None,
                   true_values=None,
                   short_name=None,
                   quantiles=None,
                   hist2d_levels=None,
                   legend=False,
                   prt_plot_style=True,
                   plot_best_fit=False,
                   color_list=None,
                   **kwargs):
    """
    Use the corner package to plotlib the posterior distributions produced by pymultinest.

    Args:
        sampledict : dict
            A dictionary of samples, each sample has shape (N_Samples,N_params). The keys of the
            dictionary correspond to the names of each retrieval, and are the prefixes to the
            post_equal_weights.dat files. These are passed as arguments to retrieve.py.
            By default, this is only the current retrieval, and plots the posteriors for a single
            retrieval. If multiple names are passed, they are overplotted on the same figure.
        parameter_names : dict
            A dictionary with keys for each retrieval name, as in sampledict. Each value of the
            dictionary is the names of the parameters to beplotted, as set in the
            run_definition file.
        output_file : str
            Output file name
        parameter_ranges : dict
            A dictionary with keys for each retrieval name as in sampledict. Each value
            contains the ranges of parameters that have a range set with corner_range in the
            parameter class. Otherwise, the range is +/- 4 sigma
        parameter_plot_indices : dict
            A dictionary with keys for each retrieval name as in sampledict. Each value
            contains the indices of the sample to plotlib, as set by the plot_in_corner
            parameter of the parameter class
        true_values : dict
            A dictionary with keys for each retrieval name as in sampledict. Each value
            contains the known values of the parameters.
        short_name : dict
            A dictionary with keys for each retrieval name as in sampledict. Each value
            contains the names to be plotted in the corner plotlib legend. If non, uses the
            retrieval names used as keys for sampledict
        quantiles : list
            A list with the quantiles to plotlib over the 1D histograms.
            Note: the conversion from sigma to quantile is:
                quantile_m = (1 - erf(sigma / sqrt(2))) / 2
                quantile_p = 1 - quantile_m
        hist2d_levels :  list
            A list with the sigmas-level to plotlib over the 2D histograms. The sigmas are converted into their
            corresponding level-value following the formula for 2D normal distribution
            (see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4358977/pdf/pone.0118537.pdf).
        legend : bool
            Turn the legend on or off
        prt_plot_style : bool
            Use the prt plotlib style, changes the colour scheme and fonts to match the rest of
            the prt plots.
        plot_best_fit : bool
            Plot the maximum likelihood values as vertical lines for each parameter.
        color_list : list
            List of colours for plotting multiple retrievals.
        kwargs : dict
            Each kwarg can be one of the kwargs used in corner.corner. These can be used to adjust
            the title_kwargs,label_kwargs,hist_kwargs, hist2d_kawargs or the contour kwargs. Each
            kwarg must be a dictionary with the arguments as keys and values as the values.
    """
    if parameter_ranges is None:
        parameter_ranges = {}

    if parameter_plot_indices is None:
        parameter_plot_indices = {}

    if prt_plot_style:
        import matplotlib as mpl

        mpl.rcParams.update(mpl.rcParamsDefault)
        font = {'family': 'serif'}
        xtick = {'top': True,
                 'bottom': True,
                 'direction': 'in'}

        ytick = {'left': True,
                 'right': True,
                 'direction': 'in'}
        xmin = {'visible': True}
        ymin = {'visible': True}
        mpl.rc('xtick', **xtick)
        mpl.rc('xtick.minor', **xmin)
        mpl.rc('ytick', **ytick)
        mpl.rc('ytick.minor', **ymin)
        mpl.rc('font', **font)

        if color_list is None:
            color_list = [
                '#009FB8', '#FF695C', '#70FF92', '#FFBB33', '#6171FF', "#FF1F69", "#52AC25", '#E574FF', "#FF261D",
                "#B429FF"
            ]
    elif color_list is None:
        color_list = [f'C{i}' for i in range(8)]  # standard matplotlib color cycle

    handles = []
    count = 0
    fig = None

    for key, samples in sampledict.items():
        if prt_plot_style and count > len(color_list):
            print("Not enough colors to continue plotting. Please add to the list.")
            print("Outputting first " + str(count) + " retrievals.")
            break

        n_samples = len(samples)
        s = n_samples

        if key not in parameter_plot_indices:
            parameter_plot_indices[key] = range(len(parameter_names[key]))
        elif parameter_plot_indices[key] is None:  # same as in the case the key doesn't exists
            parameter_plot_indices[key] = range(len(parameter_names[key]))

        if key not in parameter_ranges:
            parameter_ranges[key] = [None] * (max(parameter_plot_indices[key]) + 1)

        data_list = []
        labels_list = []

        if plot_best_fit:
            best_fit = []
            best_fit_ind = np.argmax(samples[:, -1])

            for i in parameter_plot_indices[key]:
                best_fit.append(samples[best_fit_ind][i])

        range_list = []

        for range_i, i in enumerate(parameter_plot_indices[key]):
            data_list.append(samples[len(samples) - s:, i])
            labels_list.append(parameter_names[key][i])

            if parameter_ranges[key][i] is None:
                range_mean = np.mean(samples[len(samples) - s:, i])
                range_std = np.std(samples[len(samples) - s:, i])
                low = range_mean - 4 * range_std
                high = range_mean + 4 * range_std

                if count > 0:
                    if low > range_list[range_i][0]:
                        low = range_list[range_i][0]
                    if high < range_list[range_i][1]:
                        high = range_list[range_i][1]
                    range_take = (low, high)
                    range_list[range_i] = range_take
                else:
                    range_list.append((low, high))
            else:
                range_take = (parameter_ranges[key][i][0], parameter_ranges[key][i][1])
                range_list.append(range_take)

        if parameter_plot_indices is not None and true_values is not None:
            truths_list = []

            if plot_best_fit:
                best_fit_ind = np.argmax(samples[:, -1])

                for i in parameter_plot_indices[key]:
                    truths_list.append(samples[best_fit_ind][i])
            else:
                for i in parameter_plot_indices[key]:
                    truths_list.append(true_values[key][i])
        else:
            truths_list = None

        label_kwargs = None
        title_kwargs = None
        hist_kwargs = None
        hist2d_kwargs = {}
        contour_kwargs = None

        if "label_kwargs" in kwargs.keys():
            label_kwargs = kwargs["label_kwargs"]
        if "title_kwargs" in kwargs.keys():
            title_kwargs = kwargs["title_kwargs"]
        if "hist_kwargs" in kwargs.keys():
            hist_kwargs = kwargs["hist_kwargs"]
        if "hist2d_kwargs" in kwargs.keys():
            hist2d_kwargs = kwargs["hist2d_kwargs"]
        if "contour_kwargs" in kwargs.keys():
            contour_kwargs = kwargs["contour_kwargs"]

        if count == 0:
            fig = _corner_wrap(
                data_list=data_list,
                title_kwargs=title_kwargs,
                labels_list=labels_list,
                label_kwargs=label_kwargs,
                range_list=range_list,
                color_list=color_list[count],
                hist_kwargs=hist_kwargs,
                truths_list=truths_list,
                contour_kwargs=contour_kwargs,
                quantiles=quantiles,
                hist2d_levels=hist2d_levels,
                **hist2d_kwargs,
                smooth=True,
                show_titles=True,
                title_fmt=".2f",
                truth_color='r',
                plot_contours=True
            )
            count += 1
        else:
            _ = _corner_wrap(
                data_list=data_list,
                title_kwargs=title_kwargs,
                labels_list=labels_list,
                label_kwargs=label_kwargs,
                range_list=range_list,
                color_list=color_list[count],
                hist_kwargs=hist_kwargs,
                truths_list=truths_list,
                contour_kwargs=contour_kwargs,
                quantiles=quantiles,
                hist2d_levels=hist2d_levels,
                fig=fig,
                **hist2d_kwargs,
                smooth=True,
                show_titles=False,  # only show titles (median +1sigma -1sigma) for the first sample
                title_fmt=None,
                truth_color='r',
                plot_contours=True
            )
            count += 1

        if short_name is None:
            label = key
        else:
            label = short_name[key]

        handles.append(Line2D([0], [0], marker='o', color=color_list[count], label=label, markersize=15))

    if legend:
        fig.get_axes()[2].legend(handles=handles,
                                 loc='upper right')

    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    return fig


def nice_corner(samples,
                parameter_names,
                output_file,
                n_samples=None,
                parameter_ranges=None,
                parameter_plot_indices=None,
                true_values=None,
                max_val_ratio=None):
    """
    Paul's custom hex grid corner plots.
    Won't work with sampledict setup in retrieve.py!
    """
    import seaborn as sns  # TODO is seaborn really that useful?

    font = {'family': 'serif',
            'weight': 'normal',
            'size': int(23 * 5. / len(parameter_plot_indices))}

    plt.rc('font', **font)
    plt.rc('text', usetex=True)

    if n_samples is not None:
        s = n_samples
    else:
        s = len(samples)

    try:
        if parameter_plot_indices is None:
            parameter_plot_indices = np.linspace(0, len(parameter_names) - 1,
                                                 len(parameter_names) - 1).astype('int')
    except Exception:
        pass

    if max_val_ratio is None:
        max_val_ratio = 5.

    data_list = []
    labels_list = []
    range_list = []

    for i in parameter_plot_indices:

        data_list.append(samples[len(samples) - s:, i])
        labels_list.append(parameter_names[i])

        try:
            if parameter_ranges[i] is None:
                range_mean = np.mean(samples[len(samples) - s:, i])
                range_std = np.std(samples[len(samples) - s:, i])
                range_take = (range_mean - 4 * range_std, range_mean + 4 * range_std)
                range_list.append(range_take)
            else:
                range_list.append(parameter_ranges[i])
        except Exception:
            range_mean = np.mean(samples[len(samples) - s:, i])
            range_std = np.std(samples[len(samples) - s:, i])
            range_take = (range_mean - 4 * range_std, range_mean + 4 * range_std)
            range_list.append(range_take)

    try:
        truths_list = []

        for i in parameter_plot_indices:
            truths_list.append(true_values[i])
    except Exception:
        truths_list = None

    dimensions = len(parameter_plot_indices)

    # Set up the matplotlib figure
    f, axes = plt.subplots(dimensions, dimensions, figsize=(13, 13),
                           sharex='none', sharey='none')
    i_col = 0
    i_lin = 0

    try:
        ax_array = axes.flat
    except Exception:
        ax_array = [plt.gca()]

    for ax in ax_array:

        # print(i_col, i_lin, dimensions, len(axes.flat))
        if i_col < i_lin:
            x = copy.copy(data_list[i_col])
            y = copy.copy(data_list[i_lin])
            indexx = (x >= range_list[i_col][0]) & \
                     (x <= range_list[i_col][1])
            indexy = (y >= range_list[i_lin][0]) & \
                     (y <= range_list[i_lin][1])

            x = x[indexx & indexy]
            y = y[indexx & indexy]

            gridsize = 22
            ax.hexbin(x, y, cmap='bone_r', gridsize=gridsize,
                      vmax=int(max_val_ratio * s / gridsize ** 2.),
                      rasterized=True)

            ax.set_xlim([range_list[i_col][0], range_list[i_col][1]])
            ax.set_ylim([range_list[i_lin][0], range_list[i_lin][1]])

            try:
                ax.axhline(truths_list[i_lin], color='red',
                           linestyle='--', linewidth=2.5)
                ax.axvline(truths_list[i_col], color='red',
                           linestyle='--', linewidth=2.5)
            except Exception:
                pass

            if i_col > 0:
                ax.get_yaxis().set_visible(False)

        elif i_col == i_lin:

            med = np.median(data_list[i_col])
            up = str(np.round(np.percentile(data_list[i_col], 84) - med, 2))
            do = str(np.round(med - np.percentile(data_list[i_col], 16), 2))
            med = str(np.round(med, 2))
            med = med.split('.')[0] + '.' + med.split('.')[1].ljust(2, '0')
            up = up.split('.')[0] + '.' + up.split('.')[1].ljust(2, '0')
            do = do.split('.')[0] + '.' + do.split('.')[1].ljust(2, '0')

            ax.set_title(med + r'$^{+' + up + '}_{-' + do + '}$',
                         fontdict=font, fontsize=int(22 * 5. / len(parameter_plot_indices)))

            use_data = copy.copy(data_list[i_col])
            index = (use_data >= range_list[i_col][0]) & \
                    (use_data <= range_list[i_col][1])
            use_data = use_data[index]
            sns.distplot(use_data, bins=22, kde=False,
                         rug=False, ax=ax, color='gray')

            ax.set_xlim([range_list[i_col][0], range_list[i_col][1]])
            ax.get_yaxis().set_visible(False)
            try:
                ax.axvline(truths_list[i_col], color='red',
                           linestyle='--', linewidth=2.5)
            except Exception:
                pass

            ax.axvline(float(med) + float(up), color='black',
                       linestyle=':', linewidth=1.0)
            ax.axvline(float(med), color='black',
                       linestyle=':', linewidth=1.0)
            ax.axvline(float(med) - float(do), color='black',
                       linestyle=':', linewidth=1.0)

        else:
            ax.axis('off')

        i_col += 1
        if i_col % dimensions == 0:
            i_col = 0
            i_lin += 1

    for i_col in range(dimensions):
        for i_lin in range(dimensions):
            try:
                plt.sca(axes[i_lin, i_col])
            except Exception:
                pass
            range_use = np.linspace(range_list[i_col][0],
                                    range_list[i_col][1], 5)[1:-1]

            labels_use = []
            for r in range_use:
                labels_use.append(str(np.round(r, 1)))
            plt.xticks(range_use[::2], labels_use[::2])
            plt.xlabel(labels_list[i_col])

    for i_lin in range(dimensions):
        try:
            plt.sca(axes[i_lin, 0])
        except Exception:
            pass
        plt.ylabel(labels_list[i_lin])

    plt.subplots_adjust(wspace=0, hspace=0)
    if dimensions == 1:
        plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig(output_file)
    # plt.show()
    # plt.clf()


def plot_cloud_condensation_curves(metallicities, co_ratios, pressures=None, temperatures=None):
    for metallicity in metallicities:
        for co_ratio in co_ratios:
            p, t = return_t_cond_fe(metallicity, co_ratio)
            plt.plot(t, p, label='Fe(c), [Fe/H] = ' + str(metallicity) + ', C/O = ' + str(co_ratio), color='black')
            p, t = return_t_cond_fe_l(metallicity, co_ratio)
            plt.plot(t, p, '--', label='Fe(l), [Fe/H] = ' + str(metallicity) + ', C/O = ' + str(co_ratio))
            p, t = return_t_cond_fe_comb(metallicity, co_ratio)
            plt.plot(t, p, ':', label='Fe(c+l), [Fe/H] = ' + str(metallicity) + ', C/O = ' + str(co_ratio))
            p, t = return_t_cond_mgsio3(metallicity, co_ratio)
            plt.plot(t, p, label='MgSiO3, [Fe/H] = ' + str(metallicity) + ', C/O = ' + str(co_ratio))
            p, t = return_t_cond_na2s(metallicity, co_ratio)
            plt.plot(t, p, label='Na2S, [Fe/H] = ' + str(metallicity) + ', C/O = ' + str(co_ratio))
            p, t = return_t_cond_kcl(metallicity, co_ratio)
            plt.plot(t, p, label='KCL, [Fe/H] = ' + str(metallicity) + ', C/O = ' + str(co_ratio))

    plt.yscale('log')
    plt.xlim([0., 2000.])
    plt.ylim([1e2, 1e-3])
    plt.legend(loc='best', frameon=False)
    plt.show()

    if pressures is not None or temperatures is not None:
        simple_cdf_fe(pressures, temperatures, 0., 0.55)
        simple_cdf_mgsio3(pressures, temperatures, 0., 0.55)
        simple_cdf_na2s(pressures, temperatures, 0., 0.55)
        simple_cdf_kcl(pressures, temperatures, 0., 0.55)


def plot_data(fig, ax, data, resolution=None, scaling=1.0):
    scale = data.scale_factor

    if not data.photometry:
        try:
            # Sometimes this fails, I'm not super sure why.
            resolution_data = np.mean(data.wavelengths[1:] / np.diff(data.wavelengths))
            ratio = resolution_data / resolution
            if int(ratio) > 1:
                flux, edges, _ = binned_statistic(data.wavelengths, data.spectrum, 'mean',
                                                  data.wavelengths.shape[0] / ratio)
                error, _, _ = np.array(binned_statistic(data.wavelengths, data.uncertainties,
                                                        'mean', data.wavelengths.shape[0] / ratio)) / np.sqrt(ratio)
                wlen = np.array([(edges[i] + edges[i + 1]) / 2.0 for i in range(edges.shape[0] - 1)])
            else:
                wlen = data.wavelengths
                error = data.uncertainties
                flux = data.spectrum
        except Exception:  # TODO find what is the error expected here
            wlen = data.wavelengths
            error = data.uncertainties
            flux = data.spectrum
    else:
        wlen = np.mean(data.photometric_bin_edges)
        flux = data.spectrum
        error = data.uncertainties

    marker = 'o'

    if data.photometry:
        marker = 's'

    if not data.photometry:
        ax.errorbar(wlen,
                    flux * scaling * scale,
                    yerr=error * scaling * scale,
                    marker=marker, markeredgecolor='k', linewidth=0, elinewidth=2,
                    label=data.name, zorder=10, alpha=0.9)
    else:
        ax.errorbar(wlen,
                    flux * scaling * scale,
                    yerr=error * scaling * scale,
                    xerr=data.wavelength_bin_widths / 2., linewidth=0, elinewidth=2,
                    marker=marker, markeredgecolor='k', color='grey', zorder=10,
                    label=None, alpha=0.6)

    return fig, ax


def plot_multiple_posteriors(result_directory, retrieved_parameters, log_evidences=None, true_values=None,
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
        fig_titles, _ = _prepare_multiple_retrievals_plot(result_directory, retrieved_parameters, true_values)

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

                        plot_posterior(
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


def plot_posterior(data, label=None, true_value=None, cmp=None, bins=15, color='C0',
                   axe=None, y_max=None, y_label=None, tight_layout=True, fmt='.2f'):
    if axe is None:
        fig, axe = plt.subplots(1, 1)

    median = np.median(data)
    sm = np.quantile(data, 0.16)
    sp = np.quantile(data, 0.84)

    c = axe.hist(data, bins=bins, histtype='step', color=color, density=True)

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


def plot_radtrans_opacities(radtrans, species, temperature, pressure_bar, mass_fractions=None, co_ratio=0.55,
                            log10_metallicity=0., return_opacities=False, **kwargs):
    def __compute_opacities(_pressures, _temperatures):
        """ Method to calculate and return the line opacities (assuming an abundance
        of 100% for the individual species) of the Radtrans object. This method
        updates the line_struc_kappas attribute within the Radtrans class. For the
        low resolution (`c-k`) mode, the wavelength-mean within every frequency bin
        is returned.

            Args:
                _temperatures:
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
        _opacities = radtrans._interpolate_species_opacities(
            pressures=_pressures,
            temperatures=_temperatures,
            n_g=radtrans.lines_loaded_opacities['g_gauss'].size,
            n_frequencies=radtrans.frequencies.size,
            line_opacities_grid=radtrans.lines_loaded_opacities['opacity_grid'],
            line_opacities_temperature_pressure_grid=radtrans.lines_loaded_opacities['temperature_pressure_grid'],
            has_custom_line_opacities_tp_grid=radtrans.lines_loaded_opacities['has_custom_tp_grid'],
            line_opacities_temperature_grid_size=radtrans.lines_loaded_opacities['temperature_grid_size'],
            line_opacities_pressure_grid_size=radtrans.lines_loaded_opacities['pressure_grid_size']
        )

        _opacities_dict = {}

        weights_gauss = radtrans.lines_loaded_opacities['weights_gauss'].reshape(
            (len(radtrans.lines_loaded_opacities['weights_gauss']), 1, 1)
        )

        for i, _species in enumerate(radtrans.line_species):
            _opacities_dict[_species] = np.sum(_opacities[:, :, i, :] * weights_gauss, axis=0)

        return cst.c / radtrans.frequencies, _opacities_dict

    temperatures = np.array(temperature)
    pressure_bar = np.array(pressure_bar)

    temperatures = temperatures.reshape(1)
    pressure_bar = pressure_bar.reshape(1)

    pressures = pressure_bar * 1e6

    wavelengths, opacities = __compute_opacities(pressures, temperatures)
    wavelengths *= 1e4  # cm to um

    opacities_weights = {}

    if mass_fractions is None:
        for s in species:
            opacities_weights[s] = 1.
    elif mass_fractions == 'eq':
        mass_fractions = pre_calculated_equilibrium_chemistry_table.interpolate_mass_fractions(
            co_ratios=co_ratio * np.ones_like(temperatures),
            log10_metallicities=log10_metallicity * np.ones_like(temperatures),
            temperatures=temperatures,
            pressures=pressure_bar,
            full=False
        )

        for s in species:
            opacities_weights[s] = mass_fractions[s.split('_')[0]]
    else:
        for s in species:
            opacities_weights[s] = mass_fractions[s]

    if return_opacities:
        opacities_dict = {}

        for s in species:
            opacities_dict[s] = [
                wavelengths,
                opacities_weights[s] * opacities[s]
            ]

        return opacities_dict
    else:
        for s in species:
            plt.plot(
                wavelengths,
                opacities_weights[s] * opacities[s],
                label=s,
                **kwargs
            )


def plot_result_corner(retrieval_directory, retrieved_parameters,
                       true_values=None, spectral_model=None, figure_font_size=8,
                       save=True, figure_directory='./', figure_name='result_corner', image_format='png', **kwargs):
    """Plot the posteriors of one or multiple retrievals in the same corner plot.

    Args:
        retrieval_directory:
            String or list of string containing the retrieval directories
        retrieved_parameters:
            Dictionary containing all the retrieved parameters and their prior
            # TODO retrieved parameters should be automatically stored in retrievals
        true_values:
            True values for the parameters
        spectral_model:
            SpectralModel used to make the retrievals
        figure_font_size:
            Size of the font in the figure
        save:
            If True, save the figure
        figure_directory:
            Directory to save the figure
        figure_name:
            Name of the figure
        image_format:
            Format (extension) of the figure
        **kwargs:
            contour_corner keyword arguments
    """
    (sample_dict, parameter_names_dict, parameter_plot_indices_dict, parameter_ranges_dict, true_values_dict,
        fig_titles, fig_labels) = _prepare_multiple_retrievals_plot(
        result_directory=retrieval_directory,
        retrieved_parameters=retrieved_parameters,
        true_values=true_values,
        spectral_model=spectral_model
    )

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
