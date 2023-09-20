import copy as cp
import glob

import corner
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # TODO is seaborn really that useful?
from matplotlib.lines import Line2D
from scipy.ndimage import uniform_filter1d
from scipy.stats import binned_statistic

import petitRADTRANS.physical_constants as cst


def _corner_wrap(data_list, title_kwargs, labels_list, label_kwargs, range_list, color_list,
                 truths_list, contour_kwargs, hist_kwargs, quantiles=None, hist2d_levels=None, fig=None,
                 smooth=True, show_titles=True, title_fmt=".2f", truth_color='r', plot_contours=True, **hist2d_kwargs):
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


def plot_specs(fig, ax, path, name, nsample, color1, color2, zorder, rebin_val=None):
    # TODO write generic plotting functions rather than copy pasting code.
    # Deprecated
    specs = sorted([f for f in glob.glob(path + '/' + name + '*.dat')])
    wlen = np.genfromtxt(specs[0])[:, 0]
    if rebin_val is not None:
        wlen = cst.running_mean(wlen, rebin_val)[::rebin_val]
    npoints = int(len(wlen))
    spectra = np.zeros((nsample, npoints))

    for i_s in range(nsample):
        if rebin_val is not None:
            spectra[i_s, :] = uniform_filter1d(np.genfromtxt(specs[i_s])[:, 1],
                                               rebin_val)[::rebin_val]
        else:
            wlen = np.genfromtxt(specs[i_s])[:, 0]

            for i_s2 in range(nsample):  # TODO check this weird loop
                spectra[i_s, :] = np.genfromtxt(specs[i_s2])[:, 1]

    sort_spec = np.sort(spectra, axis=0)
    # 3 sigma
    if int(nsample * 0.02275) > 1:
        ax.fill_between(wlen,
                        y1=sort_spec[int(nsample * 0.02275), :],
                        y2=sort_spec[int(nsample * (1. - 0.02275)), :],
                        color=color1, zorder=zorder * 2)
    # 1 sigma
    ax.fill_between(wlen,
                    y1=sort_spec[int(nsample * 0.16), :],
                    y2=sort_spec[int(nsample * 0.84), :],
                    color=color2, zorder=zorder * 2 + 1)
    return fig, ax


def plot_data(fig, ax, data, resolution=None, scaling=1.0):
    scale = data.scale_factor
    if not data.photometry:
        try:
            # Sometimes this fails, I'm not super sure why.
            resolution_data = np.mean(data.wlen[1:] / np.diff(data.wlen))
            ratio = resolution_data / resolution
            if int(ratio) > 1:
                flux, edges, _ = binned_statistic(data.wlen, data.flux, 'mean',
                                                  data.wlen.shape[0] / ratio)
                error, _, _ = np.array(binned_statistic(data.wlen, data.flux_error,
                                                        'mean', data.wlen.shape[0] / ratio)) / np.sqrt(ratio)
                wlen = np.array([(edges[i] + edges[i + 1]) / 2.0 for i in range(edges.shape[0] - 1)])
            else:
                wlen = data.wlen
                error = data.flux_error
                flux = data.flux
        except Exception:  # TODO find what is the error expected here
            wlen = data.wlen
            error = data.flux_error
            flux = data.flux
    else:
        wlen = np.mean(data.width_photometry)
        flux = data.flux
        error = data.flux_error

    marker = 'o'
    if data.photometry:
        marker = 's'
    if not data.photometry:
        ax.errorbar(wlen,
                    flux * scaling * scale,
                    yerr=error * scaling * scale,
                    marker=marker, markeredgecolor='k', linewidth=0, elinewidth=2,
                    label=data.name, zorder=10, alpha=0.9, )
    else:
        ax.errorbar(wlen,
                    flux * scaling * scale,
                    yerr=error * scaling * scale,
                    xerr=data.wlen_bins / 2., linewidth=0, elinewidth=2,
                    marker=marker, markeredgecolor='k', color='grey', zorder=10,
                    label=None, alpha=0.6)
    return fig, ax


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
    Use the corner package to plot the posterior distributions produced by pymultinest.

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
            contains the indices of the sample to plot, as set by the plot_in_corner
            parameter of the parameter class
        true_values : dict
            A dictionary with keys for each retrieval name as in sampledict. Each value
            contains the known values of the parameters.
        short_name : dict
            A dictionary with keys for each retrieval name as in sampledict. Each value
            contains the names to be plotted in the corner plot legend. If non, uses the
            retrieval names used as keys for sampledict
        quantiles : list
            A list with the quantiles to plot over the 1D histograms.
            Note: the conversion from sigma to quantile is:
                quantile_m = (1 - erf(sigma / sqrt(2))) / 2
                quantile_p = 1 - quantile_m
        hist2d_levels :  list
            A list with the sigmas-level to plot over the 2D histograms. The sigmas are converted into their
            corresponding level-value following the formula for 2D normal distribution
            (see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4358977/pdf/pone.0118537.pdf).
        legend : bool
            Turn the legend on or off
        prt_plot_style : bool
            Use the prt plot style, changes the colour scheme and fonts to match the rest of
            the prt plots.
        plot_best_fit :
            # TODO complete docstring
        color_list :
            color list to use to represent each samples
            # TODO fix color list needing 1 more color than samples
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

        mpl.rcParams.power_update(mpl.rcParamsDefault)
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
                title_fmt=".2f",
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
                N_samples=None,
                parameter_ranges=None,
                parameter_plot_indices=None,
                true_values=None,
                max_val_ratio=None):
    """
    Paul's custom hex grid corner plots.
    Won't work with sampledict setup in retrieve.py!
    """
    font = {'family': 'serif',
            'weight': 'normal',
            'size': int(23 * 5. / len(parameter_plot_indices))}

    plt.rc('font', **font)
    plt.rc('text', usetex=True)

    if N_samples is not None:
        s = N_samples
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
            x = cp.copy(data_list[i_col])
            y = cp.copy(data_list[i_lin])
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

            use_data = cp.copy(data_list[i_col])
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


def plot_radtrans_opacities(radtrans, species, temperature, pressure_bar, mass_fractions=None, co_ratio=0.55,
                            log10_metallicity=0., return_opacities=False, **kwargs):
    import matplotlib.pyplot as plt

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
            line_opacities_temperature_profile_grid=radtrans.lines_loaded_opacities['temperature_profile_grid'],
            has_custom_line_opacities_tp_grid=radtrans.lines_loaded_opacities['has_custom_tp_grid'],
            line_opacities_temperature_grid_size=radtrans.lines_loaded_opacities['temperature_grid_size'],
            line_opacities_pressure_grid_size=radtrans.lines_loaded_opacities['pressure_grid_size']
        )

        _opacities_dict = {}

        weights_gauss = radtrans.lines_loaded_opacities['weights_gauss'].reshape(
            (len(radtrans.lines_loaded_opacities['weights_gauss']), 1, 1)
        )

        for i, s in enumerate(radtrans.line_species):
            _opacities_dict[s] = np.sum(_opacities[:, :, i, :] * weights_gauss, axis=0)

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
        from .chemistry import interpolate_mass_fractions_chemical_table

        mass_fractions = interpolate_mass_fractions_chemical_table(
            co_ratios=co_ratio * np.ones_like(temperatures),
            log10_metallicities=log10_metallicity * np.ones_like(temperatures),
            temperatures=temperatures,
            pressures=pressure_bar
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
