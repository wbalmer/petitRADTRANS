import copy
import os
import warnings

import numpy as np
from scipy.stats import binned_statistic

import petitRADTRANS.physical_constants as cst
from petitRADTRANS.math import running_mean
from petitRADTRANS.opacities.opacities import Opacity
from petitRADTRANS.physics import wavelength2frequency
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.radtrans_core.spectrum_utils import filter_spectrum_with_spline
from petitRADTRANS.retrieval.data import PhotometryData
from petitRADTRANS.retrieval.parameter import Parameter


class RetrievalPlotter:
    """Plotting companion for :class:`Retrieval`.

    The plotter owns the plotting-related configuration that used to live in
    the retrieval configuration object. These attributes can be adjusted after
    construction through ``retrieval.plotter`` and are used by the plotting and
    model-evaluation helpers throughout the retrieval module.

    Attributes:
        spec_xlabel : str
            Label for the wavelength axis in spectrum plots.
        spec_ylabel : str
            Label for the flux axis in spectrum plots.
        y_axis_scaling : float
            Multiplicative scaling applied when plotting spectra and
            contribution-like outputs.
        xscale : str
            X-axis scaling used in spectral plots.
        yscale : str
            Y-axis scaling used in spectral plots.
        resolution : float
            Plotting resolution used when rebinned observational data are shown.
        nsample : int | float
            Default number of posterior samples used in sampled plotting
            utilities.
        temp_limits : list[float] | tuple[float, float] | None
            Optional x-axis limits for pressure-temperature plots.
        press_limits : list[float] | tuple[float, float] | None
            Optional y-axis limits for pressure-temperature plots.
        flux_lim : list[float] | tuple[float, float] | None
            Optional y-axis limits for spectral plots.
        wavelength_lim : list[float] | tuple[float, float] | None
            Optional x-axis limits for spectral plots.
        reference_data_name : str | None
            Name of the dataset used as the default plotting reference. If not
            provided at construction time, it defaults to the first key in
            ``retrieval.configuration.data``.
    """

    _local_attributes = {
        "retrieval",
        "_local_attributes",
        "spec_xlabel",
        "spec_ylabel",
        "y_axis_scaling",
        "xscale",
        "yscale",
        "resolution",
        "nsample",
        "temp_limits",
        "press_limits",
        "flux_lim",
        "wavelength_lim",
        "reference_data_name",
    }

    def __init__(self, retrieval, reference_data_name=None):
        resolved_reference_data_name = reference_data_name

        object.__setattr__(self, "retrieval", retrieval)
        object.__setattr__(self, "spec_xlabel", "Wavelength [micron]")
        object.__setattr__(self, "spec_ylabel", "Flux [W/m2/micron]")
        object.__setattr__(self, "y_axis_scaling", 1.0)
        object.__setattr__(self, "xscale", "linear")
        object.__setattr__(self, "yscale", "linear")
        object.__setattr__(self, "resolution", None)
        object.__setattr__(self, "nsample", 10.0)
        object.__setattr__(self, "temp_limits", [150, 3000])
        object.__setattr__(self, "press_limits", [1e2, 1e-5])
        object.__setattr__(self, "flux_lim", None)
        object.__setattr__(self, "wavelength_lim", None)
        object.__setattr__(
            self,
            "reference_data_name",
            (
                resolved_reference_data_name
                if resolved_reference_data_name is not None
                else next(iter(self.configuration.data), None)
            ),
        )

    def __getattr__(self, name):
        return getattr(self.retrieval, name)

    def __setattr__(self, name, value):
        if (
            name in self._local_attributes
            or name.startswith("_")
            or hasattr(type(self), name)
        ):
            object.__setattr__(self, name, value)
            return

        setattr(self.retrieval, name, value)

    def get_reference_data_name(self):
        if self.reference_data_name is not None:
            return self.reference_data_name

        if len(self.configuration.data) == 0:
            raise ValueError("retrieval configuration does not contain any data sets")

        return next(iter(self.configuration.data))

    def get_reference_data(self):
        return self.configuration.data[self.get_reference_data_name()]

    def get_reference_radtrans_name(self):
        data = self.get_reference_data()
        return data.external_radtrans_reference or data.name

    def get_reference_radtrans_data(self):
        return self.configuration.data[self.get_reference_radtrans_name()]

    def _is_plotting_root(self):
        return self.comm is None or self.rank == 0

    def _synchronize(self):
        if self.comm is not None:
            self.comm.barrier()

    @staticmethod
    def _get_clipped_contribution_output(auxiliary_outputs):
        if auxiliary_outputs is None:
            raise ValueError("best-fit model did not return any auxiliary outputs")

        contribution = None

        if isinstance(auxiliary_outputs, dict):
            for key in (
                "emission_contribution",
                "transmission_contribution",
                "contribution",
            ):
                if auxiliary_outputs.get(key) is not None:
                    contribution = auxiliary_outputs[key]
                    break
        else:
            contribution = auxiliary_outputs

        if contribution is None:
            raise ValueError(
                "best-fit model auxiliary outputs do not contain a contribution array"
            )

        contribution = np.asarray(contribution, dtype=float)
        contribution = contribution.copy()
        contribution[~np.isfinite(contribution)] = 1e-16

        return np.clip(contribution, 1e-16, None)

    def plot_abundances(
        self,
        samples_use,
        parameters_read,
        species_to_plot=None,
        contribution=False,
        refresh=True,
        model_generating_function=None,
        prt_reference=None,
        mode="bestfit",
        sample_posteriors=False,
        volume_mixing_ratio=False,
    ):
        """
        Plot the abundance profiles in mass fractions or volume mixing ratios as a function of pressure.

        Args:
            samples_use : numpy.ndarray
                An array of the samples from the post_equal_weights file, used to find the best fit sample
            parameters_read : list
                A list of the free parameters as read from the output files.
            species_to_plot : list
                A list of which molecular species to include in the plot.
            contribution : bool
                If true, overplot the emission or transmission contribution function.
            prt_reference : str
                If specified, the pRT object of the data with name pRT_reference will be used for plotting,
                instead of generating a new pRT object at R = 1000.
            model_generating_function : (callable, optional):
                A function that returns the wavelength and spectrum, and takes a Radtrans object and the
                current set of parameters stored in self.configuration.parameters. This should be the same model
                function used in the retrieval.
            refresh : bool
                If True (default value) the .npy files in the evaluate_[retrieval_name] folder will be replaced
                by recalculating the best fit model. This is useful if plotting intermediate results from a
                retrieval that is still running. If False no new spectrum will be calculated and the plot will
                be generated from the .npy files in the evaluate_[retrieval_name] folder.
            mode : str
                'bestfit' or 'median', indicating which set of values should be used for plotting the abundances.
            sample_posteriors : bool
                If true, sample the posterior distributions to calculate confidence intervals for the retrieved
                abundance profiles.
            volume_mixing_ratio : bool
                If true, plot in units of volume mixing ratio (number fraction) instead of mass fractions.

        Returns:
            fig : matplotlib.figure
                The matplotlib figure, containing the data, best fit spectrum and residuals.
            ax : matplotlib.axes
                The upper pane of the plot, containing the best fit spectrum and data.
            ax_r : matplotlib.axes
                The lower pane of the plot, containing the residuals between the fit and the data.
        """
        import matplotlib.pyplot as plt

        if self._is_plotting_root():
            print("\nPlotting Abundances profiles")

            p_global_keep = self.configuration.pressures
            min_xaxis = 1e-7
            if mode.strip("-").strip("_").lower() == "bestfit":
                log_l, best_fit_index = self.get_best_fit_likelihood(samples_use)
                self.build_param_dict(samples_use[:-1, best_fit_index], parameters_read)
                sample_use = samples_use[:-1, best_fit_index]
            elif mode.lower() == "median":
                med_params, sample_use = self.get_median_params(
                    samples_use, parameters_read, return_array=True
                )
            else:
                sample_use = None

            results = self.calculate_temperature_profile(sample_use)[self.reference_data_name]
            pressures, t = results

            if species_to_plot is None:
                species_to_plot = (
                    self.get_reference_radtrans_data().radtrans_object.line_species
                )

            prop_cycle = plt.rcParams["axes.prop_cycle"]
            colors = prop_cycle.by_key()["color"]

            fig, ax = plt.subplots(figsize=(12, 7))

            if sample_posteriors:
                abundances = {}
                for species in species_to_plot:
                    species = Opacity([species]).get_full_name().split(".R")[0]
                    abundances[species] = []

                for sample in samples_use.T:
                    if volume_mixing_ratio:
                        abund_dict, mmw = self.get_volume_mixing_ratios(
                            sample[:-1], parameters_read
                        )
                    else:
                        abund_dict, mmw = self.get_mass_fractions(
                            sample[:-1], parameters_read
                        )
                    for species in species_to_plot:
                        species = Opacity([species]).get_full_name().split(".R")[0]
                        abundances[species].append(abund_dict[species])

                for i, species in enumerate(species_to_plot):
                    species = Opacity([species]).get_full_name().split(".R")[0]
                    low, med, high = np.quantile(
                        np.array(abundances[species]), [0.159, 0.5, 0.841], axis=0
                    )
                    ax.plot(
                        med,
                        pressures,
                        label=Opacity.get_species_scientific_name(species),
                        color=colors[i % len(colors)],
                        zorder=0,
                        linewidth=2,
                    )
                    ax.plot(
                        low,
                        pressures,
                        color=colors[i % len(colors)],
                        linewidth=0.4,
                        zorder=0,
                    )
                    ax.plot(
                        high,
                        pressures,
                        color=colors[i % len(colors)],
                        linewidth=0.4,
                        zorder=0,
                    )

                    ax.fill_betweenx(
                        pressures,
                        x1=low,
                        x2=high,
                        color=colors[i % len(colors)],
                        alpha=0.15,
                        zorder=-1,
                    )
            else:
                if volume_mixing_ratio:
                    abund_dict, mmw = self.get_volume_mixing_ratios(
                        sample_use, parameters_read
                    )
                else:
                    abund_dict, mmw = self.get_mass_fractions(
                        sample_use, parameters_read
                    )
                for i, species in enumerate(species_to_plot):
                    species = Opacity([species]).get_full_name().split(".R")[0]
                    if np.min(abund_dict[species]) < min_xaxis:
                        min_xaxis = 0.9 * np.min(abund_dict[species])
                    ax.plot(
                        abund_dict[species],
                        pressures,
                        label=Opacity.get_species_scientific_name(species),
                        color=colors[i % len(colors)],
                        zorder=0,
                        linewidth=2,
                    )

            adaptive_mesh_refinement = self.configuration.adaptive_mesh_refinement
            self.configuration.adaptive_mesh_refinement = False
            pressures = self.configuration.pressures
            if contribution:
                (
                    best_fit_wavelengths,
                    best_fit_spectrum,
                    best_fit_auxiliary_outputs,
                ) = (
                    self.get_best_fit_model(
                        sample_use,
                        parameters_read,
                        model_generating_function=model_generating_function,
                        prt_reference=prt_reference,
                        refresh=refresh,
                        contribution=True,
                    )
                )
                best_fit_contribution = self._get_clipped_contribution_output(
                    best_fit_auxiliary_outputs
                )

                nu = wavelength2frequency(best_fit_wavelengths)
                mean_diff_nu = -np.diff(nu)
                diff_nu = np.zeros_like(nu)
                diff_nu[:-1] = mean_diff_nu
                diff_nu[-1] = diff_nu[-2]
                spectral_weights = (
                    best_fit_spectrum * diff_nu / np.sum(best_fit_spectrum * diff_nu)
                )

                if self.test_plotting:
                    plt.clf()
                    plt.plot(best_fit_wavelengths * 1e4, spectral_weights)
                    plt.show()
                    print(np.shape(best_fit_contribution))

                pressure_weights = np.diff(np.log10(pressures))
                weights = np.ones_like(pressures)
                weights[:-1] = pressure_weights
                weights[-1] = weights[-2]
                weights = weights / np.sum(weights)
                weights = weights.reshape(len(weights), 1)

                contr_em = best_fit_contribution / weights
                contr_em = contr_em * spectral_weights[None, :]
                contr_em = np.sum(contr_em, axis=1)
                contr_em = contr_em / np.sum(contr_em)

                if self.test_plotting:
                    plt.clf()
                    plt.yscale("log")
                    plt.ylim([pressures[-1], pressures[0]])
                    plt.plot(contr_em, pressures)
                    plt.show()

                contr_em_weigh = contr_em / np.max(contr_em)
                from scipy.interpolate import interp1d

                contr_em_weigh_intp = interp1d(pressures, contr_em_weigh)
                yborders = pressures
                for i_p in range(len(yborders) - 1):
                    mean_press = (yborders[i_p + 1] + yborders[i_p]) / 2.0
                    ax.fill_between(
                        [min_xaxis, 3],
                        yborders[i_p + 1],
                        yborders[i_p],
                        color="white",
                        alpha=min(1.0 - contr_em_weigh_intp(mean_press), 0.9),
                        linewidth=0,
                        rasterized=True,
                        zorder=100,
                    )

                ax.plot(
                    contr_em_weigh * (3 - min_xaxis) + min_xaxis,
                    pressures,
                    "--",
                    color="black",
                    linewidth=1.0,
                    zorder=120,
                    label="Contribution",
                )
            xlabel = "Mass Fraction Abundance"
            if volume_mixing_ratio:
                xlabel = "Volume Mixing Ratio"

            ax.set_xlabel(xlabel)
            ax.set_ylabel("Pressure [bar]")
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.tick_params(axis="both", which="major", labelsize=16)
            ax.tick_params(axis="both", which="minor", labelsize=12)

            ax.invert_yaxis()
            ax.set_xlim(8e-13, 3)
            ax.set_axisbelow(False)
            ax.tick_params(zorder=2)
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=18)
            plt.tight_layout()

            if not sample_posteriors:
                plt.savefig(
                    self.get_base_figure_name()
                    + "_"
                    + mode
                    + "_abundance_profiles.pdf",
                    bbox_inches="tight",
                )
            else:
                plt.savefig(
                    self.get_base_figure_name() + "_sampled_abundance_profiles.pdf",
                    bbox_inches="tight",
                )
            self.configuration.pressures = p_global_keep
            self.configuration.adaptive_mesh_refinement = adaptive_mesh_refinement
        else:
            fig = None
            ax = None

        self._synchronize()

        return fig, ax

    def plot_all(
        self,
        output_directory=None,
        ret_names=None,
        contribution=False,
        model_generating_function=None,
        prt_reference=None,
        mode="bestfit",
    ):
        """
        Produces plots for the best fit spectrum, a sample of 100 output spectra,
        the best fit PT profile and a corner plot for parameters specified in the
        run definition.
        """
        if self._is_plotting_root():
            if ret_names is None:
                ret_names = []

            if not self.configuration.run_mode == "evaluate":
                print("Not in evaluate mode. Changing run mode to evaluate.")
                self.configuration.run_mode = "evaluate"

            if output_directory is None:
                output_directory = self.output_directory

            sample_dict, parameter_dict = self.get_samples(
                names=self.corner_files,
                output_directory=output_directory,
                ret_names=ret_names,
            )

            samples_use = copy.copy(sample_dict[self.configuration.retrieval_name])
            parameters_read = copy.copy(
                parameter_dict[self.configuration.retrieval_name]
            )
            i_p = 0

            for pp in self.configuration.parameters:
                if self.configuration.parameters[pp].is_free_parameter:
                    for i_s in range(len(parameters_read)):
                        if (
                            parameters_read[i_s]
                            == self.configuration.parameters[pp].name
                        ):
                            samples_use[i_p, :] = sample_dict[
                                self.configuration.retrieval_name
                            ][i_s, :]
                    i_p += 1

            print("Best fit parameters")
            i_p = 0
            log_l, best_fit_index = self.get_best_fit_likelihood(samples_use)

            for pp in self.configuration.parameters:
                if self.configuration.parameters[pp].is_free_parameter:
                    for i_s in range(len(parameters_read)):
                        if (
                            parameters_read[i_s]
                            == self.configuration.parameters[pp].name
                        ):
                            print(
                                self.configuration.parameters[pp].name,
                                samples_use[i_p, best_fit_index],
                            )
                            i_p += 1

            self.plot_spectra(
                samples_use,
                parameters_read,
                refresh=True,
                model_generating_function=model_generating_function,
                prt_reference=prt_reference,
                mode=mode,
            )

            if self.evaluate_sample_spectra:
                self.plot_sampled(
                    samples_use,
                    parameters_read,
                    model_generating_function=model_generating_function,
                    prt_reference=prt_reference,
                )

            self.plot_pt(
                sample_dict,
                parameters_read,
                contribution=contribution,
                model_generating_function=model_generating_function,
                prt_reference=prt_reference,
                mode=mode,
                refresh=False,
            )
            self.plot_corner(sample_dict, parameter_dict, parameters_read)

            if contribution:
                self.plot_contribution(
                    samples_use,
                    parameters_read,
                    model_generating_function=model_generating_function,
                    prt_reference=prt_reference,
                    mode=mode,
                    refresh=False,
                )

            self.plot_abundances(
                samples_use,
                parameters_read,
                contribution=contribution,
                model_generating_function=model_generating_function,
                prt_reference=prt_reference,
                mode=mode,
                refresh=False,
            )
            print("Finished generating all plots!")

        self._synchronize()
        return

    def plot_contribution(
        self,
        samples_use,
        parameters_read,
        model_generating_function=None,
        prt_reference=None,
        log_scale_contribution=False,
        n_contour_levels=30,
        refresh=True,
        mode="bestfit",
    ):
        """
        Plot the contribution function of the bestfit or median model from a retrieval.
        """
        import matplotlib.pyplot as plt

        if self._is_plotting_root():

            self.evaluate_sample_spectra = False
            if not self.configuration.run_mode == "evaluate":
                print("Not in evaluate mode. Changing run mode to evaluate.")
                self.configuration.run_mode = "evaluate"
            print("\nPlotting Best-fit contribution function")

            adaptive_mesh_refinement = self.configuration.adaptive_mesh_refinement
            self.configuration.adaptive_mesh_refinement = False
            p_global_keep = self.configuration.pressures
            pressures = self.configuration.pressures

            if mode.strip("-").strip("_").lower() == "bestfit":
                log_l, best_fit_index = self.get_best_fit_likelihood(samples_use)
                self.build_param_dict(samples_use[:-1, best_fit_index], parameters_read)
                sample_use = samples_use[:-1, best_fit_index]
            elif mode.lower() == "median":
                med_params, sample_use = self.get_median_params(
                    samples_use, parameters_read, return_array=True
                )
            else:
                sample_use = None

            (
                best_fit_wavelengths,
                best_fit_spectrum,
                best_fit_auxiliary_outputs,
            ) = (
                self.get_best_fit_model(
                    sample_use,
                    parameters_read,
                    model_generating_function=model_generating_function,
                    prt_reference=prt_reference,
                    refresh=refresh,
                    contribution=True,
                    mode=mode,
                )
            )
            best_fit_contribution = self._get_clipped_contribution_output(
                best_fit_auxiliary_outputs
            )

            pressure_weights = np.diff(np.log10(pressures))
            weights = np.ones_like(pressures)
            weights[:-1] = pressure_weights
            weights[-1] = weights[-2]
            weights = weights / np.sum(weights)
            weights = weights.reshape(len(weights), 1)

            x, y = np.meshgrid(best_fit_wavelengths, pressures)

            fig, ax = plt.subplots()
            if log_scale_contribution:
                plot_cont = -np.log10(
                    best_fit_contribution * self.y_axis_scaling / weights
                )
                label = "-Log Weighted Flux"
            else:
                plot_cont = best_fit_contribution * self.y_axis_scaling / weights
                label = "Weighted Flux"

            im = ax.contourf(x, y, plot_cont, n_contour_levels, cmap="magma")
            ax.set_xlabel(self.spec_xlabel)
            ax.set_ylabel("Pressure [bar]")
            ax.set_xscale(self.xscale)
            ax.set_yscale("log")
            ax.set_ylim(pressures[-1] * 1.03, pressures[0] / 1.03)
            plt.colorbar(im, ax=ax, label=label)

            plt.savefig(
                f"{self.get_base_figure_name()}_{mode}_contribution.pdf",
                bbox_inches="tight",
            )

            self.configuration.pressures = p_global_keep
            self.configuration.adaptive_mesh_refinement = adaptive_mesh_refinement
        else:
            fig = None
            ax = None

        self._synchronize()

        return fig, ax

    def plot_corner(
        self,
        sample_dict,
        parameter_dict,
        parameters_read,
        plot_best_fit=True,
        true_values=None,
        **kwargs,
    ):
        """
        Make the corner plots.
        """
        from petitRADTRANS.plotlib.plotlib import contour_corner

        fig = None

        if not sample_dict:
            raise ValueError("cannot make a corner plot without sample")

        if not parameter_dict:
            raise ValueError("cannot make a corner plot with no parameter to plot")

        if self._is_plotting_root():
            if not self.configuration.run_mode == "evaluate":
                print("Not in evaluate mode. Changing run mode to evaluate.")
                self.configuration.run_mode = "evaluate"

            print("\nMaking corner plot")
            sample_use_dict = {}
            p_plot_inds = {}
            p_ranges = {}
            p_use_dict = {}

            for name, params in parameter_dict.items():
                samples_use = copy.copy(sample_dict[name]).T
                parameters_use = copy.copy(params)
                parameter_plot_indices = []
                parameter_ranges = []
                i_p = 0

                for pp in parameters_read:
                    if self.configuration.parameters[pp].plot_in_corner:
                        parameter_plot_indices.append(i_p)

                    if self.configuration.parameters[pp].corner_label is not None:
                        parameters_use[i_p] = self.configuration.parameters[
                            pp
                        ].corner_label

                    if self.configuration.parameters[pp].corner_transform is not None:
                        samples_use[:, i_p] = self.configuration.parameters[
                            pp
                        ].corner_transform(samples_use[:, i_p])
                    parameter_ranges.append(
                        self.configuration.parameters[pp].corner_ranges
                    )

                    i_p += 1

                p_plot_inds[name] = np.array(parameter_plot_indices)
                p_ranges[name] = parameter_ranges
                p_use_dict[name] = parameters_use
                sample_use_dict[name] = copy.copy(samples_use)

            _plot_indices = [value.size for value in p_plot_inds.values()]

            if np.max(_plot_indices) == 0:
                print(
                    "No parameter to be plotted "
                    "(all Retrieval parameters 'plot_to_corner' attribute are set to False), "
                    "skipping corner plot..."
                )

                return fig

            output_file = self.get_base_figure_name() + "_corner_plot.pdf"
            fig = contour_corner(
                sample_use_dict,
                p_use_dict,
                output_file,
                parameter_plot_indices=p_plot_inds,
                parameter_ranges=p_ranges,
                prt_plot_style=self.prt_plot_style,
                plot_best_fit=plot_best_fit,
                true_values=true_values,
                **kwargs,
            )

        self._synchronize()

        return fig

    def plot_data(self, yscale="linear"):
        """
        Plot the data used in the retrieval.
        """
        import matplotlib.pyplot as plt

        if self._is_plotting_root():
            fig, ax = plt.subplots(figsize=(14, 7))
            for name, data in self.configuration.data.items():
                if isinstance(data, PhotometryData):
                    wavelengths = np.mean(data.photometric_bin_edges)
                else:
                    wavelengths = data.wavelengths

                scale = 1.0
                offset = 0.0
                spectrum = data.spectrum
                error = data.uncertainties

                if self.configuration.run_mode == "evaluate":
                    if data.scale_flux:
                        scale = self.best_fit_parameters[f"{name}_scale_factor"].value

                    if data.scale_uncertainties:
                        errscale = self.best_fit_parameters[
                            f"{name}_scale_factor"
                        ].value
                        error = error * errscale

                    if data.fit_flux_offset:
                        offset = self.best_fit_parameters[f"{name}_offset"].value

                spectrum = (spectrum * scale) - offset
                ax.errorbar(wavelengths, spectrum, yerr=error, label=name, marker="o")
                ax.set_yscale(yscale)
            ax.legend(fontsize=6)

            plt.savefig(self.get_base_figure_name() + "_Data.pdf")

        self._synchronize()

    def plot_pt(
        self,
        sample_dict,
        parameters_read,
        contribution=False,
        refresh=False,
        model_generating_function=None,
        prt_reference=None,
        mode="bestfit",
    ):
        """
        Plot the PT profile with error contours
        """
        import matplotlib.pyplot as plt
        from petitRADTRANS.plotlib.style import prt_default_colour_cycle

        if self._is_plotting_root():

            print("\nPlotting PT profiles")
            if not self.configuration.run_mode == "evaluate":
                print("Not in evaluate mode. Changing run mode to evaluate.")
                self.configuration.run_mode = "evaluate"

            samples_use = copy.copy(sample_dict[self.configuration.retrieval_name])

            adaptive_mesh_refinement = self.configuration.adaptive_mesh_refinement
            self.configuration.adaptive_mesh_refinement = False

            pressures = self.configuration.pressures
            if adaptive_mesh_refinement:
                for name, data in self.configuration.data.items():
                    if data.external_radtrans_reference is None:
                        data.radtrans_object._pressures = pressures * 1e6
            press_file = f"{self.get_base_figure_name()}_pressures"
            temp_file = f"{self.get_base_figure_name()}_temps"
            reference_data_name = self.get_reference_data_name()

            if (
                os.path.exists(press_file + ".npy")
                and os.path.exists(temp_file + ".npy")
                and not refresh
            ):
                pressures = np.load(press_file + ".npy")
                temps_sort = np.load(temp_file + ".npy")
            else:
                pressures, temps = self.calculate_temperature_profile(
                    samples_use[:-1, :],
                    data=reference_data_name,
                )

                temps = np.asarray(temps, dtype=float)
                temps_sort = np.sort(temps, axis=0)
                np.save(press_file, pressures)
                np.save(temp_file, temps_sort)

            if temps_sort.shape[0] == 0:
                raise ValueError("no valid temperature profiles available for PT plot")

            len_samp = temps_sort.shape[0]
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(
                pressures,
                temps_sort[0, :],
                color=prt_default_colour_cycle[0],
                alpha=0.8,
                zorder=0,
                linewidth=1.2
            )
            ax.plot(
                pressures,
                temps_sort[-1, :],
                color=prt_default_colour_cycle[0],
                alpha=0.8,
                zorder=0,
                linewidth=1.2
            )
            ax.plot(
                pressures,
                np.median(temps_sort, axis=0),
                color='k',
                alpha=1.0,
                zorder=10,
                linewidth=2.0
            )
            ax.fill_betweenx(
                pressures,
                x1=temps_sort[0, :],
                x2=temps_sort[-1, :],
                color=prt_default_colour_cycle[0],
                alpha=0.1,
                label="all",
                zorder=0,
            )
            ax.fill_betweenx(
                pressures,
                x1=temps_sort[int(len_samp * (0.5 - 0.997 / 2.0)), :],
                x2=temps_sort[int(len_samp * (0.5 + 0.997 / 2.0)), :],
                color=prt_default_colour_cycle[0],
                alpha=0.3,
                label=r"$3 \sigma$",
                zorder=1,
            )
            ax.fill_betweenx(
                pressures,
                x1=temps_sort[int(len_samp * (0.5 - 0.95 / 2.0)), :],
                x2=temps_sort[int(len_samp * (0.5 + 0.95 / 2.0)), :],
                color=prt_default_colour_cycle[0],
                alpha=0.7,
                label=r"$2 \sigma$",
                zorder=2,
            )
            ax.fill_betweenx(
                pressures,
                x1=temps_sort[int(len_samp * (0.5 - 0.68 / 2.0)), :],
                x2=temps_sort[int(len_samp * (0.5 + 0.68 / 2.0)), :],
                color=prt_default_colour_cycle[0],
                alpha=1.0,
                label=r"$1 \sigma$",
                zorder=3,
            )

            if self.temp_limits is not None:
                tlims = self.temp_limits
                ax.set_xlim(self.temp_limits)
            else:
                tlims = (np.min(temps_sort) * 0.97, np.max(temps_sort) * 1.03)
                ax.set_xlim(tlims)

            if contribution:
                if mode.strip("-").strip("_").lower() == "bestfit":
                    log_l, best_fit_index = self.get_best_fit_likelihood(samples_use)
                    self.build_param_dict(
                        samples_use[:-1, best_fit_index], parameters_read
                    )
                    sample_use = samples_use[:-1, best_fit_index]
                elif mode.lower() == "median":
                    med_param, sample_use = self.get_median_params(
                        samples_use, parameters_read, return_array=True
                    )
                else:
                    sample_use = None

                (
                    best_fit_wavelengths,
                    best_fit_spectrum,
                    best_fit_auxiliary_outputs,
                ) = (
                    self.get_best_fit_model(
                        sample_use,
                        parameters_read,
                        model_generating_function=model_generating_function,
                        prt_reference=prt_reference,
                        refresh=refresh,
                        contribution=True,
                        mode=mode,
                    )
                )
                best_fit_contribution = self._get_clipped_contribution_output(
                    best_fit_auxiliary_outputs
                )

                nu = wavelength2frequency(best_fit_wavelengths)
                mean_diff_nu = -np.diff(nu)
                diff_nu = np.zeros_like(nu)
                diff_nu[:-1] = mean_diff_nu
                diff_nu[-1] = diff_nu[-2]
                spectral_weights = (
                    best_fit_spectrum * diff_nu / np.sum(best_fit_spectrum * diff_nu)
                )

                if self.test_plotting:
                    plt.clf()
                    plt.plot(best_fit_wavelengths * 1e4, spectral_weights)
                    plt.show()
                    print(np.shape(best_fit_contribution))

                pressure_weights = np.diff(np.log10(pressures))
                weights = np.ones_like(pressures)
                weights[:-1] = pressure_weights
                weights[-1] = weights[-2]
                weights = weights / np.sum(weights)
                weights = weights.reshape(len(weights), 1)

                contr_em = best_fit_contribution / weights
                contr_em = contr_em * spectral_weights[None, :]
                contr_em = np.sum(contr_em, axis=1)
                contr_em = contr_em / np.sum(contr_em)

                if self.test_plotting:
                    plt.clf()
                    plt.yscale("log")
                    plt.ylim([pressures[-1], pressures[0]])
                    plt.plot(contr_em, pressures)
                    plt.show()

                contr_em_weigh = contr_em / np.max(contr_em)
                from scipy.interpolate import interp1d

                contr_em_weigh_intp = interp1d(pressures, contr_em_weigh)

                yborders = pressures
                for i_p in range(len(yborders) - 1):
                    mean_press = (yborders[i_p + 1] + yborders[i_p]) / 2.0
                    ax.fill_between(
                        tlims,
                        yborders[i_p + 1],
                        yborders[i_p],
                        color="white",
                        alpha=min(1.0 - contr_em_weigh_intp(mean_press), 0.9),
                        linewidth=0,
                        rasterized=True,
                        zorder=4,
                    )

                ax.plot(
                    contr_em_weigh * (tlims[1] - tlims[0]) + tlims[0],
                    pressures,
                    "--",
                    color="black",
                    linewidth=1.0,
                    label="Spectrally weighted contribution",
                    zorder=5,
                )
            ax.set_yscale("log")
            try:
                ax.set_ylim(self.press_limits)
            except Exception:
                ax.set_ylim([pressures[-1] * 1.03, pressures[0] / 1.03])

            ax.set_xlabel("Temperature [K]")
            ax.set_ylabel("Pressure [bar]")
            ax.legend(loc="best")
            plt.savefig(
                f"{self.get_base_figure_name()}_PT_envelopes.pdf", bbox_inches="tight"
            )
            self.configuration.adaptive_mesh_refinement = adaptive_mesh_refinement
        else:
            fig = None
            ax = None

        self._synchronize()

        return fig, ax

    def plot_sampled(
        self,
        samples_use,
        parameters_read,
        downsample_factor=None,
        save_outputs=True,
        nsample=None,
        model_generating_function=None,
        prt_reference=None,
        refresh=True,
    ):
        """
        Plot a set of randomly sampled output spectra for each dataset in the retrieval.
        """
        import matplotlib.pyplot as plt
        from petitRADTRANS.plotlib.plotlib import plot_data

        if self._is_plotting_root():

            if not self.configuration.run_mode == "evaluate":
                print("Not in evaluate mode. Changing run mode to evaluate.")
                self.configuration.run_mode = "evaluate"

            self.nsample = int(self.nsample)

            print(
                "\nPlotting Best-fit spectrum with " + str(self.nsample) + " samples."
            )
            print("This could take some time...")
            len_samples = samples_use.shape[1]
            path = os.path.join(
                self.output_directory, "evaluate_" + self.configuration.retrieval_name
            )

            wavelength_min = np.inf
            wavelength_max = -np.inf

            for name, data in self.configuration.data.items():
                if data.wavelength_boundaries[0] < wavelength_min:
                    wavelength_min = data.wavelength_boundaries[0]

                if data.wavelength_boundaries[1] > wavelength_max:
                    wavelength_max = data.wavelength_boundaries[1]

            wavelength_min = np.inf
            wavelength_max = -np.inf

            for name, data in self.configuration.data.items():
                if data.wavelength_boundaries[0] < wavelength_min:
                    wavelength_min = data.wavelength_boundaries[0]

                if data.wavelength_boundaries[1] > wavelength_max:
                    wavelength_max = data.wavelength_boundaries[1]

            p = self.configuration.pressures

            species = copy.copy(
                self.get_reference_radtrans_data().radtrans_object.line_species
            )

            atmosphere = Radtrans(
                pressures=p,
                line_species=tuple(species),
                rayleigh_species=copy.copy(self.configuration.rayleigh_species),
                gas_continuum_contributors=copy.copy(
                    self.configuration.continuum_opacities
                ),
                cloud_species=copy.copy(self.configuration.cloud_species),
                line_opacity_mode=self.get_reference_radtrans_data().line_opacity_mode,
                wavelength_boundaries=np.array(
                    [wavelength_min * 0.98, wavelength_max * 1.02]
                ),
                scattering_in_emission=self.configuration.scattering_in_emission,
            )

            fig, ax = plt.subplots(figsize=(14, 7))
            n_samples = nsample

            if nsample is None:
                n_samples = self.nsample

            random_ints = np.random.randint(
                low=0, high=len_samples, size=int(n_samples)
            )
            sample_matrix = np.asarray(samples_use[:-1, random_ints].T, dtype=float)
            sampled_wavelengths = [None] * len(random_ints)
            sampled_models = [None] * len(random_ints)
            missing_sample_rows = []
            missing_file_stems = []

            for i_sample, random_index in enumerate(random_ints):
                file_stem = os.path.join(
                    path, "posterior_sampled_spectra_" + str(random_index).zfill(5)
                )
                cache_file = file_stem + ".npy"

                if os.path.exists(cache_file):
                    cached_output = np.load(cache_file)

                    if cached_output.ndim != 2:
                        raise ValueError(
                            f"expected cached sampled spectrum '{cache_file}' to be 2-D"
                        )

                    if cached_output.shape[1] == 2:
                        wavelengths = cached_output[:, 0]
                        model = cached_output[:, 1]
                    elif cached_output.shape[0] == 2:
                        wavelengths = cached_output[0, :]
                        model = cached_output[1, :]
                    else:
                        raise ValueError(
                            f"expected cached sampled spectrum '{cache_file}' to have two columns"
                        )

                    sampled_wavelengths[i_sample] = np.asarray(wavelengths, dtype=float)
                    sampled_models[i_sample] = np.asarray(model, dtype=float)
                    continue

                missing_sample_rows.append(i_sample)
                missing_file_stems.append(file_stem)

            if missing_sample_rows:
                print(
                    f"Generating {len(missing_sample_rows)} sampled spectra with vmap..."
                )
                batched_wavelengths, batched_models = (
                    self._get_batched_full_range_model_outputs(
                        sample_matrix[np.asarray(missing_sample_rows)],
                        parameters_read,
                        model_generating_function=model_generating_function,
                        prt_reference=prt_reference,
                        prt_object=atmosphere,
                    )
                )

                if batched_wavelengths.ndim == 1:
                    batched_wavelengths = batched_wavelengths[np.newaxis, :]
                    batched_models = batched_models[np.newaxis, :]

                for batch_index, sample_row in enumerate(missing_sample_rows):
                    wavelengths = np.asarray(
                        batched_wavelengths[batch_index], dtype=float
                    )
                    model = np.asarray(batched_models[batch_index], dtype=float)
                    sampled_wavelengths[sample_row] = wavelengths
                    sampled_models[sample_row] = model

                    if save_outputs:
                        np.save(
                            missing_file_stems[batch_index],
                            np.column_stack((wavelengths, model)),
                        )

            for wavelengths, model in zip(
                sampled_wavelengths, sampled_models, strict=True
            ):
                if downsample_factor is not None:
                    model = running_mean(model, downsample_factor)[::downsample_factor]
                    wavelengths = wavelengths[::downsample_factor]

                ax.plot(
                    wavelengths,
                    model * self.y_axis_scaling,
                    color="#00d2f3",
                    alpha=1 / self.nsample + 0.1,
                    linewidth=0.2,
                    marker=None,
                )

            log_l, best_fit_index = self.get_best_fit_likelihood(samples_use)

            best_fit_wavelengths, best_fit_spectrum = self.get_best_fit_model(
                samples_use[:-1, best_fit_index],
                parameters_read,
                model_generating_function=model_generating_function,
                prt_reference=prt_reference,
                refresh=refresh,
            )
            chi2 = self.get_reduced_chi2_from_model(
                wavelengths_model=best_fit_wavelengths,
                spectrum_model=best_fit_spectrum,
                subtract_n_parameters=True,
                parameters=self.best_fit_parameters,
                verbose=True,
                show_chi2=True,
            )
            ax.plot(
                best_fit_wavelengths,
                best_fit_spectrum * self.y_axis_scaling,
                marker=None,
                label=rf"Best fit, $\chi^{{2}}$={chi2:.2f}",
                linewidth=4,
                alpha=0.5,
                color="r",
            )

            for name, data in self.configuration.data.items():
                fig, ax = plot_data(
                    fig,
                    ax,
                    data,
                    resolution=self.resolution,
                    scaling=self.y_axis_scaling,
                )

            ax.set_xlabel("Wavelength [micron]")
            ax.set_ylabel(self.spec_ylabel)
            ax.legend(loc="best")
            plt.tight_layout()
            plt.savefig(
                os.path.join(path, self.configuration.retrieval_name + "_sampled.pdf"),
                bbox_inches=0.0,
            )
        else:
            fig = None
            ax = None

        self._synchronize()

        return fig, ax

    def plot_spectra(
        self,
        samples_use,
        parameters_read,
        model_generating_function=None,
        prt_reference=None,
        refresh=True,
        mode="bestfit",
        marker_color_type=None,
        marker_cmap=None,
        marker_label="",
        only_save_best_fit_spectra=False,
    ):
        """
        Plot the best fit spectrum, the data from each dataset and the residuals between the two.
        """
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator, MultipleLocator, NullFormatter

        def __rebin_data(
            _resolution, _wavelengths, _spectrum, _uncertainties, _data_wavelength_bins
        ):
            _resolution_data = np.mean(_wavelengths[1:] / np.diff(_wavelengths))

            if _resolution is not None and _resolution < _resolution_data:
                _ratio = _resolution_data / _resolution
                _flux, _edges, _ = binned_statistic(
                    _wavelengths, _spectrum, "mean", _wavelengths.shape[0] / _ratio
                )

                _error, _, _ = binned_statistic(
                    _wavelengths, _uncertainties, "mean", _wavelengths.shape[0] / _ratio
                )
                _error = np.array(_error) / np.sqrt(_ratio)
                _wavelengths = np.array(
                    [
                        (_edges[_i] + _edges[_i + 1]) * 0.5
                        for _i in range(_edges.shape[0] - 1)
                    ]
                )
                _wavelengths_bins = np.zeros_like(_wavelengths)
                _wavelengths_bins[:-1] = np.diff(_wavelengths)
                _wavelengths_bins[-1] = _wavelengths_bins[-2]
            else:
                _flux = _spectrum
                _error = _uncertainties
                _wavelengths_bins = _data_wavelength_bins

            return _wavelengths, _flux, _error, _wavelengths_bins

        def __add_data_to_axis(
            _ax,
            _data,
            _wavelengths,
            _flux,
            _error,
            _wavelengths_bins,
            _marker,
            _markersize,
            _markerfacecolors,
            _ecolor,
            _y_axis_scaling,
            _label,
            _alpha,
        ):
            xerr = None
            if isinstance(_data, PhotometryData):
                xerr = _wavelengths_bins / 2
            _ax.errorbar(
                _wavelengths,
                _flux * _y_axis_scaling,
                yerr=_error * _y_axis_scaling,
                xerr=xerr,
                marker=_marker,
                markersize=_markersize,
                markerfacecolor=_markerfacecolors,
                ecolor=_ecolor,
                markeredgecolor="k",
                linewidth=0,
                elinewidth=2,
                label=_label,
                zorder=10,
                alpha=_alpha,
            )

        def __calculate_standardized_residuals(
            _flux,
            _model_flux,
            _error,
            _covariance,
        ):
            residual = np.asarray(_flux, dtype=float) - np.asarray(
                _model_flux, dtype=float
            )

            if _covariance is not None and residual.ndim == 1:
                covariance = np.asarray(_covariance, dtype=float)
                covariance_cholesky = np.linalg.cholesky(covariance)
                return np.linalg.solve(covariance_cholesky, residual)

            return residual / np.asarray(_error, dtype=float)

        def plot_best_fit_model(
            _ax,
            _best_fit_wavelengths,
            _best_fit_spectrum,
            _y_axis_scaling,
            _chi2,
            _mode,
        ):
            _spectrum_type_label = "Best fit" if _mode == "bestfit" else "Median"
            _ax.plot(
                _best_fit_wavelengths,
                _best_fit_spectrum * _y_axis_scaling,
                label=rf"{_spectrum_type_label} model, $\chi^2/\nu$={_chi2:.2f}",
                linewidth=3,
                alpha=0.8,
                color="k",
            )

        if marker_cmap is None:
            marker_cmap = plt.colormaps["bwr"]

        if self._is_plotting_root():
            check = self.evaluate_sample_spectra
            self.evaluate_sample_spectra = False

            if not self.configuration.run_mode == "evaluate":
                print("Not in evaluate mode. Changing run mode to evaluate.")
                self.configuration.run_mode = "evaluate"

            print("\nPlotting Best-fit spectrum")

            fig, axes = plt.subplots(
                nrows=2,
                ncols=1,
                sharex="col",
                sharey=False,
                gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.1},
                figsize=(14, 7),
            )

            ax = axes[0]
            ax_r = axes[1]

            log_l, best_fit_index = self.get_best_fit_likelihood(samples_use)
            sample_use = samples_use[:-1, best_fit_index]

            if mode.lower() == "median":
                med_param, sample_use = self.get_median_params(
                    samples_use, parameters_read, return_array=True
                )

            if only_save_best_fit_spectra:
                self.save_best_fit_outputs(
                    self.best_fit_parameters,
                    retain_best_fit_spectra=False,
                )
                return None, None, None

            best_fit_wavelengths, best_fit_spectrum = self.get_best_fit_model(
                sample_use,
                parameters_read,
                model_generating_function=model_generating_function,
                prt_reference=prt_reference,
                refresh=refresh,
                mode=mode,
            )
            chi2 = self.get_reduced_chi2_from_model(
                wavelengths_model=best_fit_wavelengths,
                spectrum_model=best_fit_spectrum,
                subtract_n_parameters=True,
                parameters=self.best_fit_parameters,
                verbose=True,
                show_chi2=True,
            )

            if not only_save_best_fit_spectra:
                self.save_best_fit_outputs(self.best_fit_parameters)

            plotted_model_wavelengths = np.asarray(best_fit_wavelengths, dtype=float)
            plotted_model_spectrum = np.asarray(best_fit_spectrum, dtype=float)

            reference_data_name = self.get_reference_data_name()
            reference_data = self.configuration.data.get(reference_data_name)

            if (
                reference_data is not None
                and not isinstance(reference_data, PhotometryData)
                and reference_data_name in self.best_fit_spectra
            ):
                plotted_model_wavelengths = np.asarray(
                    self.best_fit_spectra[reference_data_name][0], dtype=float
                )
                plotted_model_spectrum = np.asarray(
                    self.best_fit_spectra[reference_data_name][1], dtype=float
                )

            markersize = None
            markerfacecolors = {}
            has_correlated_noise = False

            if marker_color_type is not None:
                from matplotlib.colors import Normalize
                from matplotlib.cm import ScalarMappable

                markersize = 10
                l, b, w, h = ax.get_position().bounds
                cax = fig.add_axes([l + w + 0.015 * w, b, 0.025 * w, h])

                vmin, vmax = np.inf, -np.inf
                for name, data in self.configuration.data.items():
                    assert hasattr(data, marker_color_type)

                    markerfacecolors[name] = getattr(data, marker_color_type)
                    vmin = min([markerfacecolors[name].min(), vmin])
                    vmax = max([markerfacecolors[name].max(), vmax])

                if marker_color_type.startswith("delta_"):
                    vmax = np.max(np.abs([vmin, vmax]))
                    vmin = -1 * vmax

                norm = Normalize(vmin=vmin, vmax=vmax)
                for name, data in self.configuration.data.items():
                    markerfacecolors[name] = norm(markerfacecolors[name])
                    markerfacecolors[name] = marker_cmap(markerfacecolors[name])

                fig.colorbar(
                    ScalarMappable(norm=norm, cmap=marker_cmap),
                    ax=ax,
                    cax=cax,
                    orientation="vertical",
                )
                cax.set_ylabel(marker_label)

                if marker_color_type == "pareto_k":
                    cax.axhline(0.7, c="k", ls="--")

            for name, data in self.configuration.data.items():
                if not isinstance(data, PhotometryData):
                    wavelengths, flux, error, wavelengths_bins = __rebin_data(
                        self.resolution,
                        data.wavelengths,
                        data.spectrum,
                        data.uncertainties,
                        data.wavelength_bin_widths,
                    )
                else:
                    wavelengths = np.mean(data.photometric_bin_edges)
                    flux = data.spectrum
                    error = data.uncertainties
                    wavelengths_bins = data.wavelength_bin_widths

                flux = np.asarray(
                    data.apply_flux_scaling(data.spectrum, self.best_fit_parameters),
                    dtype=float,
                )
                flux = np.asarray(
                    data.offset_flux(flux, self.best_fit_parameters),
                    dtype=float,
                )
                error, covariance = data.scale_and_inflate_uncertainties(
                    self.best_fit_parameters
                )
                error = np.asarray(error, dtype=float)

                if covariance is not None:
                    covariance = np.asarray(covariance, dtype=float)
                    has_correlated_noise = True

                best_fit_spectrum_name = name
                if (
                    best_fit_spectrum_name not in self.best_fit_spectra
                    and data.external_radtrans_reference is not None
                ):
                    best_fit_spectrum_name = data.external_radtrans_reference

                best_fit_binned = np.asarray(
                    self.best_fit_spectra[best_fit_spectrum_name][1],
                    dtype=float,
                )
                residuals = __calculate_standardized_residuals(
                    flux,
                    best_fit_binned,
                    error,
                    covariance,
                )

                wavelengths = np.asarray(wavelengths, dtype=float)
                wavelengths_bins = np.asarray(wavelengths_bins, dtype=float)

                marker = "o" if not isinstance(data, PhotometryData) else "s"
                label = data.name

                if np.ndim(flux) == 0:
                    wavelengths = np.asarray([wavelengths], dtype=float)
                    flux = np.asarray([flux], dtype=float)
                    error = np.asarray([error], dtype=float)
                    best_fit_binned = np.asarray([best_fit_binned], dtype=float)
                    residuals = np.asarray([residuals], dtype=float)
                    wavelengths_bins = np.asarray([wavelengths_bins], dtype=float)

                for i in range(len(flux)):
                    if i > 0:
                        label = None

                    color_i = "C0"
                    ecolor = "C0"
                    alpha = 0.9
                    if isinstance(data, PhotometryData):
                        color_i = "grey"
                        ecolor = "grey"
                        label = None
                        alpha = 0.6

                    if marker_color_type is not None:
                        color_i = markerfacecolors[name][i]
                        ecolor = "k"

                    __add_data_to_axis(
                        ax,
                        data,
                        _wavelengths=wavelengths[i],
                        _flux=flux[i],
                        _error=error[i],
                        _wavelengths_bins=wavelengths_bins[i],
                        _marker=marker,
                        _markersize=markersize,
                        _markerfacecolors=color_i,
                        _ecolor=ecolor,
                        _y_axis_scaling=self.y_axis_scaling,
                        _label=label,
                        _alpha=alpha,
                    )

                    __add_data_to_axis(
                        ax_r,
                        data,
                        _wavelengths=wavelengths[i],
                        _flux=residuals[i],
                        _error=1,
                        _wavelengths_bins=wavelengths_bins[i],
                        _marker=marker,
                        _markersize=markersize,
                        _markerfacecolors=color_i,
                        _ecolor=ecolor,
                        _y_axis_scaling=1.0,
                        _label=None,
                        _alpha=alpha,
                    )

            plot_best_fit_model(
                ax,
                plotted_model_wavelengths,
                plotted_model_spectrum,
                self.y_axis_scaling,
                chi2,
                mode,
            )

            yabs_max = abs(max(ax_r.get_ylim(), key=abs))
            lims = ax.get_xlim()
            lim_y = ax.get_ylim()
            lim_y = [lim_y[0], lim_y[1] * 1.05]

            if self.flux_lim is not None:
                ax.set_ylim(self.flux_lim)
            else:
                ax.set_ylim(lim_y)

            if self.xscale == "log":
                lims = [plotted_model_wavelengths[0] * 0.98, lims[1] * 1.02]
            else:
                lims = [
                    plotted_model_wavelengths[0] * 0.98,
                    plotted_model_wavelengths[-1] * 1.02,
                ]

            if self.wavelength_lim is not None:
                ax.set_xlim(self.wavelength_lim)
                ax_r.set_xlim(self.wavelength_lim)
            else:
                ax.set_xlim(lims)
                ax_r.set_xlim(lims)

            ax_r.set_ylim(ymin=-yabs_max, ymax=yabs_max)
            ax_r.fill_between(lims, -1, 1, color="dimgrey", alpha=0.4, zorder=-10)
            ax_r.fill_between(lims, -3, 3, color="darkgrey", alpha=0.3, zorder=-9)
            ax_r.fill_between(lims, -5, 5, color="lightgrey", alpha=0.3, zorder=-8)
            ax_r.axhline(linestyle="--", color="k", alpha=0.8, linewidth=2)

            if self.xscale is not None:
                ax.set_xscale(self.xscale)
            if self.yscale is not None:
                ax.set_yscale(self.yscale)

            ax.tick_params(
                axis="both",
                direction="in",
                length=10,
                bottom=True,
                top=True,
                left=True,
                right=True,
            )

            try:
                ax.xaxis.set_major_formatter("{x:.1f}")
            except Exception:
                warnings.warn("Please update to matplotlib 3.3.4 or greater")

            min_wavelength = plotted_model_wavelengths[0]
            max_wavelength = plotted_model_wavelengths[-1]

            ax.tick_params(
                axis="both",
                which="minor",
                bottom=True,
                top=True,
                left=True,
                right=True,
                direction="in",
                length=5,
            )

            if self.xscale == "log":
                if min_wavelength < 0:
                    min_wavelength = 0.08
                x_major_ticks = []
                for i_tick in np.linspace(
                    min_wavelength, 1.0, int(10 * (1.0 - min_wavelength))
                ):
                    x_major_ticks.append(round(i_tick, 1))
                for i_tick in range(1, int(round(max_wavelength, 0))):
                    x_major_ticks.append(int(i_tick))
                ax.set_xticks(x_major_ticks)
                ax.set_xticklabels(x_major_ticks)
                x_minor = MultipleLocator(1)
                ax.xaxis.set_minor_locator(x_minor)
                ax.xaxis.set_minor_formatter(NullFormatter())
            else:
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.tick_params(
                    axis="both",
                    which="minor",
                    bottom=True,
                    top=True,
                    left=True,
                    right=True,
                    direction="in",
                    length=5,
                )

            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.set_ylabel(self.spec_ylabel)

            ax_r.tick_params(
                axis="both",
                direction="in",
                length=10,
                bottom=True,
                top=True,
                left=True,
                right=True,
            )

            try:
                ax_r.xaxis.set_major_formatter("{x:.1f}")
            except Exception:
                warnings.warn("Please update to matplotlib 3.3.4 or greater")

            ax_r.tick_params(
                axis="both",
                which="minor",
                bottom=True,
                top=True,
                left=True,
                right=True,
                direction="in",
                length=5,
            )

            if self.xscale == "log":
                if min_wavelength < 0:
                    min_wavelength = 0.08
                x_major_ticks = []

                for i_tick in np.linspace(
                    min_wavelength, 1.0, int(10 * (1.0 - min_wavelength))
                ):
                    x_major_ticks.append(round(i_tick, 1))

                for i_tick in range(1, int(round(max_wavelength, 0))):
                    x_major_ticks.append(int(i_tick))

                ax_r.set_xticks(x_major_ticks)
                ax_r.set_xticklabels(x_major_ticks)
                x_minor = MultipleLocator(1)
                ax_r.xaxis.set_minor_locator(x_minor)
                ax_r.xaxis.set_minor_formatter(NullFormatter())
            else:
                ax_r.xaxis.set_minor_locator(AutoMinorLocator())
                ax_r.tick_params(
                    axis="both",
                    which="minor",
                    bottom=True,
                    top=True,
                    left=True,
                    right=True,
                    direction="in",
                    length=5,
                )
            ax_r.yaxis.set_minor_locator(AutoMinorLocator())
            if has_correlated_noise:
                ax_r.set_ylabel(r"Whitened residuals [$\sigma$]")
            else:
                ax_r.set_ylabel(r"Residuals [$\sigma$]")
            ax_r.set_xlabel(self.spec_xlabel)
            ax.legend(
                loc="upper center", ncol=len(self.configuration.data.keys()) + 1
            ).set_zorder(1002)

            fig.align_ylabels()
            plt.tight_layout()
            plt.savefig(self.get_base_figure_name() + "_" + mode + "_spec.pdf")
            self.evaluate_sample_spectra = check
        else:
            fig = None
            ax = None
            ax_r = None

        self._synchronize()

        return fig, ax, ax_r
