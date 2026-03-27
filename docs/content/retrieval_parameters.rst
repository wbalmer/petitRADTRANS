.. _retrieval_parameters:

Retrieval Parameter Reference
==============================

This document describes all parameters available for use in petitRADTRANS atmospheric retrievals.
Parameters are passed as a dictionary of :class:`~petitRADTRANS.retrieval.parameter.Parameter` objects
to the model generating function, and are registered via :meth:`~petitRADTRANS.retrieval.retrieval_config.RetrievalConfig.add_parameter`.

.. contents:: Contents
   :local:
   :depth: 2

----

Overview of Built-in Model Functions
--------------------------------------

The built-in model functions available in :mod:`petitRADTRANS.retrieval.models` are listed below.
Each function accepts the same signature ``(prt_object, parameters, pt_plot_mode, adaptive_mesh_refinement)``
and computes either an emission or transmission spectrum.

.. list-table::
   :widths: 35 15 20 30
   :header-rows: 1

   * - Function
     - Type
     - Temperature Profile
     - Notes
   * - ``spline_eddington_adiabatic_emission``
     - Emission
     - Mollière 2020 spline
   * - ``emission_model_diseq_patchy_clouds``
     - Emission
     - Mollière 2020 spline
     - Full two-column patchy-cloud model; requires separate clear-column T-P parameters
   * - ``guillot_emission``
     - Emission
     - Guillot 2010
     - Simple analytic profile
   * - ``guillot_emission_add_gaussian_temperature``
     - Emission
     - Guillot 2010 + Gaussian bump
     - Can add a localised temperature perturbation
   * - ``interpolated_profile_emission``
     - Emission
     - Cubic/linear spline nodes
     - Flexible node-based profile; number of nodes is a fixed parameter
   * - ``gradient_profile_emission``
     - Emission
     - d(log T)/d(log P) gradients (Zhang 2023)
     - Parameterised by per-layer slopes
     - Recommended default for emission retrievals
   * - ``power_law_profile_emission``
     - Emission
     - Power law T(P)
   * - ``madhushudhan_seager_emission``
     - Emission
     - Madhusudhan & Seager 2009
     - Three-region profile joined at knot pressures
   * - ``guillot_transmission``
     - Transmission
     - Guillot 2010
     - Supports physical and gray/haze clouds
   * - ``madhushudhan_seager_transmission``
     - Transmission
     - Madhusudhan & Seager 2009
     - Supports physical and gray/haze clouds
   * - ``isothermal_transmission``
     - Transmission
     - Isothermal
     - Simplest transmission model

.. note::

   ``guillot_patchy_emission``, ``guillot_patchy_transmission``, and
   ``madhu_seager_patchy_transmission`` are deprecated aliases and simply delegate
   to the non-patchy variants.  Use ``patchiness`` instead (see :ref:`cloud-patchy`).

----

.. _params-planet:

Planetary / Orbital Parameters
--------------------------------

These parameters describe the bulk properties of the planet and are required by **all** model functions.
Exactly **two** of the three physical quantities (``log_g``, ``planet_radius``, ``mass``) must be provided;
the third is derived automatically.

.. list-table::
   :widths: 20 15 10 55
   :header-rows: 1

   * - Key
     - Units
     - Required
     - Description
   * - ``log_g``
     - log\ :sub:`10`\ (cm s\ :sup:`-2`\ )
     - Two of three
     - Log base-10 of the surface gravity.
   * - ``planet_radius``
     - cm
     - Two of three
     - Planet radius at the reference pressure.
   * - ``mass``
     - g
     - Two of three
     - Planet mass.  Also accepted as ``planet_mass``.
   * - ``distance_to_system``
     - cm
     - Emission only
     - Distance from the observer to the planetary system.  Used to convert flux to irradiance.
   * - ``reference_pressure``
     - bar
     - Transmission (optional)
     - Pressure level at which ``planet_radius`` is defined.  Defaults to 100 bar.

.. rubric:: Gravity / radius derivation rules

* ``log_g`` **and** ``mass`` → radius derived from :meth:`Planet.reference_gravity2radius`
* ``log_g`` **and** ``planet_radius`` → gravity = 10\ :sup:`log_g`
* ``mass`` **and** ``planet_radius`` → gravity derived from :meth:`Planet.mass2reference_gravity`

----

.. _params-temperature:

Temperature–Pressure Profile Parameters
-----------------------------------------

Each model function uses a distinct set of T–P parameters.  Only the parameters belonging
to the chosen model function should be included in the retrieval configuration.

.. _tp-molliere:

Mollière 2020 Spline Profile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used by ``spline_eddington_adiabatic_emission`` and ``emission_model_diseq_patchy_clouds``.

The profile is constructed from four anchor temperatures (T\ :sub:`1`–T\ :sub:`3` and T\ :sub:`int`\)
and a power-law optical depth parameterisation τ = δ × P\ :sup:`α`.
The three spline fractions T\ :sub:`1`–T\ :sub:`3` are dimensionless values in (0, 1) that set
each anchor temperature relative to the one below it.

.. list-table::
   :widths: 20 15 10 55
   :header-rows: 1

   * - Key
     - Units
     - Required
     - Description
   * - ``T_int``
     - K
     - Yes
     - Interior (intrinsic) temperature of the planet.
   * - ``T3``
     - —  (0–1)
     - Yes
     - Innermost spline fraction; sets T\ :sub:`3` = (3/4 · T\ :sub:`int`\ :sup:`4` · (0.1 + 2/3))\ :sup:`0.25` · (1 − T3).
   * - ``T2``
     - —  (0–1)
     - Yes
     - Middle spline fraction; sets T\ :sub:`2` = T\ :sub:`3` · (1 − T2).
   * - ``T1``
     - —  (0–1)
     - Yes
     - Outer spline fraction; sets T\ :sub:`1` = T\ :sub:`2` · (1 − T1).
   * - ``log_delta``
     - —
     - Yes
     - Proportionality factor in the optical depth law; mapped from (0, 1) to (−3, 2).
   * - ``alpha``
     - —
     - Yes
     - Power-law index in τ = δ × P\ :sup:`α`.

For ``emission_model_diseq_patchy_clouds`` the following additional parameters describe the
**clear-column** temperature profile:

.. list-table::
   :widths: 20 15 10 55
   :header-rows: 1

   * - Key
     - Units
     - Required
     - Description
   * - ``T3_clear``
     - —  (0–1)
     - Yes
     - Innermost spline fraction for the cloud-free column.
   * - ``T2_clear``
     - —  (0–1)
     - Yes
     - Middle spline fraction for the cloud-free column.
   * - ``T1_clear``
     - —  (0–1)
     - Yes
     - Outer spline fraction for the cloud-free column.
   * - ``log_delta_clear``
     - —
     - Yes
     - ``log_delta`` for the clear column.
   * - ``alpha_clear``
     - —
     - Yes
     - ``alpha`` for the clear column.

.. _tp-guillot:

Guillot 2010 Profile
~~~~~~~~~~~~~~~~~~~~~

Used by ``guillot_emission``, ``guillot_emission_add_gaussian_temperature``,
``guillot_transmission``.

The profile follows the global Guillot (2010) analytical solution for an irradiated atmosphere.

.. list-table::
   :widths: 20 15 10 55
   :header-rows: 1

   * - Key
     - Units
     - Required
     - Description
   * - ``T_int``
     - K
     - Yes
     - Interior temperature.
   * - ``T_equ``
     - K
     - Yes
     - Planet equilibrium temperature.
   * - ``gamma``
     - —
     - Yes
     - Ratio of optical to infrared mean opacity, κ\ :sub:`vis` / κ\ :sub:`IR`.
   * - ``log_kappa_IR``
     - log\ :sub:`10`\ (cm\ :sup:`2` g\ :sup:`-1`\ )
     - Yes
     - Log of the IR opacity ratio used to scale the infrared opacity.

``guillot_emission_add_gaussian_temperature`` additionally requires:

.. list-table::
   :widths: 20 15 10 55
   :header-rows: 1

   * - Key
     - Units
     - Required
     - Description
   * - ``temperature_peak``
     - K
     - Yes
     - Amplitude of the Gaussian temperature perturbation.
   * - ``temperature_location_log_pressure``
     - log\ :sub:`10`\ (bar)
     - Yes
     - Pressure level at the centre of the Gaussian bump.
   * - ``temperature_width_log_pressure``
     - log\ :sub:`10`\ (bar)
     - Yes
     - Width (σ) of the Gaussian in log-pressure space.

.. _tp-spline-nodes:

Interpolated Spline Node Profile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used by ``interpolated_profile_emission``.  The total number of temperature nodes is
``nnodes + 2`` (including fixed top and bottom boundaries).

.. list-table::
   :widths: 20 15 10 55
   :header-rows: 1

   * - Key
     - Units
     - Required
     - Description
   * - ``nnodes``
     - integer
     - Yes
     - Number of *interior* temperature nodes (excluding the top and bottom boundaries).
       Total nodes = ``nnodes + 2``.
   * - ``T{i}``
     - K
     - Yes
     - Temperature at node *i*, for *i* = 0, 1, …, ``nnodes + 1``.
   * - ``gamma``
     - —
     - Yes
     - Smoothness penalty weight applied to the curvature of the profile.
   * - ``linear``
     - flag
     - Optional
     - If this key is present in the parameter dictionary (any value), a linear
       spline is used instead of the default cubic spline.
   * - ``log_prior_weight``
     - —
     - Internal
     - Written by the model function at runtime; its value is automatically added
       to the log-likelihood as a regularisation prior.

.. _tp-gradient:

Gradient (dT/d log P) Profile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used by ``gradient_profile_emission``.  Atmosphere is divided into ``N_layers`` layers,
each characterised by a constant log-pressure temperature slope.

.. list-table::
   :widths: 20 15 10 55
   :header-rows: 1

   * - Key
     - Units
     - Required
     - Description
   * - ``N_layers``
     - integer
     - Yes
     - Number of layers between which the temperature gradient is interpolated.
   * - ``T_bottom``
     - K
     - Yes
     - Temperature at the base of the atmosphere (deepest pressure level).
   * - ``PTslope_{i}``
     - K / dec
     - Yes
     - d (log T)/d(log P) for layer *i*, for *i* = 1, …, ``N_layers``.
   * - ``top_of_atmosphere_pressure``
     - log\ :sub:`10`\ (bar)
     - Optional
     - Log pressure at the top of the atmosphere.  Defaults to −3.
   * - ``bottom_of_atmosphere_pressure``
     - log\ :sub:`10`\ (bar)
     - Optional
     - Log pressure at the base of the atmosphere.  Defaults to 3.

.. _tp-powerlaw:

Power-Law Profile
~~~~~~~~~~~~~~~~~~

Used by ``power_law_profile_emission``.

.. list-table::
   :widths: 20 15 10 55
   :header-rows: 1

   * - Key
     - Units
     - Required
     - Description
   * - ``alpha``
     - —
     - Yes
     - Power-law exponent: T(P) ∝ P\ :sup:`alpha`.
   * - ``T_0``
     - K
     - Yes
     - Multiplicative scale factor for the temperature profile.

.. _tp-madhusudhan:

Madhusudhan & Seager 2009 Profile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used by ``madhushudhan_seager_emission`` and ``madhushudhan_seager_transmission``.
The profile is split into three pressure regions joined at knot pressures P\ :sub:`1` < P\ :sub:`2` < P\ :sub:`3`.

.. list-table::
   :widths: 20 15 10 55
   :header-rows: 1

   * - Key
     - Units
     - Required
     - Description
   * - ``log_P_set``
     - log\ :sub:`10`\ (bar)
     - Yes
     - Reference pressure used to anchor the profile.  Defaults to 10 bar.
   * - ``T_set``
     - K
     - Yes
     - Temperature at ``P_set``.
   * - ``log_P3``
     - log\ :sub:`10`\ (bar)
     - Yes
     - Log pressure at the top of the deepest atmospheric layer.
   * - ``P2``
     - —  (0–1)
     - Yes
     - Fractional inner pressure node; sets P\ :sub:`2` = P\ :sub:`3` · (1 − P2).
   * - ``P1``
     - —  (0–1)
     - Yes
     - Fractional outer pressure node; sets P\ :sub:`1` = P\ :sub:`2` · (1 − P1).
   * - ``alpha_0``
     - —
     - Yes
     - Slope of the upper atmospheric layer.
   * - ``alpha_1``
     - —
     - Yes
     - Slope of the middle atmospheric layer.
   * - ``beta``
     - —
     - Optional
     - Power-law exponent for the slopes.  Default = 0.5.  Not recommended to change.

.. _tp-isothermal:

Isothermal Profile
~~~~~~~~~~~~~~~~~~~

Used by ``isothermal_transmission``.

.. list-table::
   :widths: 20 15 10 55
   :header-rows: 1

   * - Key
     - Units
     - Required
     - Description
   * - ``temperature``
     - K
     - Yes
     - Uniform temperature throughout the atmosphere.

----

.. _params-chemistry:

Chemistry Parameters
---------------------

Two chemistry modes are supported: **equilibrium chemistry**, in which the abundances of all
species are derived from the elemental ratios and temperature–pressure profile, and
**free chemistry** (also called disequilibrium or free-retrieval chemistry), in which each
molecular abundance is an independent retrieved parameter.  It is also possible to use
equilibrium chemistry for most species while retrieving individual abundances for a subset.

.. _chemistry-equilibrium:

Equilibrium Chemistry
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 15 10 55
   :header-rows: 1

   * - Key
     - Units
     - Required
     - Description
   * - ``Fe/H``
     - solar
     - Yes
     - Atmospheric metallicity [M/H] relative to solar.
   * - ``C/O``
     - —
     - One of two
     - Carbon-to-oxygen ratio.  Mutually exclusive with specifying ``C`` and ``O`` individually.
   * - ``C``
     - —
     - One of two
     - Carbon elemental abundance.  Used together with ``O`` instead of ``C/O``.
   * - ``O``
     - —
     - One of two
     - Oxygen elemental abundance.  Used together with ``C`` instead of ``C/O``.
   * - ``log_pquench``
     - log\ :sub:`10`\ (bar)
     - Optional
     - Quench pressure for CO, CH\ :sub:`4`, and H\ :sub:`2`\ O.  If omitted,
       full thermochemical equilibrium is assumed.
   * - ``use_easychem``
     - flag
     - Optional
     - If present, use EasyChem for on-the-fly equilibrium chemistry calculations.

.. _chemistry-free:

Free (Uncorrelated) Chemistry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In free-chemistry mode each molecular species that contributes line opacity is assigned an
independent log\ :sub:`10` mass-fraction parameter.  The parameter key must **exactly match**
the species name as registered in ``Radtrans.line_species``.

.. list-table::
   :widths: 30 15 10 45
   :header-rows: 1

   * - Key pattern
     - Units
     - Required
     - Description
   * - ``{species_name}``
     - log\ :sub:`10`\ (mass fraction)
     - One per species
     - Log base-10 mass fraction of a line opacity species, e.g. ``H2O__POKAZATEL``,
       ``CO-NatAbun``, ``CH4__MM``.  Typical prior range: (−12.0, −0.5).

.. _chemistry-pressure-varying:

Pressure-Varying Abundances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added via ``RetrievalConfig.add_pressure_varying_line_species()``.  Enables a vertically
structured abundance profile parameterised by a set of nodes in log-pressure space.

.. list-table::
   :widths: 30 15 10 45
   :header-rows: 1

   * - Key pattern
     - Type
     - Required
     - Description
   * - ``{species}_n_abundance_nodes``
     - integer
     - Fixed
     - Total number of abundance nodes (set automatically).
   * - ``{species}_n_pressure_nodes``
     - integer
     - Fixed
     - Number of free interior pressure nodes.
   * - ``{species}_interpolation_mode``
     - string
     - Fixed
     - Interpolation scheme: ``'relative'``, ``'absolute'``, or ``'fixed'``.
   * - ``{species}_pressure_node_{i}``
     - log\ :sub:`10`\ (bar)
     - Free (unless ``'fixed'`` mode)
     - Log pressure of the *i*-th interior node, for *i* = 0, …, n_pressure_nodes − 1.
   * - ``{species}_abundance_node_{i}``
     - log\ :sub:`10`\ (mass fraction)
     - Free
     - Log\ :sub:`10` mass fraction at abundance node *i*, for *i* = 0, …, n_abundance_nodes.

----

.. _params-clouds:

Cloud Parameters
-----------------

Cloud parameters are split into two groups: **physical cloud parameters** (microphysical
clouds with real scattering properties) and **simple cloud / haze parameters** (gray opacity
decks and power-law hazes used primarily in transmission spectroscopy).

.. _cloud-physical:

Physical Cloud Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

These parameters are used by all model functions that call
``clouds.setup_clouds_from_legacy_dict()``.

**Sedimentation**

.. list-table::
   :widths: 25 15 10 50
   :header-rows: 1

   * - Key
     - Units
     - Required
     - Description
   * - ``fsed``
     - —
     - Yes (if clouds present)
     - Sedimentation efficiency parameter.  Can be specified globally or per-species via
       ``fsed_{cloud_name}`` (e.g. ``fsed_MgSiO3``).

**Particle size distribution** — choose exactly one:

.. list-table::
   :widths: 25 15 10 50
   :header-rows: 1

   * - Key
     - Units
     - Exclusion group
     - Description
   * - ``sigma_lnorm``
     - —
     - 1 of 2
     - Width σ of the log-normal particle size distribution.
   * - ``b_hans``
     - —
     - 1 of 2
     - Width parameter *b* of the Hansen distribution.  Can be specified per-species
       via ``b_hans_{cloud_name}``.

**Particle radius** — choose exactly one:

.. list-table::
   :widths: 25 15 10 50
   :header-rows: 1

   * - Key
     - Units
     - Exclusion group
     - Description
   * - ``log_kzz``
     - log\ :sub:`10`\ (cm\ :sup:`2` s\ :sup:`-1`\ )
     - 1 of 2
     - Log base-10 of the eddy diffusion coefficient.  Used to compute the equilibrium
       particle radius together with ``fsed``.
   * - ``log_cloud_radius_{cloud_name}``
     - log\ :sub:`10`\ (cm)
     - 1 of 2
     - Log base-10 of the central particle radius for the named species, e.g.
       ``log_cloud_radius_MgSiO3``.  Overrides the Kzz-based calculation.

**Cloud base abundance** — choose exactly one:

.. list-table::
   :widths: 25 15 10 50
   :header-rows: 1

   * - Key
     - Units
     - Exclusion group
     - Description
   * - ``eq_scaling_{cloud_name}``
     - —
     - 1 of 2
     - Multiplicative scaling factor applied to the equilibrium cloud abundance.
       E.g. ``eq_scaling_MgSiO3``.  Typical prior range: (−3, 1) in log.
   * - ``log_X_cb_{cloud_name}``
     - log\ :sub:`10`\ (mass fraction)
     - 1 of 2
     - Log base-10 mass fraction of the cloud species at the cloud base.
       E.g. ``log_X_cb_MgSiO3``.

**Cloud base pressure** (optional):

.. list-table::
   :widths: 25 15 10 50
   :header-rows: 1

   * - Key
     - Units
     - Required
     - Description
   * - ``log_Pbase_{cloud_name}``
     - log\ :sub:`10`\ (bar)
     - Optional
     - Log base-10 of the cloud base pressure for the named species.
       If omitted, the base is derived from the condensation curve.

.. _cloud-patchy:

**Cloud coverage fractions** (optional):

.. list-table::
   :widths: 25 15 10 50
   :header-rows: 1

   * - Key
     - Units
     - Required
     - Description
   * - ``cloud_fraction``
     - —  (0–1)
     - Optional
     - Fraction of the planetary disc covered by clouds.  The complementary fraction
       ``(1 − cloud_fraction)`` is cloud-free.  Also accepted as ``patchiness``.
   * - ``complete_coverage_clouds``
     - list
     - Optional
     - List of cloud species with 100 % coverage even in the nominally clear column.

.. _cloud-simple:

Simple Cloud / Haze Parameters (Transmission)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used by transmission model functions that call
``clouds.setup_simple_clouds_hazes_from_legacy_dict()``.  These parameters are
intended as simpler alternatives to the full microphysical cloud treatment.

**Gray cloud deck** — the following keys are all aliases for the same ``pcloud`` quantity:

.. list-table::
   :widths: 30 15 10 45
   :header-rows: 1

   * - Key
     - Units
     - Notes
     - Description
   * - ``log_Pcloud``
     - log\ :sub:`10`\ (bar)
     - Preferred
     - Log base-10 pressure of the opaque gray cloud deck.
   * - ``Pcloud``
     - bar
     - Alias
     - Cloud deck pressure in linear units.
   * - ``log_opaque_cloud_top_pressure``
     - log\ :sub:`10`\ (bar)
     - Alias
     - Alternative name for the log-pressure cloud deck.
   * - ``opaque_cloud_top_pressure``
     - bar
     - Alias
     - Alternative name for the linear-pressure cloud deck.

**Power-law haze:**

.. list-table::
   :widths: 30 15 10 45
   :header-rows: 1

   * - Key
     - Units
     - Notes
     - Description
   * - ``power_law_opacity_350nm``
     - cm\ :sup:`2` g\ :sup:`-1`
     - Also ``kappa_0`` (log\ :sub:`10` form)
     - Haze scattering opacity at 350 nm.
   * - ``power_law_opacity_coefficient``
     - —
     - Also ``gamma_scat``
     - Wavelength power-law index for the haze opacity, κ(λ) ∝ λ\ :sup:`γ`.
   * - ``haze_factor``
     - —
     - Also ``log_haze_factor`` (log\ :sub:`10` form)
     - Multiplicative scaling factor applied to the Rayleigh scattering cross-section.

.. note::

   Simple (gray/haze) cloud parameters and physical microphysical cloud parameters can both be
   specified simultaneously in transmission models.  Both ``pcloud`` and ``fsed``-based clouds
   can therefore coexist.

----

.. _params-disk-extinction:

Optional Emission Model Parameters
------------------------------------

.. list-table::
   :widths: 25 15 10 50
   :header-rows: 1

   * - Key
     - Units
     - Applicable to
     - Description
   * - ``contribution``
     - bool
     - All
     - If ``True``, the model returns the emission/transmission contribution function
       as a third output.  Set internally by the retrieval when in evaluate mode.
   * - ``T_disk_blackbody``
     - K
     - Emission
     - Temperature of a circumplanetary blackbody disk component added to the flux.
   * - ``disk_radius``
     - cm
     - Emission
     - Radius of the circumplanetary disk component.  Required when ``T_disk_blackbody``
       is set.
   * - ``v_band_extinction``
     - mag
     - Emission
     - V-band extinction applied to the model spectrum as a reddening correction.
   * - ``v_band_reddening``
     - —
     - Emission
     - V-band reddening parameter R\ :sub:`V`.  Required when ``v_band_extinction`` is set.
   * - ``stellar_radius``
     - cm
     - Emission
     - Stellar radius used to normalise the stellar flux (when the star is included).

----

.. _params-data:

Per-Dataset Instrumental / Calibration Parameters
---------------------------------------------------

These parameters are added for each dataset individually.  The ``{data_name}`` prefix must
match the key used when adding the dataset to the ``RetrievalConfig``.

.. list-table::
   :widths: 30 15 10 45
   :header-rows: 1

   * - Key
     - Units
     - Condition
     - Description
   * - ``{data_name}_scale_factor``
     - —
     - ``scale_flux=True`` or ``scale_uncertainties=True``
     - Multiplicative scale factor applied to the model flux or the uncertainties.
   * - ``{data_name}_offset``
     - flux units
     - ``fit_flux_offset=True``
     - Additive constant flux offset applied to the model.
   * - ``{data_name}_b``
     - —
     - Manual
     - Log\ :sub:`10` of a variance floor added in quadrature to the data uncertainties.
   * - ``{data_name}_radial_velocity``
     - km s\ :sup:`-1`
     - Manual
     - Per-dataset radial velocity shift applied as a wavelength stretch.
   * - ``system_radial_velocity``
     - km s\ :sup:`-1`
     - Manual
     - Global radial velocity offset applied to all datasets that do not have their own.
   * - ``{data_name}_R_slope``
     - —
     - ``fit_instrumental_resolution=True``
     - Slope of the wavelength-dependent instrumental resolving power R(λ).
   * - ``{data_name}_R_int``
     - —
     - ``fit_instrumental_resolution=True``
     - Intercept of the wavelength-dependent instrumental resolving power R(λ).

----

.. _params-covariance:

Gaussian-Process Covariance Hyperparameters
---------------------------------------------

Added automatically when a dataset is included with ``fit_covariance=True``.
The ``covariance_mode`` argument to ``add_data()`` controls which GP kernels are active.

.. list-table::
   :widths: 30 15 10 45
   :header-rows: 1

   * - Key
     - Units
     - ``covariance_mode``
     - Description
   * - ``{data_name}_gp_global_amp``
     - flux units
     - ``'global'``, ``'global_local'``
     - Amplitude of the global GP kernel.  Prior is auto-set from data uncertainties.
   * - ``{data_name}_gp_global_length``
     - µm
     - ``'global'``, ``'global_local'``
     - Length scale of the global GP kernel.
   * - ``{data_name}_gp_local_{k}_amp``
     - flux units
     - ``'local'``, ``'global_local'``
     - Amplitude of the *k*-th local GP kernel (*k* = 1, …, ``n_local_covariance_kernels``).
   * - ``{data_name}_gp_local_{k}_length``
     - µm
     - ``'local'``, ``'global_local'``
     - Length scale of the *k*-th local GP kernel.
   * - ``{data_name}_gp_local_{k}_center``
     - µm
     - ``'local'``, ``'global_local'``
     - Central wavelength of the *k*-th local GP kernel.

Available ``covariance_mode`` values: ``'none'``, ``'global'``, ``'local'``, ``'global_local'``.

Available global kernel shapes (``global_covariance_kernel``): ``'squared_exponential'``, ``'matern32'``.

----

.. _params-uncertainties:

Uncertainties Mode Parameters
-------------------------------

The ``uncertainties_mode`` argument to :class:`~petitRADTRANS.retrieval.retrieval.Retrieval` controls
how observational uncertainties are handled.  Some modes introduce additional retrieved parameters.

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Mode
     - Extra parameters
     - Description
   * - ``"default"``
     - None
     - Uncertainties are taken from the data file and kept fixed throughout the retrieval.
   * - ``"optimize"``
     - None
     - Uncertainties are analytically optimised at each likelihood evaluation following
       Gibson et al. (2020, MNRAS 496, 1355).
   * - ``"retrieve"``
     - ``{data_name}_scale_factor``
     - Uncertainties are multiplied by a retrieved coefficient.
   * - ``"retrieve_add"``
     - ``{data_name}_offset``
     - A fixed scalar term is added to all uncertainties and is retrieved.

----

.. _params-compatibility:

Parameter Compatibility Matrix
--------------------------------

The table below summarises which parameter groups can be used together.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Choice
     - Options (choose exactly one unless noted)
   * - **Gravity** (choose 2 of 3)
     - ``log_g``, ``planet_radius``, ``mass``
   * - **Chemistry mode**
     - Equilibrium: {``Fe/H``, ``C/O`` (or ``C`` + ``O``), optional ``log_pquench``}
       **or** Free: {per-species ``{species_name}``}
       **or** Mixed: equilibrium for background + free for selected species.
   * - **Particle size distribution** (choose 1 of 2)
     - ``sigma_lnorm`` (log-normal) **xor** ``b_hans`` (Hansen)
   * - **Particle radius** (choose 1 of 2)
     - ``log_kzz`` (Kzz-derived) **xor** ``log_cloud_radius_{cloud_name}`` (direct)
   * - **Cloud abundance** (choose 1 of 2)
     - ``eq_scaling_{cloud_name}`` (equilibrium scaling) **xor** ``log_X_cb_{cloud_name}`` (free)
   * - **Gray cloud deck** (all are aliases, at most one value)
     - ``log_Pcloud``, ``Pcloud``, ``log_opaque_cloud_top_pressure``, ``opaque_cloud_top_pressure``
   * - **Haze factor** (aliases)
     - ``haze_factor`` **or** ``log_haze_factor``
   * - **Haze opacity at 350 nm** (aliases)
     - ``power_law_opacity_350nm`` **or** ``kappa_0``
   * - **Haze slope** (aliases)
     - ``power_law_opacity_coefficient`` **or** ``gamma_scat``
   * - **Radial velocity** (per-dataset overrides global)
     - ``{data_name}_radial_velocity`` **or** ``system_radial_velocity`` (fallback)
   * - **C/O specification** (choose 1 of 2)
     - ``C/O`` (scalar ratio) **or** (``C`` + ``O``) (individual elemental abundances)

----

Prior Types for ``RetrievalParameter``
-----------------------------------------

When using the :class:`~petitRADTRANS.retrieval.parameter.RetrievalParameter` class
(used with ``Retrieval.from_data()``), the prior type and its arguments are specified
via ``prior_type`` and ``prior_parameters``.

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - ``prior_type``
     - ``prior_parameters`` meaning
     - Description
   * - ``'uniform'``
     - ``[lower, upper]``
     - Uniform distribution between ``lower`` and ``upper``.
   * - ``'log'``
     - ``[log10_lower, log10_upper]``
     - Uniform distribution in log\ :sub:`10` space.
   * - ``'gaussian'``
     - ``[mu, sigma]``
     - Normal distribution with mean ``mu`` and standard deviation ``sigma``.
   * - ``'log_gaussian'``
     - ``[mu, sigma]``
     - Log-normal distribution: Gaussian in log\ :sub:`10` space.
   * - ``'delta'``
     - ``[fixed_value, _]``
     - Delta function; the parameter is fixed to ``fixed_value``.
   * - ``'custom'``
     - function
     - Arbitrary callable mapping the unit hypercube [0, 1] to physical space.

----

.. _params-infrastructure:

Internal Infrastructure Parameters
-------------------------------------

These parameters are created automatically by ``RetrievalConfig`` and should not be modified
by the user.

.. list-table::
   :widths: 25 15 10 50
   :header-rows: 1

   * - Key
     - Type
     - Free?
     - Description
   * - ``pressure_scaling``
     - integer
     - No
     - Adaptive mesh refinement (AMR) pressure-level resolution multiplier.
   * - ``pressure_width``
     - integer
     - No
     - Number of base-grid pressure levels replaced by the AMR grid around the cloud base.
   * - ``pressure_simple``
     - integer
     - No
     - Number of pressure levels in the nominal (non-AMR) grid.

----

Example: Minimal Emission Retrieval Setup
-------------------------------------------

The following example shows how to register a typical set of parameters for an emission
retrieval using the Guillot profile and equilibrium chemistry::

    from petitRADTRANS.retrieval.retrieval_config import RetrievalConfig
    import numpy as np

    pressures = np.logspace(-6, 3, 100)

    rc = RetrievalConfig(
        retrieval_name='my_emission_run',
        run_mode='retrieval',
        pressures=pressures,
    )

    # Planet properties (2 of 3 required)
    rc.add_parameter('log_g',        is_free_parameter=True,
                     transform_prior_cube_coordinate=lambda x: 2.0 + 3.0 * x)
    rc.add_parameter('planet_radius', is_free_parameter=False, value=1.2 * 6.991e9)

    # Guillot T-P profile
    rc.add_parameter('T_int',  is_free_parameter=True,
                     transform_prior_cube_coordinate=lambda x: 200 + 1800 * x)
    rc.add_parameter('T_equ',  is_free_parameter=True,
                     transform_prior_cube_coordinate=lambda x: 500 + 2000 * x)
    rc.add_parameter('gamma',  is_free_parameter=True,
                     transform_prior_cube_coordinate=lambda x: 0.01 + 1.99 * x)
    rc.add_parameter('log_kappa_IR', is_free_parameter=True,
                     transform_prior_cube_coordinate=lambda x: -3.0 + 4.0 * x)

    # Equilibrium chemistry
    rc.add_parameter('Fe/H', is_free_parameter=True,
                     transform_prior_cube_coordinate=lambda x: -1.5 + 3.0 * x)
    rc.add_parameter('C/O',  is_free_parameter=True,
                     transform_prior_cube_coordinate=lambda x: 0.1 + 1.5 * x)
    rc.add_parameter('log_pquench', is_free_parameter=True,
                     transform_prior_cube_coordinate=lambda x: -6.0 + 9.0 * x)

    # System distance (required for emission)
    rc.add_parameter('distance_to_system', is_free_parameter=False, value=10 * 3.086e18)
