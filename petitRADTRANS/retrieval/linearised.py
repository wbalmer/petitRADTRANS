"""Closed-form modulation solve for linearised time-series retrievals.

The time variation of a rotating object's spectrum is small -- of order one
percent -- so to first order the model is linear in the modulation and the
modulation amplitudes enter as a linear-Gaussian problem.  They can therefore be
integrated out ANALYTICALLY, leaving a sampler to explore only the static
reference state.  This module holds the algebra for that marginalisation.

The statistical model
---------------------
The module fits the LOG of the observed flux directly::

    ln d_ij = ln M_i(theta_ref) + a_i + sum_pm J_ip Phi_jm beta_pm + eps_ij

for wavelength ``i`` and epoch ``j``, with

``ln M_i(theta_ref)``
    the reference model spectrum, projected onto the observation grid;
``a_i ~ N(0, s^2)``
    a STATIC per-wavelength offset representing residual error in the forward
    model.  It is a nuisance: never sampled, always marginalised;
``J_ip = dln M_i / dtheta_p``
    the log Jacobian, the fractional change in flux per unit change in
    parameter ``p`` -- one "response spectrum" per modulated parameter;
``Phi_jm``
    the harmonic basis in time, anchored on the reference set;
``beta_pm``
    the modulation amplitudes, the quantity of interest;
``eps_ij ~ N(0, S_i)``
    measurement noise, with ``S_i`` diagonal (or AR(1)-correlated in time,
    see ``correlation_rho``).

Marginalising ``a_i`` analytically -- it is Gaussian and enters linearly -- gives
an effective noise covariance at each wavelength::

    Sigma_i = S_i + s^2 * 1 1^T

which is "diagonal plus rank one" and is inverted in closed form by the
Sherman-Morrison identity (see :func:`build_linear_system`).  Every contraction
below is therefore a weighted sum over epochs; no ``(n_time, n_time)`` matrix is
ever formed.

Argument roles
--------------
``setup``
    Evaluated once when the runtime builds the observation.  Concrete values;
    may raise Python exceptions.  ``reference_bin_indices``,
    ``reference_weights``, ``log_observation``, ``harmonic_outer_products``,
    ``smooth_wavelength_modes``.
``traced``
    Evaluated inside the likelihood, once per sample.  Every array argument may
    be a tracer.  ``reference_anchored_basis``, ``build_linear_system``,
    ``modulation_log_evidence``, ``solve_modulation``.
``diagnostic``
    Run once, off the sampling path, to validate or interpret a fit.
    ``amplitude_bias_projection``, ``evidence_prior_scan``,
    ``posterior_eigenspectrum``, ``predictive_log_density``,
    ``modulation_time_series``, ``mixture_moments``.

Conventions
-----------
``n_wavelength`` indexes the observation grid, ``n_time`` the epochs,
``n_response`` the columns of the log Jacobian (modulated parameters, plus any
nuisance response modes appended by the caller), and ``n_column = n_response *
2 * n_harmonics`` the amplitudes.  Amplitude columns are ordered
``(response, [c1, s1, c2, s2, ...])`` and flattened response-major, matching
``jnp.kron(outer(J, J), tt)``.
"""

import jax.numpy as jnp
from jax import Array, config
from jax.scipy.linalg import cho_solve

__all__ = [
    "amplitude_bias_projection",
    "amplitude_column_names",
    "amplitude_fixed_point",
    "basis_orthogonality",
    "build_linear_system",
    "collapse_reference_weights",
    "curvature_expansion_parameter",
    "curvature_linearising_exponent",
    "curvature_response_columns",
    "evidence_prior_scan",
    "harmonic_outer_products",
    "integer_period_window",
    "log_observation",
    "mixture_moments",
    "modulation_log_evidence",
    "modulation_time_series",
    "noise_log_determinant",
    "posterior_eigenspectrum",
    "predictive_log_density",
    "reference_anchored_basis",
    "reference_bin_indices",
    "reference_weights",
    "require_double_precision",
    "second_order_field",
    "smooth_wavelength_modes",
    "solve_modulation",
    "trigonometric_resampling_matrix",
]

#: A quantity that may be a Python scalar at setup and a tracer under jit.
#: ``period`` and ``offset_variance`` are such arguments.
Scalar = Array | float


# ---------------------------------------------------------------------------
# 0. Precision policy
# ---------------------------------------------------------------------------

def require_double_precision() -> None:
    """Force double precision, single precision is insufficient for linearised time-series retrievals.
    """
    if not config.jax_enable_x64:
        raise RuntimeError(
            "Linearised time-series retrievals require double precision. "
            "Add 'jax.config.update(\"jax_enable_x64\", True)' before importing "
            "petitRADTRANS. The marginal likelihood is compared between models "
            "at the level of a few nats; single precision leaves a ~0.03 nat "
            "floor on it."
        )


# ---------------------------------------------------------------------------
# 1. Reference set  (setup)
# ---------------------------------------------------------------------------

def integer_period_window(
    observation_times: Array,
    period: float,
    n_periods: int = 1,
    start: int = 0,
) -> Array:
    """Epoch indices spanning a WHOLE number of rotations.

    Args:
        period:
            The period to crop against.  Concrete, not traced.  Cropping is a choice
            about which data to use and must not change while sampling.
        n_periods:
            How many whole rotations to retain.
        start:
            First epoch of the window, for choosing which rotation to keep.

    Returns:
        ``(n_kept,)`` int array of consecutive indices.
    """
    if n_periods < 1:
        raise ValueError(f"n_periods must be >= 1, got {n_periods}")
    if observation_times.ndim != 1:
        raise ValueError("observation_times must be 1-D")
    n_time = observation_times.shape[0]
    if not 0 <= start < n_time:
        raise ValueError(f"start must lie in [0, {n_time}), got {start}")

    # Even sampling makes the window length an integer multiple of the period
    # when n*dt = n_periods*P; choose n by rounding, then verify.
    cadence = float(jnp.median(jnp.diff(observation_times)))
    if not cadence > 0.0:
        raise ValueError("observation_times must be increasing")
    n_kept = int(round(n_periods * float(period) / cadence))
    if n_kept < 2 * n_periods:
        raise ValueError(
            f"{n_periods} period(s) spans only {n_kept} epochs at this cadence"
        )
    if start + n_kept > n_time:
        raise ValueError(
            f"{n_periods} period(s) needs {n_kept} epochs from index {start}, "
            f"but only {n_time - start} remain; the baseline covers "
            f"{float(jnp.ptp(observation_times)) / float(period):.3f} periods"
        )
    return jnp.arange(start, start + n_kept)


def trigonometric_resampling_matrix(
    node_times: Array,
    target_times: Array,
    period: Scalar,
    n_harmonics: int,
) -> Array:
    """Exact interpolation in time for a band-limited signal, as one matrix.

    Setup (concrete node/target grids), though it is differentiable in
    ``period`` if that is ever wanted.

    The spectrum's time dependence is band-limited, and a band-limited periodic
    signal is recovered EXACTLY from``2H + 1`` samples per period by fitting
    the harmonics. This builds the ``(n_target, n_node)`` matrix that does it::

        interpolated = matrix @ spectra_at_nodes

    This requires ``2 * n_harmonics + 1`` samples at DISTINCT ROTATIONAL PHASES.

    Choosing ``n_harmonics``.  The MODEL's time dependence is not band-limited
    at ``K`` even though the parameters are: the spectrum is a nonlinear
    function of the parameters, so a modulation truncated at ``K`` produces
    spectral harmonics up to ``2K`` at second order and beyond.  Pass ``2 * K``
    to capture the second-order content, which needs ``4K + 1`` nodes -- 13 for
    ``K = 3``, comfortably inside the 16 already being computed.

    Args:
        node_times, target_times:
            ``(n_node,)`` and ``(n_target,)``, seconds, same origin.
        period:
            Seconds.
        n_harmonics:
            Highest harmonic to reconstruct.  Static.

    Returns:
        ``(n_target, n_node)``.
    """
    n_node = node_times.shape[0]
    if n_harmonics < 1:
        raise ValueError(f"n_harmonics must be >= 1, got {n_harmonics}")
    if n_node < 2 * n_harmonics + 1:
        raise ValueError(
            f"{n_node} nodes cannot determine {n_harmonics} harmonics; "
            f"at least {2 * n_harmonics + 1} are required"
        )

    def design_matrix(times: Array) -> Array:
        angle = (2.0 * jnp.pi / period) * times[:, None] * jnp.arange(
            1, n_harmonics + 1, dtype=times.dtype
        )[None, :]
        return jnp.concatenate(
            [
                jnp.ones((times.shape[0], 1), dtype=times.dtype),
                jnp.cos(angle),
                jnp.sin(angle),
            ],
            axis=1,
        )

    at_nodes = design_matrix(node_times)

    # Degeneracy check. Skipped silently under tracing, where the singular
    # values are not available as numbers; the node grid is a setup-time choice,
    # so in practice this runs on the concrete arrays that define it.
    try:
        singular = jnp.linalg.svd(at_nodes, compute_uv=False)
        ratio = float(singular[-1] / singular[0])
    except Exception:                      # traced period or node times
        ratio = None
    if ratio is not None and ratio < 1e-10:
        raise ValueError(
            f"the {n_node} node times do not determine {n_harmonics} harmonics "
            f"(design matrix condition number {1 / max(ratio, 1e-300):.2e}); the usual "
            "cause is duplicated rotational phases, for example both endpoints "
            "of a grid spanning exactly one period"
        )

    # Least squares rather than a square solve: with more nodes than harmonics
    # the extra samples average down any content above the truncation instead of
    # aliasing it, which is the behaviour we want when the model is only
    # approximately band-limited.
    return design_matrix(target_times) @ jnp.linalg.pinv(at_nodes)


def basis_orthogonality(basis: Array) -> tuple[Array, Array]:
    """How close the basis is to orthogonal, in the space the fit works in.

    Diagnostic.  Returns ``(max_off_diagonal, max_column_mean)``, both relative,
    both zero for a perfectly orthogonal zero-mean basis.  The first governs how
    much one amplitude's error leaks into another; the second how much a
    constant offset leaks into the amplitudes.

    The column means are removed before the Gram matrix is formed, and that is
    not a cosmetic choice.  Anchoring displaces the basis by a rank-one term,
    ``Phi = psi - 1 c^T`` with ``c = <psi>_ref``, and for a narrow reference bin
    at low phase ``c`` is of order one.  That term dominates ``Phi^T Phi`` and
    hides the orthogonality entirely.
    """
    centred = basis - jnp.mean(basis, axis=0, keepdims=True)
    gram = centred.T @ centred
    diagonal = jnp.diag(gram)
    off_diagonal = gram - jnp.diag(diagonal)
    return (
        jnp.max(jnp.abs(off_diagonal)) / jnp.max(jnp.abs(diagonal)),
        jnp.max(jnp.abs(jnp.mean(basis, axis=0))),
    )


def reference_bin_indices(
    observation_times: Array,
    period: float,
    n_phase_bins: int = 24,
    reference_bin: int = 0,
) -> Array:
    """Epoch indices forming the reference set, as one bin of equal phase count.

    Setup only.  The reference set is deliberately held FIXED at a nominal
    period rather than re-derived as the period varies: re-binning by phase
    would move epochs between bins discretely and put a step discontinuity into
    an otherwise smooth objective.  ``reference_anchored_basis`` then tracks the
    fitted period continuously over this fixed index set.

    Args:
        observation_times:
            ``(n_time,)`` float array, seconds, any origin.
        period:
            Nominal rotation period in seconds.  Concrete, not traced: it selects
            which epochs form the set, which is a structural choice.

    Returns:
        ``(n_reference,)`` int array, sorted ascending.
    """
    if n_phase_bins < 1:
        raise ValueError(f"n_phase_bins must be >= 1, got {n_phase_bins}")
    if not 0 <= reference_bin < n_phase_bins:
        raise ValueError(
            f"reference_bin must lie in [0, {n_phase_bins}), got {reference_bin}"
        )
    if observation_times.ndim != 1:
        raise ValueError(
            f"observation_times must be 1-D, got ndim={observation_times.ndim}"
        )

    n_time = observation_times.shape[0]
    phase = jnp.mod((observation_times - observation_times[0]) / period, 1.0)
    order = jnp.argsort(phase)
    edges = jnp.linspace(0, n_time, n_phase_bins + 1).astype(int)
    lo, hi = int(edges[reference_bin]), int(edges[reference_bin + 1])
    if hi <= lo:
        raise ValueError(
            f"reference bin {reference_bin} of {n_phase_bins} is empty for "
            f"{n_time} epochs; reduce n_phase_bins"
        )
    return jnp.sort(order[lo:hi])


def reference_weights(uncertainties: Array, reference_indices: Array) -> Array:
    """Inverse-variance weights over the reference set, normalised per wavelength.

    Setup only.  Used to define what "the state on the reference set" means when
    the reference epochs have unequal signal-to-noise.

    Args:
        uncertainties:
            ``(n_time, n_wavelength)``.
        reference_indices:
            ``(n_reference,)`` from :func:`reference_bin_indices`.

    Returns:
        ``(n_reference, n_wavelength)``, columns summing to one.
    """
    weights = 1.0 / jnp.square(uncertainties[reference_indices])
    return weights / jnp.sum(weights, axis=0, keepdims=True)


def collapse_reference_weights(weights: Array) -> Array:
    """Collapse per-wavelength reference weights to one vector over epochs.

    The anchoring weights are formally wavelength dependent, because the
    signal-to-noise of an epoch varies across the spectrum.  A wavelength
    dependent anchor would make the design non-separable and forfeit the
    Kronecker structure that :func:`build_linear_system` depends on, so the
    weights are collapsed to a single vector over epochs.

    Under the OFFSET-marginalised likelihood this collapse is harmless:
    two anchorings differ by a constant shift of the basis columns, which changes
    the model by a quantity that is constant in time, and a time-constant quantity
    is absorbed by the per-wavelength offset ``a_i``. The residual sensitivity is
    suppressed by ``1/(1 + s^2 G_i)`` and vanishes as the offset prior widens.

    Args:
        weights:
            ``(n_reference, n_wavelength)`` from :func:`reference_weights`.

    Returns:
        ``(n_reference,)``, summing to one.
    """
    collapsed = jnp.mean(weights, axis=1)
    return collapsed / jnp.sum(collapsed)


# ---------------------------------------------------------------------------
# 2. Temporal basis  (traced)
# ---------------------------------------------------------------------------

def reference_anchored_basis(
    observation_times: Array,
    period: Scalar,
    reference_indices: Array,
    n_harmonics: int,
    reference_weight_vector: Array,
) -> Array:
    """Harmonic basis anchored on the reference SET.

        Phi_j = psi(t_j) - <psi>_ref,    psi_k(t) = (cos k w t, sin k w t)

    ``<psi>_ref`` is the weighted mean of the raw basis over the reference
    epochs, so every column vanishes there by construction and the reference
    state is unambiguously the state on that set.

    Differentiable in ``period``, which is the point: the period enters the
    likelihood ONLY through this basis -- never through the reference spectrum
    or the Jacobian -- so varying it costs no radiative transfer.  Note that the
    likelihood is NOT linear in the period, so the period belongs in the outer
    sampler alongside ``theta_ref``; only the amplitudes are marginalised.

    Args:
        observation_times:
            ``(n_time,)``.
        period:
            Seconds.  May be traced.
        reference_indices:
            ``(n_reference,)``.
        n_harmonics:
            Static; sets the column count.
        reference_weight_vector:
            ``(n_reference,)``, already normalised to sum to one; from
            :func:`collapse_reference_weights`.

    Returns:
        ``(n_time, 2 * n_harmonics)``.
    """
    orders = jnp.arange(1, n_harmonics + 1, dtype=observation_times.dtype)
    angle = (2.0 * jnp.pi / period) * observation_times[:, None] * orders[None, :]
    # (n_time, K, 2) -> (n_time, 2K) puts the columns in [c1, s1, c2, s2, ...]
    # order, which is the order amplitude_column_names reports and the order the
    # flattened amplitude vector assumes.
    psi = jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=-1).reshape(
        observation_times.shape[0], 2 * n_harmonics
    )
    psi_reference = reference_weight_vector @ psi[reference_indices]
    return psi - psi_reference[None, :]


def harmonic_outer_products(basis: Array) -> tuple[Array, Array]:
    """Per-epoch outer products of the basis, flattened for a single matmul.

    Traced (the basis depends on the period), but cheap: ``n_time * 4K^2``.

    :func:`build_linear_system` needs, at each wavelength, the weighted sums

        sum_j  a_ij Phi_jm Phi_jn        (diagonal noise term)
        sum_j  b_ij (Phi_jm Phi_j+1,n + Phi_j+1,m Phi_jn)   (AR(1) term)

    Precomputing the two epoch-indexed outer products turns each of those into a
    single ``(n_wavelength, n_time) @ (n_time, 4K^2)`` matrix product.

    Returns:
        ``(same_epoch, adjacent_epoch)`` with shapes ``(n_time, 4K^2)`` and
        ``(n_time - 1, 4K^2)``.
    """
    n_time, n_mode = basis.shape
    same = (basis[:, :, None] * basis[:, None, :]).reshape(n_time, n_mode * n_mode)
    cross = basis[:-1, :, None] * basis[1:, None, :]
    adjacent = (cross + cross.transpose(0, 2, 1)).reshape(n_time - 1, n_mode * n_mode)
    return same, adjacent


def amplitude_column_names(
    modulated_parameters: tuple[str, ...], n_harmonics: int
) -> tuple[str, ...]:
    """Labels for the flattened amplitude vector, in solve order."""
    return tuple(
        f"{name}:{trig}{order}"
        for name in modulated_parameters
        for order in range(1, n_harmonics + 1)
        for trig in ("c", "s")
    )


# ---------------------------------------------------------------------------
# 3. Data transformation  (setup)
# ---------------------------------------------------------------------------

def log_observation(spectrum: Array, uncertainties: Array) -> tuple[Array, Array]:
    """Log of the observed flux and the variance of that log.

    Setup only: this depends on the data alone, so the runtime computes it once
    and stores it on the observation rather than rebuilding it per likelihood
    call.

    For a small relative error, ``Var(ln x) ~= Var(x) / x^2``.  Two second-order
    effects are worth naming because the whole method is sensitive to biases that
    are coherent across the dataset:

    * ``E[ln d] = ln d_true - sigma^2 / (2 d^2)``, so taking a logarithm biases
      the data low.  The bias is common to every epoch at a given wavelength to
      the extent that ``sigma/d`` is constant in time, and a time-constant bias
      is absorbed exactly by the marginalised offset ``a_i``.  Only its time
      VARIATION survives, of order ``(sigma/d)^2 * d(sigma/d)/(sigma/d)``; at the
      signal-to-noise of this dataset that is ~1e-5 of the modulation signal.
    * using the observed ``d`` rather than the model in the denominator makes
      the weights slightly data-dependent, which upweights points that
      fluctuated high.  Same order, same conclusion.

    Both are recorded rather than corrected.  If the per-point signal-to-noise
    ever drops below ~30 they stop being negligible and the weights should be
    formed from the model spectrum instead.

    Args:
        spectrum, uncertainties:
            ``(n_time, n_wavelength)``, strictly positive fluxes.

    Returns:
        ``(log_spectrum, var_point)``, both ``(n_wavelength, n_time)`` -- transposed
        relative to the input, because every downstream contraction runs over time
        within a single wavelength and that ordering keeps the summed axis
        contiguous.
    """
    if spectrum.shape != uncertainties.shape:
        raise ValueError(
            f"spectrum {spectrum.shape} and uncertainties {uncertainties.shape} differ"
        )
    if spectrum.ndim != 2:
        raise ValueError(
            f"expected a (n_time, n_wavelength) spectrum, got ndim={spectrum.ndim}"
        )
    return jnp.log(spectrum).T, jnp.square(uncertainties / spectrum).T


def smooth_wavelength_modes(wavelengths: Array, n_modes: int) -> Array:
    """Orthonormal smooth response spectra, for absorbing model error.

    Setup only.

    The per-wavelength offset ``a_i`` absorbs STATIC forward-model error exactly,
    however it is correlated in wavelength, because each wavelength gets its own
    free offset.  What it cannot absorb is model error that VARIES IN TIME and is
    smooth in wavelength -- for example a modulated cloud whose spectral response
    the frozen-cloud model cannot express.  That is the residual structure behind
    the measured disagreement between the blue and red halves of the spectrum.

    This returns a low-order polynomial basis in ``ln(wavelength)``, orthonormal
    over the observation grid, intended to be concatenated onto the physical log
    Jacobian.  Those extra columns then acquire their own harmonic amplitudes,
    which are marginalised along with the physical ones and reported separately.
    The model stays exactly linear-Gaussian -- nothing about the closed form is
    given up -- and the physical amplitudes acquire the extra uncertainty that
    comes from not knowing how much of the signal a smooth time-varying
    component could have produced.

    Mode 0 is constant in wavelength: a GREY time-varying component, which is the
    single most useful nuisance mode, since grey variability is exactly what a
    cloud produces and what a temperature response does not.

    Args:
        wavelengths:
            ``(n_wavelength,)``, strictly positive, in any consistent unit.
        n_modes:
            Number of polynomial orders, starting at 0 (constant).  Static.

    Returns:
        ``(n_wavelength, n_modes)`` with orthonormal columns.
    """
    if n_modes < 1:
        raise ValueError(f"n_modes must be >= 1, got {n_modes}")
    if wavelengths.ndim != 1:
        raise ValueError(f"wavelengths must be 1-D, got ndim={wavelengths.ndim}")

    x = jnp.log(wavelengths)
    x = 2.0 * (x - jnp.min(x)) / (jnp.max(x) - jnp.min(x)) - 1.0
    powers = jnp.stack([x**k for k in range(n_modes)], axis=1)
    # A raw power basis is badly conditioned; orthonormalise it. The Q factor of
    # a thin QR is the Gram-Schmidt result without the loss of orthogonality.
    q, _ = jnp.linalg.qr(powers)
    return q


# ---------------------------------------------------------------------------
# 4. Noise model
# ---------------------------------------------------------------------------

def _band_weights(
    var_point: Array, correlation_rho: Scalar
) -> tuple[Array, Array, Array, Array]:
    """Decompose the noise precision into diagonal and adjacent-epoch weights.

    The measurement noise at wavelength ``i`` is modelled as

        S_i = D_i^{1/2} R D_i^{1/2},   D_i = diag(var_point_i)

    with ``R`` the correlation matrix of a first-order autoregressive process,
    ``R_jk = rho^{|j-k|}``.  ``rho = 0`` recovers independent noise exactly.

    An AR(1) process is the discrete analogue of an exponentially decaying
    correlation, ``exp(-|dt|/tau)`` with ``rho = exp(-1/tau)`` for ``tau`` in
    units of epochs.  It is used here because its INVERSE is tridiagonal in
    closed form -- it has no long-range terms at all -- so a correlated noise
    model costs no more than an uncorrelated one.  For a length-``n`` chain

        R^-1 = 1/(1-rho^2) * tridiag(-rho, [1, 1+rho^2, ..., 1+rho^2, 1], -rho)
        ln|R| = (n-1) ln(1-rho^2)

    (Both are standard; they follow from writing the process as
    ``x_j = rho x_{j-1} + noise`` and reading off the quadratic form.)

    The precision is then ``T_i = D_i^{-1/2} R^-1 D_i^{-1/2}``, and for any two
    length-``n_time`` vectors

        u^T T_i v = sum_j a_ij u_j v_j + sum_j b_ij (u_j v_j+1 + u_j+1 v_j)

    which is what the two returned weight arrays hold.

    Returns:
        ``(diagonal, adjacent, precision_ones, total_precision)``:
        ``diagonal`` ``(n_wavelength, n_time)``, ``adjacent``
        ``(n_wavelength, n_time - 1)``, ``precision_ones = T_i 1``
        ``(n_wavelength, n_time)``, and ``total_precision = 1^T T_i 1``
        ``(n_wavelength,)``.
    """
    inverse_sd = 1.0 / jnp.sqrt(var_point)                       # (n_wl, n_t)
    n_time = var_point.shape[1]
    scale = 1.0 / (1.0 - jnp.square(correlation_rho))

    main = jnp.full((n_time,), 1.0 + jnp.square(correlation_rho))
    main = main.at[0].set(1.0).at[-1].set(1.0)
    diagonal = scale * jnp.square(inverse_sd) * main[None, :]
    adjacent = (
        -correlation_rho * scale * inverse_sd[:, :-1] * inverse_sd[:, 1:]
    )                                                            # (n_wl, n_t-1)

    # T 1: row sums. Interior rows collect both neighbours, ends collect one.
    precision_ones = (
        diagonal
        + jnp.pad(adjacent, ((0, 0), (0, 1)))
        + jnp.pad(adjacent, ((0, 0), (1, 0)))
    )
    return diagonal, adjacent, precision_ones, jnp.sum(precision_ones, axis=1)


def noise_log_determinant(
    var_point: Array, offset_variance: Scalar, correlation_rho: Scalar = 0.0
) -> tuple[Array, int]:
    """``ln|Sigma|`` and the number of constrained data points.

    Setup/diagnostic.  Not needed on the sampling path, where it is a constant.

    The marginal likelihood carries a term ``-ln|Sigma|/2 - (n/2) ln 2pi`` that
    depends only on the data and the NOISE SETTINGS.  It is independent of
    ``theta_ref``, so it cancels in any comparison at fixed noise model and the
    sampler need not evaluate it.  It does NOT cancel when the offset width or
    the correlation are changed, so it must be added back before comparing those.

    With ``Sigma_i = S_i + s^2 1 1^T``, the matrix determinant lemma (the
    determinant counterpart of Sherman-Morrison, ``|A + uv^T| = |A|(1 + v^T A^-1
    u)``) gives

        ln|Sigma_i| = sum_j ln var_ij + (n_time - 1) ln(1 - rho^2)
                      + ln(1 + s^2 G_i),      G_i = 1^T T_i 1.

    An infinite offset width is an improper prior and has no normalisation: the
    ``ln(1 + s^2 G_i)`` term diverges.  The standard resolution -- the same one
    used in restricted maximum likelihood -- is to score only the part of the
    data the offset does not span: the divergent ``ln s^2`` is dropped, leaving
    ``ln G_i``, and one degree of freedom per wavelength is removed from the
    count.  That is what this returns.  The dropped piece is the same for every
    model on the same data, so no comparison is affected.

    Returns:
        ``(log_determinant, n_constrained)`` where ``n_constrained`` is
        ``n_wavelength * n_time`` for a finite offset width and
        ``n_wavelength * (n_time - 1)`` for an infinite one.
    """
    n_wavelength, n_time = var_point.shape
    _, _, _, total_precision = _band_weights(var_point, correlation_rho)

    base = jnp.sum(jnp.log(var_point)) + n_wavelength * (n_time - 1) * jnp.log(
        1.0 - jnp.square(correlation_rho)
    )
    if offset_variance == jnp.inf:
        return base + jnp.sum(jnp.log(total_precision)), n_wavelength * (n_time - 1)
    return (
        base + jnp.sum(jnp.log1p(offset_variance * total_precision)),
        n_wavelength * n_time,
    )


# ---------------------------------------------------------------------------
# 5. Normal equations  (traced, hot)
# ---------------------------------------------------------------------------

def build_linear_system(
    log_jacobian: Array,
    basis: Array,
    residual: Array,
    var_point: Array,
    offset_variance: Scalar,
    correlation_rho: Scalar = 0.0,
    basis_outer: tuple[Array, Array] | None = None,
) -> tuple[Array, Array, Array]:
    """Assemble the normal equations for the modulation amplitudes.

    The design separates, ``A[(i,j),(p,m)] = J_p(lam_i) Phi_jm``, so

        A^T Sigma^-1 A = sum_i (J_i J_i^T) kron (Phi^T Sigma_i^-1 Phi)

    and the full ``(n_wavelength * n_time, n_column)`` design -- 208,000 x 30 for
    a typical dataset -- is never formed.  Each wavelength contributes a
    ``(2K, 2K)`` inner product weighted by an ``(n_response, n_response)`` outer
    product.  ("Separable" means every entry of the design is a product of one
    factor depending only on wavelength and one depending only on time; that is
    what a Kronecker product is, and it is what makes the problem cheap.)

    **Inverting the noise covariance.**
    ``Sigma_i = S_i + s^2 1 1^T`` where ``S_i`` is the measurement noise
    (diagonal, or AR(1) in time) and the rank-one term comes from marginalising
    the static per-wavelength offset.  Writing ``T_i = S_i^-1``, the
    Sherman-Morrison identity

        (T^-1 + s^2 1 1^T)^-1 = T - s^2 (T 1)(T 1)^T / (1 + s^2 1^T T 1)

    turns the inverse into a rank-one update of something we can already apply,
    so no ``(n_time, n_time)`` matrix is formed or factorised.  Writing
    ``g_i = T_i 1`` and ``G_i = 1^T T_i 1``, for ANY two vectors u, v

        u^T Sigma_i^-1 v = u^T T_i v - kappa_i (g_i.u)(g_i.v),
        kappa_i = 1 / (1/s^2 + G_i).

    Args:
        log_jacobian:
            ``(n_wavelength, n_response)``, ``dln M / dtheta_p`` on the OBSERVATION
            grid.  Extra nuisance columns from :func:`smooth_wavelength_modes` may be
            concatenated on the right; this function does not distinguish them.
        basis:
            ``(n_time, 2 * n_harmonics)`` from :func:`reference_anchored_basis`.
        residual:
            ``(n_wavelength, n_time)``.  The data minus the static model in log
            space, ``ln d_ij - ln M_i(theta_ref)``, from :func:`log_observation`
            and the projected reference spectrum.
        var_point:
            ``(n_wavelength, n_time)`` variance of the log data.
        offset_variance:
            Scalar ``s^2``, the prior variance of the static per-wavelength offset;
            the square of the fractional static model error tolerated.  ``0.05**2``
            is the value the absolute-flux anchor previously used.  ``jnp.inf``
            recovers pure shape-only inference.
        correlation_rho:
            AR(1) coefficient of the noise in time, ``exp(-1/tau)`` for ``tau`` in
            epochs.  ``0.0`` is independent noise.
        basis_outer:
            Optional precomputed ``harmonic_outer_products(basis)``.  Pass it when
            the basis is reused across calls; omitted, it is recomputed.

    Returns:
        ``(normal_matrix, normal_vector, chi2_data)`` where
        ``chi2_data = r^T Sigma^-1 r`` before any modulation is subtracted.
    """
    n_wavelength, n_time = residual.shape
    if log_jacobian.shape[0] != n_wavelength:
        raise ValueError(
            f"log_jacobian has {log_jacobian.shape[0]} wavelengths, "
            f"residual has {n_wavelength}"
        )
    if basis.shape[0] != n_time:
        raise ValueError(f"basis has {basis.shape[0]} epochs, residual has {n_time}")
    if var_point.shape != residual.shape:
        raise ValueError(
            f"var_point {var_point.shape} and residual {residual.shape} differ"
        )

    n_response = log_jacobian.shape[1]
    n_mode = basis.shape[1]
    if basis_outer is None:
        basis_outer = harmonic_outer_products(basis)
    same_epoch, adjacent_epoch = basis_outer

    diagonal, adjacent, precision_ones, total_precision = _band_weights(
        var_point, correlation_rho
    )

    # Phi^T T_i Phi, as two matrix products against the precomputed outer
    # products. Each is (n_wl, n_t) @ (n_t, 4K^2).
    basis_inner = (diagonal @ same_epoch + adjacent @ adjacent_epoch).reshape(
        n_wavelength, n_mode, n_mode
    )
    # Phi^T T_i r
    basis_residual = (
        (diagonal * residual) @ basis
        + (adjacent * residual[:, :-1]) @ basis[1:]
        + (adjacent * residual[:, 1:]) @ basis[:-1]
    )
    # r^T T_i r
    residual_inner = jnp.sum(diagonal * jnp.square(residual), axis=1) + 2.0 * jnp.sum(
        adjacent * residual[:, :-1] * residual[:, 1:], axis=1
    )

    # Sherman-Morrison rank-one correction for the marginalised static offset.
    # The scalar is promoted to an array first so that the two limiting values
    # are reached by arithmetic rather than by a branch: Python would raise on
    # 1.0/0.0, whereas the array reciprocal gives the infinity the algebra wants.
    # (Scalars are exempt from the module's no-coercion rule; the rule exists to
    # keep large arrays off the per-call conversion path.)
    inverse_offset = jnp.reciprocal(
        jnp.asarray(offset_variance, dtype=var_point.dtype)
    )
    correction = 1.0 / (inverse_offset + total_precision)          # (n_wl,)
    ones_basis = precision_ones @ basis                            # (n_wl, 2K)
    ones_residual = jnp.sum(precision_ones * residual, axis=1)     # (n_wl,)

    basis_inner -= correction[:, None, None] * (
        ones_basis[:, :, None] * ones_basis[:, None, :]
    )
    basis_residual -= correction[:, None] * ones_residual[:, None] * ones_basis
    chi2_data = jnp.sum(residual_inner - correction * jnp.square(ones_residual))

    parameter_outer = log_jacobian[:, :, None] * log_jacobian[:, None, :]
    normal_matrix = jnp.einsum(
        "ipq,imn->pmqn", parameter_outer, basis_inner
    ).reshape(n_response * n_mode, n_response * n_mode)
    normal_vector = jnp.einsum("ip,im->pm", log_jacobian, basis_residual).reshape(-1)

    return normal_matrix, normal_vector, chi2_data


# ---------------------------------------------------------------------------
# 6. Solve and evidence  (traced)
# ---------------------------------------------------------------------------

def _posterior_cholesky(normal_matrix: Array, prior_variance: Array) -> Array:
    """Lower Cholesky factor of ``A^T Sigma^-1 A + Sigma_beta^-1``.

    ``prior_variance`` must already be ``(n_column,)``; it is not broadcast.  The
    prior must also be PROPER: under a flat prior the evidence is unnormalised
    and can be increased without limit by driving the Jacobian toward
    degeneracy, and an optimiser given that freedom walks into a singular design
    and fails here.
    """
    return jnp.linalg.cholesky(normal_matrix + jnp.diag(1.0 / prior_variance))


def solve_modulation(
    normal_matrix: Array,
    normal_vector: Array,
    prior_variance: Array,
    n_response: int,
    n_harmonics: int,
) -> tuple[Array, Array]:
    """Posterior mean and covariance of the amplitudes -- the exact answer.

        C = (A^T Sigma^-1 A + Sigma_beta^-1)^-1,   beta = C A^T Sigma^-1 r

    This is the Rodgers optimal-estimation linear retrieval, with ``A`` the
    weighting-function matrix and ``Sigma_beta`` the a-priori covariance.  No
    sampler is involved and no iteration is required: the likelihood is
    quadratic in the amplitudes, so the posterior is Gaussian and available in
    closed form.

    The covariance costs an ``(n_column, n_column)`` triangular solve that the
    evidence does not need, so the sampling path should call
    :func:`modulation_log_evidence` instead and reach for this only when a
    per-sample posterior is actually being retained.

    Args:
        prior_variance:
            ``(n_column,)``, strictly positive.
        n_response, n_harmonics:
            Static; they only reshape the mean.

    Returns:
        ``(mean, covariance)`` with ``mean`` shaped ``(n_response, 2*n_harmonics)``
        and ``covariance`` ``(n_column, n_column)``.
    """
    cholesky = _posterior_cholesky(normal_matrix, prior_variance)
    mean = cho_solve((cholesky, True), normal_vector)
    identity = jnp.eye(normal_matrix.shape[0], dtype=normal_matrix.dtype)
    covariance = cho_solve((cholesky, True), identity)
    return mean.reshape(n_response, 2 * n_harmonics), covariance


def modulation_log_evidence(
    normal_matrix: Array,
    normal_vector: Array,
    chi2_data: Array,
    prior_variance: Array,
    noise_normalisation: Scalar = 0.0,
) -> tuple[Array, Array]:
    """Marginal likelihood of the modulation, amplitudes integrated out.

        ln Z = -chi2/2 - ln|P|/2 - ln|Sigma_beta|/2 + noise_normalisation

    where ``noise_normalisation`` is ``-ln|Sigma|/2 - (n/2) ln 2pi``.

    This is the hot path: it is what a sampler evaluates at every step, and it
    deliberately does NOT form the posterior covariance.

    Two terms are easy to lose and both corrupt model comparison silently.

    ``-ln|Sigma_beta|/2`` depends on the NUMBER of amplitudes and on their prior
    widths, so it cancels only between variants that share both -- never true
    when comparing, say, temperature-modulated against cloud-modulated models.
    For a five-parameter set at K=3 it is about -46 nats.

    ``noise_normalisation`` is independent of ``theta_ref`` and so defaults to
    zero, which is correct and free for a sampler exploring one model on fixed
    data.  It is NOT independent of the noise settings, so it must be supplied
    from :func:`noise_log_determinant` whenever the offset width or the
    correlation differ between the things being compared.

    **A caution about what this quantity can and cannot settle.**
    ``ln Z`` here is conditional on the amplitude priors being right.  Widening
    one parameter's prior by a factor of ten moves its variant by
    ``2K ln 10 ~ 14`` nats with no change in the data or the physics.  That is a
    general property of marginal likelihoods with weakly informative priors, and
    it bites hardest in exactly the comparison this framework exists to make.
    Never report a Bayes factor from this function without running
    :func:`evidence_prior_scan` and confirming the sign survives; where it does
    not, :func:`predictive_log_density` answers the same question without
    depending on the prior width at all.

    Returns:
        ``(log_evidence, chi2)`` where ``chi2`` is evaluated at the posterior mean.
    """
    cholesky = _posterior_cholesky(normal_matrix, prior_variance)
    mean = cho_solve((cholesky, True), normal_vector)
    chi2 = chi2_data - normal_vector @ mean
    log_determinant_precision = 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky)))
    log_determinant_prior = jnp.sum(jnp.log(prior_variance))
    log_evidence = (
        -0.5 * (chi2 + log_determinant_precision + log_determinant_prior)
        + noise_normalisation
    )
    return log_evidence, chi2


# ---------------------------------------------------------------------------
# 7. Diagnostics  (off the sampling path)
# ---------------------------------------------------------------------------

def amplitude_bias_projection(
    log_jacobian: Array,
    basis: Array,
    error_field: Array,
    var_point: Array,
    offset_variance: Scalar,
    prior_variance: Array,
    correlation_rho: Scalar = 0.0,
    units: str = "sigma",
) -> Array:
    """How far a known model error moves the amplitudes, in units of their error.

    THIS is the gate on the linearisation.

    What biases an amplitude is not the MEAN of the error but its
    PROJECTION onto the columns of the design,

        delta_beta = C A^T Sigma^-1 r.

    Usage.  Evaluate the exact nonlinear model and the linearised model on the
    same grid at the same amplitudes, form ``error_field = ln M_exact -
    ln M_linear``, and pass it here.  Every returned value should be small
    compared with one; the amplitude with the largest value is the one the
    linearisation is least able to support.

    The same function answers a second question with a different input: pass a
    fit residual instead of a linearisation error and it reports how far an
    unmodelled systematic could move each amplitude.

    Args:
        error_field:
            ``(n_wavelength, n_time)``, a known additive error in log space.

    **Validity of the single pass.**
    The shift returned is the LEADING term.  Strictly the calculation is
    self-referential: ``error_field`` must be evaluated at the fitted
    amplitudes, but those amplitudes are themselves displaced by the very shift
    being computed, so the exact answer is a fixed point.  Iterating would
    require re-evaluating the exact forward model, which this module cannot do.

    It does not need to, because the correction is second order in the shift:
    re-evaluating at the displaced amplitudes changes the answer by
    ``O(shift^2)``.  The function is therefore self-diagnosing -- a returned
    value well below one means the single pass is accurate, and a value near or
    above one means the linearisation is failing badly enough that this estimate
    is itself unreliable and the quadratic term must be retained.  Read a large
    return as "not trustworthy", never as a correction to apply.

    Returns:
        ``(n_column,)``.  With ``units="sigma"`` (the default) the shift in each
        amplitude divided by its posterior standard deviation; with
        ``units="absolute"`` the shift itself, which is what
        :func:`amplitude_fixed_point` subtracts.

        A returned value far above one is a DIAGNOSIS, not a measurement.  The
        projection is itself linear in the error field, and the error field is
        evaluated at amplitudes that the same bias has displaced; when the
        displacement is large the single pass only gives the leading term.  Iterate
        with :func:`amplitude_fixed_point` before quoting a corrected number.
    """
    normal_matrix, normal_vector, _ = build_linear_system(
        log_jacobian,
        basis,
        error_field,
        var_point,
        offset_variance,
        correlation_rho,
    )
    cholesky = _posterior_cholesky(normal_matrix, prior_variance)
    shift = cho_solve((cholesky, True), normal_vector)
    identity = jnp.eye(normal_matrix.shape[0], dtype=normal_matrix.dtype)
    covariance = cho_solve((cholesky, True), identity)
    if units not in ("sigma", "absolute"):
        raise ValueError(f"units must be 'sigma' or 'absolute', got {units!r}")
    if units == "absolute":
        return shift
    return shift / jnp.sqrt(jnp.diag(covariance))


def curvature_expansion_parameter(
    log_jacobian: Array,
    log_curvature: Array,
    excursions: Array,
) -> Array:
    """The dimensionless number that decides whether linearising is allowed.

    Diagnostic.  Cheap: one Hessian diagonal from the model, no sampler.

    Taylor's theorem says the term we drop carries the square of the
    displacement, so linearising is safe when the displacement is "small".  The
    question is small COMPARED WITH WHAT, and the obvious answers are wrong.
    An excursion divided by its own parameter value is not even dimensionless
    when the parameter is a logarithm, and the largest excursion in the fit is
    not the controlling quantity either -- halving it on this dataset changed
    the error by 12%, not by the factor of four a naive reading predicts.

    The invariant measure is the ratio of the dropped term to the kept one::

        eps_p = | G_pp delta_p^2 / 2 |  /  | J_p delta_p |
              = | delta_p / 2 | * | G_pp / J_p |

    evaluated at the largest excursion each parameter actually reaches.  It is
    the fractional error the linearisation makes in that parameter's own
    response, so ``eps = 0.04`` means the recovered amplitude is wrong by about
    4%.  Compare it with the amplitude's fractional PRECISION: a parameter
    measured to 1% and linearised to 4% is being reported wrongly, whatever the
    formal error bar says.

    Note what this measure does NOT respond to.  It is invariant under any
    LINEAR change of the modulated coordinates, because such a change rescales
    ``delta``, ``J`` and ``G`` in exactly compensating ways -- the same physical
    trajectory is described, so the same term is dropped.

    Args:
        log_jacobian:
            ``(n_wavelength, n_response)``, ``dln M/dtheta_p``.
        log_curvature:
            ``(n_wavelength, n_response)``, the DIAGONAL ``d2ln M/dtheta_p^2``.
        excursions:
            ``(n_time, n_response)`` from :func:`modulation_time_series`, or any
            representative displacement per parameter.

    Returns:
        ``(n_response,)`` values of ``eps``, one per modulated parameter, using the
        wavelength at which that parameter's response is strongest -- the place the
        fit takes most of its information from.
    """
    if log_jacobian.shape != log_curvature.shape:
        raise ValueError(
            f"log_jacobian {log_jacobian.shape} and log_curvature "
            f"{log_curvature.shape} differ"
        )
    if excursions.shape[1] != log_jacobian.shape[1]:
        raise ValueError(
            f"excursions has {excursions.shape[1]} parameters, log_jacobian "
            f"has {log_jacobian.shape[1]}"
        )
    strongest = jnp.argmax(jnp.abs(log_jacobian), axis=0)          # (n_response,)
    index = jnp.arange(log_jacobian.shape[1])
    ratio = jnp.abs(
        log_curvature[strongest, index] / log_jacobian[strongest, index]
    )
    return 0.5 * jnp.max(jnp.abs(excursions), axis=0) * ratio


def curvature_linearising_exponent(
    log_jacobian: Array, log_curvature: Array
) -> tuple[Array, Array]:
    """The change of variable that would straighten each parameter's response.

    Diagnostic.  Follows directly from
    :func:`curvature_expansion_parameter`, and is the only kind of
    reparameterisation that can reduce it.

    Under a change of the modulated variable ``u = g(theta)`` the expansion
    parameter becomes

        eps_u = |delta / 2| * | G/J  -  g''/g' |

    (differentiate ``J`` and ``G`` through the chain rule; the factors of ``g'``
    cancel between the rescaled excursion and the rescaled derivatives).  Two
    things follow.

    First, a LINEAR change of variable has ``g'' = 0`` and leaves ``eps``
    exactly unchanged.  Rewriting a gradient modulation as a temperature
    modulation is such a change, because the Zhang profile builds ``ln T`` by
    cumulative sums of the gradients; it cannot help, however much it alters
    the numerical size of the excursion.

    Second, the curvature is removed exactly when ``g''/g' = G/J``.  If ``G/J``
    is a constant ``c``, that is solved by ``g(theta) = exp(c theta)/c`` -- an
    exponential change of variable.  The physically important case is a Wien-like
    response, ``ln M ~ -a/T``: there ``G/J = -1`` in ``ln T``, so ``c = -1`` and
    ``g ~ 1/T``.  Modulating the INVERSE temperature makes that response exactly
    linear, which is what one would expect, since the emergent flux is
    exponential in ``-1/T`` and the logarithm of it is not.

    The catch is that a single scalar transform can only work if ``c`` is the
    same at every wavelength.  Different wavelengths probe different depths and
    different opacities, so it need not be.  This returns both the exponent and
    the spread, so the question is answered by measurement rather than hope:
    a small spread means the transform will work, a large one means the
    curvature is irreducibly wavelength-dependent and must be carried
    (:func:`curvature_response_columns`) rather than transformed away.

    Args:
        log_jacobian, log_curvature:
            ``(n_wavelength, n_response)``; the curvature is the DIAGONAL of the
            Hessian, as returned with ``diagonal_only=True``.

    Returns:
        ``(exponent, spread)``, each ``(n_response,)``.  ``exponent`` is the
        Jacobian-weighted mean of ``G/J`` -- weighted because wavelengths where the
        response is weak carry no information and would otherwise dominate a plain
        ratio.  ``spread`` is the weighted standard deviation of ``G/J`` divided by
        the mean, so it is dimensionless: below about 0.3 a single transform is
        worth trying, above about 1 it is not.
    """
    if log_jacobian.shape != log_curvature.shape:
        raise ValueError(
            f"log_jacobian {log_jacobian.shape} and log_curvature "
            f"{log_curvature.shape} differ"
        )
    ratio = log_curvature / log_jacobian
    weight = jnp.square(log_jacobian)
    weight = weight / jnp.sum(weight, axis=0, keepdims=True)
    exponent = jnp.sum(weight * ratio, axis=0)
    variance = jnp.sum(weight * jnp.square(ratio - exponent[None, :]), axis=0)
    spread = jnp.sqrt(variance) / jnp.abs(exponent)
    return exponent, spread


def second_order_field(log_curvature_full: Array, excursions: Array) -> Array:
    """The term the linearisation drops, evaluated at a given modulation.

        r_ij = (1/2) sum_pq G_pq(lam_i) delta_p(t_j) delta_q(t_j)

    Diagnostic.  This is the analytic form of the error field that
    :func:`amplitude_bias_projection` gates on, and it is the cheap substitute
    for evaluating the exact nonlinear model.

    Args:
        log_curvature_full:
            ``(n_wavelength, n_response, n_response)``, symmetric in the last two
            axes.
        excursions:
            ``(n_time, n_response)``.

    Returns:
        ``(n_wavelength, n_time)`` additive error in log space.
    """
    return 0.5 * jnp.einsum("ipq,tp,tq->it", log_curvature_full, excursions,
                            excursions)


def curvature_response_columns(
    log_curvature_full: Array, excursions: Array
) -> Array:
    """Response spectra for the curvature, to be marginalised rather than dropped.

    Setup/diagnostic.

    When the expansion parameter of :func:`curvature_expansion_parameter` is too
    large to ignore but not so large that the expansion is worthless, there are
    two ways forward.  One is to subtract the second-order term iteratively
    (:func:`amplitude_fixed_point`), which corrects the amplitudes but leaves
    the posterior covariance and the evidence describing a model that is no
    longer the one being fitted.  The other is to admit that the curvature
    contributes a response we cannot pin down, give it its own amplitudes, and
    integrate them out with the rest.

    This builds the columns for the second route.  Contracting the Hessian with
    the excursion gives, for each parameter ``q``, a wavelength-dependent
    response ``sum_p G_pq delta_p`` -- the direction the true model departs from
    the linear one.  Appending these to the log Jacobian keeps the design
    separable and the whole framework exactly linear-Gaussian; the physical
    amplitudes simply become less certain, by the amount that the curvature
    could have masqueraded as signal.

    That is the statistically honest move: it converts a bias into a variance,
    and unlike a subtraction it leaves the marginal likelihood meaning what it
    says.  The cost is ``n_response`` extra columns and the risk that they are
    degenerate with the original ones -- check with
    :func:`posterior_eigenspectrum` before trusting the result, because for a
    multiplicative response ``G_pp`` often resembles ``J_p`` in shape.

    Args:
        log_curvature_full:
            ``(n_wavelength, n_response, n_response)``.
        excursions:
            ``(n_time, n_response)``; the root-mean-square over time sets the scale
            of each column.

    Returns:
        ``(n_wavelength, n_response)`` extra response spectra, scaled so that their
        amplitudes are of order the physical ones.
    """
    scale = jnp.sqrt(jnp.mean(jnp.square(excursions), axis=0))     # (n_response,)
    return 0.5 * jnp.einsum("ipq,p->iq", log_curvature_full, scale)


def amplitude_fixed_point(
    mean: Array,
    error_field_at: "callable",
    log_jacobian: Array,
    basis: Array,
    var_point: Array,
    offset_variance: Scalar,
    prior_variance: Array,
    correlation_rho: Scalar = 0.0,
    n_iteration: int = 6,
    tolerance: float = 1e-3,
) -> tuple[Array, Array, bool]:
    """Iterate the bias correction to a fixed point.

    Diagnostic, and NOT a foundation -- read the caveats before using the result.

    :func:`amplitude_bias_projection` computes the shift a known model error
    induces, but the error itself depends on the amplitudes, which the shift has
    moved.  One pass gives the leading term.  When the shift is comparable to or
    larger than the uncertainty, the leading term is not the answer, and the
    correction must be iterated::

        beta <- beta_hat - C A^T Sigma^-1 r(beta)

    until it stops moving.  Convergence is not guaranteed: the iteration
    contracts only while the curvature is small enough that the second-order
    term is a correction rather than a competitor, so failure to converge is
    itself the finding that the expansion has broken down.

    **What a converged answer does and does not give you.**
    It corrects the MEAN.  It does not correct the posterior covariance, which
    still comes from the linear design, so the reported error bars belong to a
    model whose centre has just moved.  It does not correct the marginal
    likelihood either, so model comparison -- the reason this framework exists
    -- is not restored by it.  And if the correction is applied at every
    reference-state draw, as consistency with the mixture over ``theta_ref``
    demands, each draw costs a full exact-model evaluation, which is the cost
    the linearisation was adopted to avoid.  The dimensional saving survives;
    the claim that the marginalisation is free does not.

    Use it to measure how large the correction really is.  Prefer
    :func:`curvature_response_columns` if the answer is to be published.

    Args:
        mean:
            ``(n_response, 2 * n_harmonics)`` starting amplitudes.
        error_field_at:
            Callable taking amplitudes shaped like ``mean`` and returning the
            ``(n_wavelength, n_time)`` error field there.  Typically a closure over
            the exact model, or :func:`second_order_field` composed with the basis.
        n_iteration, tolerance:
            Stop when the largest change is below ``tolerance`` standard
            deviations, or after this many passes.

    Returns:
        ``(corrected_mean, shift_history, converged)`` where ``shift_history`` is
        ``(n_step,)`` largest absolute change per iteration in units of the
        posterior standard deviation.
    """
    shape = mean.shape
    # The normal MATRIX depends only on the design, not on the error field, so
    # it is built once; only the right-hand side moves between iterations.
    normal_matrix, _, _ = build_linear_system(
        log_jacobian, basis, jnp.zeros_like(var_point), var_point,
        offset_variance, correlation_rho,
    )
    cholesky = _posterior_cholesky(normal_matrix, prior_variance)
    identity = jnp.eye(normal_matrix.shape[0], dtype=normal_matrix.dtype)
    standard_deviation = jnp.sqrt(
        jnp.diag(cho_solve((cholesky, True), identity))
    )

    current = mean
    history = []
    converged = False
    for _ in range(n_iteration):
        _, normal_vector, _ = build_linear_system(
            log_jacobian, basis, error_field_at(current), var_point,
            offset_variance, correlation_rho,
        )
        shift = cho_solve((cholesky, True), normal_vector)
        updated = mean - shift.reshape(shape)
        # Convergence is a property of the STEP, not of the total correction:
        # a large but stable shift is a converged answer, whereas a shift that
        # keeps moving means the expansion is not controlling its own error.
        step = float(
            jnp.max(jnp.abs((updated - current).reshape(-1) / standard_deviation))
        )
        history.append(step)
        current = updated
        if step < tolerance:
            converged = True
            break
    return current, jnp.asarray(history), converged


def evidence_prior_scan(
    normal_matrix: Array,
    normal_vector: Array,
    chi2_data: Array,
    prior_variance: Array,
    scales: Array,
) -> Array:
    """``ln Z`` as the amplitude prior is widened and narrowed.

    A marginal likelihood is a statement about a model INCLUDING its priors.
    Because the ``-ln|Sigma_beta|/2`` term grows like the number of amplitudes
    times the log of the prior width, a comparison between models with different
    numbers of modulated parameters can be driven entirely by a width that was
    chosen for convenience.  This is a well-known property of Bayes factors under
    vague priors, sometimes called the Bartlett or Lindley paradox, and it is at
    its worst exactly here: the priors are deliberately generous, and the models
    being compared differ in how many amplitudes they carry.

    Run this for every variant being compared and plot ``ln Z`` against the
    scale.  A conclusion is only reportable if the ORDERING of the variants is
    the same across the range of widths that are physically defensible.  Where it
    is not, the data do not settle the question and
    :func:`predictive_log_density` should be used instead.

    Args:
        scales:
            ``(n_scale,)`` multipliers applied to the prior standard deviations, so
            ``prior_variance`` is multiplied by ``scale**2``.

    Returns:
        ``(n_scale,)`` values of ``ln Z``, excluding the data normalisation, which is
        the same at every scale.
    """

    def one(scale):
        evidence, _ = modulation_log_evidence(
            normal_matrix,
            normal_vector,
            chi2_data,
            prior_variance * jnp.square(scale),
        )
        return evidence

    return jnp.stack([one(scale) for scale in scales])


def posterior_eigenspectrum(
    normal_matrix: Array, prior_variance: Array
) -> tuple[Array, Array]:
    """Directions in amplitude space the data determine, best first.

    An individual amplitude is only meaningful if the design can tell it apart
    from the others.  Where two response spectra look alike -- a deep temperature
    change and a temperature-gradient change, for instance -- the data constrain
    a combination tightly and each amplitude separately hardly at all, and
    quoting per-amplitude error bars then overstates what was measured.

    This diagonalises the posterior precision.  Each eigenvalue is the precision
    (one over variance) along its eigenvector, so the ratio of the largest to the
    smallest is the condition number of the inference: how many times better the
    best-determined direction is measured than the worst.  Report results in the
    leading directions, and treat any direction whose precision is close to the
    prior's own, ``1/prior_variance``, as unmeasured.

    Returns:
        ``(precisions, directions)`` with ``precisions`` ``(n_column,)`` descending
        and ``directions`` ``(n_column, n_column)`` whose COLUMNS are the
        corresponding unit vectors in amplitude space.
    """
    precision = normal_matrix + jnp.diag(1.0 / prior_variance)
    values, vectors = jnp.linalg.eigh(precision)
    order = jnp.argsort(values)[::-1]
    return values[order], vectors[:, order]


def predictive_log_density(
    normal_matrix_train: Array,
    prior_variance: Array,
    normal_matrix_test: Array,
    normal_vector_test: Array,
    chi2_test: Array,
    noise_log_determinant_test: Scalar,
    n_test: int,
) -> Array:
    """Log probability of held-out data under the fit, prior width and all.

    A prior-insensitive alternative to comparing marginal likelihoods.  Fit on
    part of the data, then ask how well the fitted model predicts the part that
    was withheld.  A model that wins only because its priors are wide will not
    predict better, so this answers the model-comparison question without
    inheriting the sensitivity that :func:`evidence_prior_scan` exposes.

    The predictive distribution of the held-out data is Gaussian with covariance
    ``Sigma_test + A_test C A_test^T``: the measurement noise, plus the
    uncertainty in the amplitudes propagated through the design.  Both terms are
    handled without forming any large matrix, using two standard identities.

    The Woodbury identity generalises Sherman-Morrison from a rank-one update to
    a rank-M one:

        (S + A C A^T)^-1 = S^-1 - S^-1 A (C^-1 + A^T S^-1 A)^-1 A^T S^-1

    and the matching determinant identity is

        |S + A C A^T| = |S| |C| |C^-1 + A^T S^-1 A|.

    Both right-hand sides need only ``A^T S^-1 A`` and ``A^T S^-1 r`` on the
    held-out block -- which is exactly what :func:`build_linear_system` returns
    when it is handed the test epochs and the residual about the fitted model.

    Usage.  Split the epochs into folds.  For each fold: build the system on the
    training epochs, solve for the mean, form the residual of the held-out
    epochs about the predicted modulation, build a second system on those epochs
    from that residual, and call this.  Sum over folds.

    Args:
        normal_matrix_train:
            ``A^T Sigma^-1 A`` on the TRAINING epochs.
        normal_matrix_test, normal_vector_test, chi2_test:
            ``build_linear_system`` output on the TEST epochs, built from the
            residual of the test data about the training posterior mean.
        noise_log_determinant_test:
            ``ln|Sigma_test|`` from :func:`noise_log_determinant`.
        n_test:
            Number of constrained points in the test block.

    Returns:
        Scalar log predictive density.
    """
    train_cholesky = _posterior_cholesky(normal_matrix_train, prior_variance)
    total_cholesky = jnp.linalg.cholesky(
        normal_matrix_train + normal_matrix_test + jnp.diag(1.0 / prior_variance)
    )
    solved = cho_solve((total_cholesky, True), normal_vector_test)
    quadratic = chi2_test - normal_vector_test @ solved
    log_determinant = (
        noise_log_determinant_test
        + 2.0 * jnp.sum(jnp.log(jnp.diag(total_cholesky)))
        - 2.0 * jnp.sum(jnp.log(jnp.diag(train_cholesky)))
    )
    return -0.5 * (quadratic + log_determinant + n_test * jnp.log(2.0 * jnp.pi))


# ---------------------------------------------------------------------------
# 8. Reporting
# ---------------------------------------------------------------------------

def modulation_time_series(
    mean: Array, covariance: Array, basis: Array
) -> tuple[Array, Array]:
    """Parameter excursion about the reference state, with its 1-sigma band.

        delta_p(t) = sum_m beta_pm Phi_tm

    Reporting only; not on the sampling path.

    Returns ``(delta, delta_sd)``, both ``(n_time, n_response)``.  The band is
    the CONDITIONAL uncertainty at a fixed reference state; the full uncertainty
    also carries the scatter of ``mean`` across reference-state samples, which
    :func:`mixture_moments` supplies.
    """
    n_response, n_mode = mean.shape
    delta = jnp.einsum("pm,tm->tp", mean, basis)
    blocks = covariance.reshape(n_response, n_mode, n_response, n_mode)
    diagonal_blocks = jnp.einsum("pmpn->pmn", blocks)
    variance = jnp.einsum("tm,pmn,tn->tp", basis, diagonal_blocks, basis)
    return delta, jnp.sqrt(jnp.clip(variance, 0.0, None))


def mixture_moments(means: Array, covariances: Array) -> tuple[Array, Array, Array]:
    """Combine per-sample amplitude posteriors into one, splitting the variance.

    The closed-form posterior of :func:`solve_modulation` is CONDITIONAL on the
    reference state.  The unconditional posterior is the mixture over the
    reference-state samples,

        p(beta | data) = (1/S) sum_s N(beta; beta_s, C_s),

    which is a Rao-Blackwellised estimate: each draw contributes an exact
    Gaussian rather than a single point, so it is less noisy than sampling the
    amplitudes would be.

    The law of total variance splits its covariance into two pieces that mean
    different things:

        Cov(beta) = <C_s>_s          (statistical: noise in the data)
                  + Cov_s(beta_s)    (systematic: not knowing theta_ref)

    Keeping them separate matters because the second has been measured to
    dominate.  Reporting only the first understates the uncertainty badly; the
    conditional posteriors must therefore be computed AT EVERY DRAW, since
    fixing the reference state and solving once afterwards discards the second
    term entirely.

    Args:
        means:
            ``(n_sample, n_column)``.
        covariances:
            ``(n_sample, n_column, n_column)``.

    Returns:
        ``(mean, total_covariance, systematic_covariance)``.
    """
    if means.shape[0] != covariances.shape[0]:
        raise ValueError(
            f"{means.shape[0]} means but {covariances.shape[0]} covariances"
        )
    mean = jnp.mean(means, axis=0)
    centred = means - mean[None, :]
    systematic = (centred.T @ centred) / means.shape[0]
    statistical = jnp.mean(covariances, axis=0)
    return mean, statistical + systematic, systematic
