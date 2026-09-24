import os
from pathlib import Path
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np

from .context import petitRADTRANS  # noqa: F401
from .utils import test_parameters


def test_float32_scattering_in_emission_is_rejected():
    """Emission scattering must refuse to run in single precision.

    The Feautrier scattering solver is an iterative source-function scheme that
    *diverges* in float32 rather than losing a few digits: measured against the
    identical float64 calculation it returns a median flux error of ~5%, single
    bins wrong by >1000x, and can overflow to +inf. A wrong spectrum is worse
    than no spectrum, so Radtrans refuses the combination up front instead of
    returning it.

    This previously asserted only that the flux stayed finite, which the diverged
    float32 result satisfied for years while being orders of magnitude wrong.
    """
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()

    # Exercise the *installed* petitRADTRANS (with its compiled native
    # extensions such as prt_ck_mixing_ffi), mirroring tests/context.py. Under
    # tox the package is installed into site-packages and the native extension
    # is NOT built into the source tree, so the source tree must not shadow the
    # install. For `python -c`, sys.path[0] is "" (the CWD, set to repo_root
    # below), which would otherwise import the un-built source package; strip
    # every path entry that resolves to the repo root and append it only as a
    # last-resort fallback for pure source checkouts.
    bootstrap = (
        "import sys, os\n"
        f"_repo_root = {repr(str(repo_root))}\n"
        "sys.path = [p for p in sys.path "
        "if os.path.abspath(p or os.getcwd()) != _repo_root]\n"
        "sys.path.append(_repo_root)\n"
    )

    code = bootstrap + textwrap.dedent(
        """
        from jax import config
        config.update("jax_enable_x64", False)

        import jax.numpy as jnp
        from petitRADTRANS.radtrans import Radtrans

        pressures = jnp.logspace(-6, 2, 27, dtype=jnp.float32)
        kwargs = dict(
            pressures=pressures,
            line_species=("H2O__POKAZATEL",),
            rayleigh_species=("H2", "He"),
            gas_continuum_contributors=("H2--H2", "H2--He"),
            wavelength_boundaries=(1.0, 1.1),
            line_opacity_mode="c-k",
        )

        try:
            Radtrans(scattering_in_emission=True, **kwargs)
        except ValueError as error:
            print("rejected", "64-bit" in str(error))
        else:
            print("not rejected")

        # Emission without scattering agrees with float64 to ~1e-6, so it stays
        # supported in single precision and must keep working.
        atmosphere = Radtrans(scattering_in_emission=False, **kwargs)
        temperatures = jnp.full_like(pressures, 1200.0)
        mass_fractions = {
            "H2O__POKAZATEL": jnp.full_like(pressures, 1e-3),
            "H2": jnp.full_like(pressures, 0.74),
            "He": jnp.full_like(pressures, 0.24),
        }
        mean_molar_masses = jnp.full_like(pressures, 2.33)

        _, flux, _ = atmosphere.calculate_flux(
            temperatures=temperatures,
            mass_fractions=mass_fractions,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=jnp.asarray(10 ** 2.45, dtype=jnp.float32),
        )

        print("no_scattering_finite", bool(jnp.all(jnp.isfinite(flux))))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"float32 scattering regression subprocess failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "rejected True" in result.stdout, result.stdout
    assert "no_scattering_finite True" in result.stdout, result.stdout


def test_float32_scattering_in_emission_is_rejected_by_the_setter():
    """The combination must not be reachable by flipping the flag afterwards."""
    repo_root = Path(__file__).resolve().parents[1]

    bootstrap = (
        "import sys, os\n"
        f"_repo_root = {repr(str(repo_root))}\n"
        "sys.path = [p for p in sys.path "
        "if os.path.abspath(p or os.getcwd()) != _repo_root]\n"
        "sys.path.append(_repo_root)\n"
    )

    code = bootstrap + textwrap.dedent(
        """
        from jax import config
        config.update("jax_enable_x64", False)

        import warnings
        import jax.numpy as jnp
        from petitRADTRANS.radtrans import Radtrans

        atmosphere = Radtrans(
            pressures=jnp.logspace(-6, 2, 27, dtype=jnp.float32),
            gas_continuum_contributors=("H2--H2",),
            wavelength_boundaries=(1.0, 1.1),
            line_opacity_mode="c-k",
            scattering_in_emission=False,
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                atmosphere.scattering_in_emission = True
        except ValueError:
            print("setter_rejected True")
        else:
            print("setter_rejected False")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"float32 scattering setter subprocess failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "setter_rejected True" in result.stdout, result.stdout


def _cia_opacities(dtype):
    """Interpolate the test CIA tables onto a small grid in a single dtype."""
    from petitRADTRANS.opacities.cia import compute_cia_opacities
    from petitRADTRANS.radtrans import Radtrans

    atmosphere = Radtrans(
        pressures=test_parameters["pressures"],
        gas_continuum_contributors=test_parameters["spectrum_parameters"][
            "continuum_opacities"
        ],
        wavelength_boundaries=test_parameters["spectrum_parameters"][
            "wavelength_range_correlated_k"
        ],
        line_opacity_mode="c-k",
        scattering_in_emission=False,
    )
    pressures = jnp.asarray(atmosphere._resolve_solver_pressures(None), dtype=dtype)
    ones = jnp.ones_like(pressures)

    return lambda: compute_cia_opacities(
        cia_dicts=atmosphere._cias_loaded_opacities,
        mass_fractions={"H2": 0.74 * ones, "He": 0.24 * ones},
        pressures=pressures,
        temperatures=1200.0 * ones,
        frequencies=atmosphere._frequencies,
        mean_molar_masses=2.33 * ones,
    )


def test_cia_opacities_do_not_depend_on_compilation():
    """The CIA number density factor must not be subnormal in float32.

    ``(1 / L0) ** 2`` is 1.4e-39, below the smallest normal float32 (1.2e-38).
    XLA's op-by-op kernels treat subnormal inputs as zero, so storing it as a
    standalone float32 constant zeroed every CIA opacity outside ``jit`` (and
    everywhere, compiled or not, without ``jax_enable_x64``).
    """
    compute = _cia_opacities(
        dtype=jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    )

    compiled = np.asarray(jax.jit(compute)())

    with jax.disable_jit():
        uncompiled = np.asarray(compute())

    assert np.count_nonzero(compiled) == compiled.size
    np.testing.assert_allclose(uncompiled, compiled, rtol=1e-5)


def test_cia_opacities_are_nonzero_without_x64():
    """Without ``jax_enable_x64`` the CIA factor stays in float32 end to end.

    A subnormal float32 constant is flushed to zero by XLA's float32 kernels in
    that configuration whether or not the spectral function is compiled, so this
    runs in a subprocess where x64 is off from the first import.
    """
    repo_root = Path(__file__).resolve().parents[1]

    # See test_float32_scattering_flux_stays_finite for why sys.path is rebuilt.
    bootstrap = (
        "import sys, os\n"
        f"_repo_root = {repr(str(repo_root))}\n"
        "sys.path = [p for p in sys.path "
        "if os.path.abspath(p or os.getcwd()) != _repo_root]\n"
        "sys.path.append(_repo_root)\n"
    )

    code = bootstrap + textwrap.dedent(
        """
        from jax import config
        config.update("jax_enable_x64", False)

        import jax
        import jax.numpy as jnp
        import numpy as np

        from petitRADTRANS.opacities.cia import compute_cia_opacities
        from petitRADTRANS.radtrans import Radtrans

        atmosphere = Radtrans(
            pressures=jnp.logspace(-6, 2, 27, dtype=jnp.float32),
            gas_continuum_contributors=("H2--H2", "H2--He"),
            wavelength_boundaries=(1.0, 1.1),
            line_opacity_mode="c-k",
            scattering_in_emission=False,
        )
        pressures = jnp.asarray(
            atmosphere._resolve_solver_pressures(None), dtype=jnp.float32
        )
        ones = jnp.ones_like(pressures)

        def compute():
            return compute_cia_opacities(
                cia_dicts=atmosphere._cias_loaded_opacities,
                mass_fractions={"H2": 0.74 * ones, "He": 0.24 * ones},
                pressures=pressures,
                temperatures=1200.0 * ones,
                frequencies=atmosphere._frequencies,
                mean_molar_masses=2.33 * ones,
            )

        compiled = np.asarray(jax.jit(compute)())

        with jax.disable_jit():
            uncompiled = np.asarray(compute())

        print("dtype", compiled.dtype)
        print("all_compiled_nonzero", np.count_nonzero(compiled) == compiled.size)
        print(
            "all_uncompiled_nonzero",
            np.count_nonzero(uncompiled) == uncompiled.size,
        )
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"float32 CIA regression subprocess failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "dtype float32" in result.stdout
    assert "all_compiled_nonzero True" in result.stdout
    assert "all_uncompiled_nonzero True" in result.stdout
