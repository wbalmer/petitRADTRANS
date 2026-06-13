import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_float32_scattering_flux_stays_finite():
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

        import warnings
        warnings.filterwarnings("error", message="overflow encountered.*")

        import jax.numpy as jnp
        from petitRADTRANS.radtrans import Radtrans

        pressures = jnp.logspace(-6, 2, 27, dtype=jnp.float32)
        atmosphere = Radtrans(
            pressures=pressures,
            line_species=("H2O__POKAZATEL",),
            rayleigh_species=("H2", "He"),
            gas_continuum_contributors=("H2--H2", "H2--He"),
            wavelength_boundaries=(1.0, 1.1),
            line_opacity_mode="c-k",
            scattering_in_emission=True,
        )
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

        print(
            "finite",
            bool(jnp.all(jnp.isfinite(flux))),
            float(jnp.nanmin(flux)),
            float(jnp.nanmax(flux)),
        )
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
    assert "finite True" in result.stdout
