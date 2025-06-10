import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax, jit
from petitRADTRANS.fortran_rebin import fortran_rebin

def rebinning_interpolation(input_wavelengths, input_spectrum, rebin_bin_low, rebin_bin_high, n_rebin):
    rebinned_spectrum = jnp.zeros(n_rebin)
    i_input = 0

    def rebin_step(i, carry):
        i_input, rebinned_spectrum = carry
        
        def while_input_count_initialisation(carry):
            i_input = carry
            return i_input + 1
        
        def while_cond_initialisation(carry):
            i_input = carry
            return input_wavelengths[i_input] < rebin_bin_low[i]
        
        def while_cond(carry):
            i_input, _, _ = carry
            return input_wavelengths[i_input] < rebin_bin_high[i]

        def while_body(carry):
            i_input, rebinned_spectrum, del_nu_in = carry
            nu1 = jnp.maximum(input_wavelengths[i_input - 1], rebin_bin_low[i])
            nu2 = jnp.minimum(input_wavelengths[i_input], rebin_bin_high[i])
            ddel_nu = nu2 - nu1

            slope = (input_spectrum[i_input] - input_spectrum[i_input - 1]) / (input_wavelengths[i_input] - input_wavelengths[i_input - 1])
            y_ax = input_spectrum[i_input - 1]

            add_fl = (y_ax - slope * input_wavelengths[i_input - 1]) * (nu2 - nu1) + slope * (nu2 - nu1) * (nu2 + nu1) / 2.0
            rebinned_spectrum = rebinned_spectrum.at[i].add(add_fl)

            del_nu_out = del_nu_in + ddel_nu
            i_input = i_input + 1
            return i_input, rebinned_spectrum, del_nu_out
        
        del_nu = 0.0
        i_input = lax.while_loop(while_cond_initialisation, while_input_count_initialisation,(i_input))
        i_input, rebinned_spectrum, del_nu = lax.while_loop(while_cond, while_body, (i_input, rebinned_spectrum, del_nu))

        nu1 = jnp.maximum(input_wavelengths[i_input - 1], rebin_bin_low[i])
        nu2 = jnp.minimum(input_wavelengths[i_input], rebin_bin_high[i])
        ddel_nu = nu2 - nu1

        slope = (input_spectrum[i_input] - input_spectrum[i_input - 1]) / (input_wavelengths[i_input] - input_wavelengths[i_input - 1])
        y_ax = input_spectrum[i_input - 1]

        add_fl = (y_ax - slope * input_wavelengths[i_input - 1]) * (nu2 - nu1) + slope * (nu2 ** 2.0 - nu1 ** 2.0) / 2.0
        rebinned_spectrum = rebinned_spectrum.at[i].add(add_fl)
        
        del_nu = del_nu + ddel_nu
        rebinned_spectrum = rebinned_spectrum.at[i].set(rebinned_spectrum[i] / del_nu)

        return i_input, rebinned_spectrum

    _, rebinned_spectrum = lax.fori_loop(0, n_rebin, rebin_step, (i_input, rebinned_spectrum))
    
    return rebinned_spectrum

@jit
def rebin_spectrum(input_wavelengths, input_spectrum, rebinned_wavelengths):
    n_rebin = rebinned_wavelengths.size
    rebinned_spectrum = jnp.zeros(n_rebin)

    if n_rebin < 2:
        raise ValueError(f"Error: output wavelength array size must be >= 2, but is of size {n_rebin}")

    rebin_bin_low = jnp.zeros(n_rebin)
    rebin_bin_high = jnp.zeros(n_rebin)

    rebin_bin_low = rebin_bin_low.at[0].set(rebinned_wavelengths[0] - (rebinned_wavelengths[1] - rebinned_wavelengths[0]) / 2.0)
    rebin_bin_low = rebin_bin_low.at[1:].set(rebinned_wavelengths[1:] - (rebinned_wavelengths[1:] - rebinned_wavelengths[:-1]) / 2.0)

    rebin_bin_high = rebin_bin_high.at[:-1].set(rebin_bin_low[1:])
    rebin_bin_high = rebin_bin_high.at[-1].set(rebinned_wavelengths[-1] + (rebinned_wavelengths[-1] - rebinned_wavelengths[-2]) / 2.0)

    #if (input_wavelengths[0] >= rebin_bin_low[0]) or (input_wavelengths[-1] <= rebin_bin_high[-1]):
    #    raise ValueError(f"Error: input wavelength array needs to extend at least half a bin width further than output wavelength array on both sides:\n"
    #                     f"input interval: {input_wavelengths[0]} -- {input_wavelengths[-1]}\n"
    #                     f"required interval: {rebin_bin_low[0]} -- {rebin_bin_high[-1]}")
    #
    rebinned_spectrum = rebinning_interpolation(input_wavelengths, input_spectrum, rebin_bin_low, rebin_bin_high, n_rebin)
    return rebinned_spectrum

@jit
def rebin_spectrum_bin(input_wavelengths, input_spectrum, rebinned_wavelengths, bin_widths):
    n_rebin = rebinned_wavelengths.size
    rebinned_spectrum = jnp.zeros(n_rebin)

    if n_rebin < 2:
        raise ValueError(f"Error: output wavelength array size must be >= 2, but is of size {n_rebin}")

    rebin_bin_low = rebinned_wavelengths - (bin_widths / 2.0)
    rebin_bin_high = rebinned_wavelengths + (bin_widths / 2.0)

    #if (input_wavelengths[0] >= rebin_bin_low[0]) or (input_wavelengths[-1] <= rebin_bin_high[-1]):
    #    raise ValueError(f"Error: input wavelength array needs to extend at least half a bin width further than output wavelength array on both sides:\n"
    #                     f"input interval: {input_wavelengths[0]} -- {input_wavelengths[-1]}\n"
    #                     f"required interval: {rebin_bin_low[0]} -- {rebin_bin_high[-1]}")

    rebinned_spectrum = rebinning_interpolation(input_wavelengths, input_spectrum, rebin_bin_low, rebin_bin_high, n_rebin)
    return rebinned_spectrum

"""
# Example usage:
input_wavelengths = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
input_spectrum = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
rebinned_wavelengths = jnp.array([2.5, 3.5, 5.5, 6.5])
bin_widths = jnp.array([1.0, 0.2, 0.6, 1.2])

rebinned_spectrum = rebin_spectrum(input_wavelengths, input_spectrum, rebinned_wavelengths)
print(rebinned_spectrum)

rebinned_spectrum_bin = rebin_spectrum_bin(input_wavelengths, input_spectrum, rebinned_wavelengths, bin_widths)
print(rebinned_spectrum_bin)

from time import time
start = time()
rebinned_spectrum = fortran_rebin.rebin_spectrum(input_wavelengths, input_spectrum, rebinned_wavelengths)
#print(rebinned_spectrum)

end = time()
ft = end-start
print(f"Fortran rebin: {end-start} s")
rebinned_spectrum_bin = fortran_rebin.rebin_spectrum_bin(input_wavelengths, input_spectrum, rebinned_wavelengths, bin_widths)
print(rebinned_spectrum_bin)

start = time()
rebinned_spectrum = rebin_spectrum(input_wavelengths, input_spectrum, rebinned_wavelengths)
end = time()
print(f"JAX rebin: {end-start} s")
print(f"Ratio = {ft/(end-start)}")
rebinned_spectrum_bin = rebin_spectrum_bin(input_wavelengths, input_spectrum, rebinned_wavelengths, bin_widths)

"""