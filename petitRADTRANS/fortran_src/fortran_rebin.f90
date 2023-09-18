! """
! Utility function to rebin synthetic spectrum to data spectrum for the petitRADTRANS radiative transfer package.
! """

module fortran_rebin
    implicit none

    contains
        subroutine rebin_spectrum(input_wavelengths, input_spectrum, rebinned_wavelengths, &
                                  rebinned_spectrum, n_rebin, n_wavelengths)
            implicit none

            integer, intent(in) :: n_rebin, n_wavelengths
            double precision, intent(in) :: rebinned_wavelengths(n_rebin)
            double precision, intent(in) :: input_wavelengths(n_wavelengths), input_spectrum(n_wavelengths)
            double precision, intent(out) :: rebinned_spectrum(n_rebin)

            double precision :: nu_obs_bin_bords(n_rebin + 1)
            integer :: i_nu, i_nu_synth
            double precision :: nu1, nu2, y_ax, del_nu, ddel_nu, slope, add_fl, min_flux, max_flux

            rebinned_spectrum = 0d0

            ! Check stuff...
            if (n_rebin < 2) then
                write(*, *) "rebin.f90 Error: output wavelength array size must be >= 2, but is of size ", &
                    n_rebin

                rebinned_spectrum = -1d0

                return
            end if

            ! Get intp bin bords
            nu_obs_bin_bords(1) = rebinned_wavelengths(1) - (rebinned_wavelengths(2) - rebinned_wavelengths(1)) / 2d0

            do i_nu = 2, n_rebin
                nu_obs_bin_bords(i_nu) = rebinned_wavelengths(i_nu) &
                    - (rebinned_wavelengths(i_nu) - rebinned_wavelengths(i_nu - 1)) / 2d0
            end do

            nu_obs_bin_bords(n_rebin + 1) = rebinned_wavelengths(n_rebin) &
                + (rebinned_wavelengths(n_rebin) - rebinned_wavelengths(n_rebin-1)) / 2d0

            if ((input_wavelengths(1) >= nu_obs_bin_bords(1)) &
                    .or. (input_wavelengths(n_wavelengths) <= nu_obs_bin_bords(n_rebin + 1))) then
                write(*, *) "rebin.f90 Error: input wavelength array needs to extend at least half" // &
                     " a bin width further than output wavelength array on both sides:"
                write(*, *) "         required interval: ", nu_obs_bin_bords(1), "--", nu_obs_bin_bords(n_rebin + 1)
                write(*, *) " input_wavelength interval: ", input_wavelengths(1), "--", input_wavelengths(n_wavelengths)

                rebinned_spectrum = -1d0

                return
            end if

            ! Start interpolation
            i_nu_synth = 1

            do i_nu = 1, n_rebin
                do while (input_wavelengths(i_nu_synth) < nu_obs_bin_bords(i_nu))
                    i_nu_synth = i_nu_synth + 1
                end do

                del_nu = 0d0

                do while (input_wavelengths(i_nu_synth) < nu_obs_bin_bords(i_nu + 1))
                    nu1 = max(input_wavelengths(i_nu_synth - 1), nu_obs_bin_bords(i_nu))
                    nu2 = min(input_wavelengths(i_nu_synth), nu_obs_bin_bords(i_nu + 1))
                    ddel_nu = nu2 - nu1

                    slope = (input_spectrum(i_nu_synth) - input_spectrum(i_nu_synth - 1)) &
                            / (input_wavelengths(i_nu_synth) - input_wavelengths(i_nu_synth - 1))

                    y_ax = input_spectrum(i_nu_synth - 1)

                    min_flux = min(input_spectrum(i_nu_synth), input_spectrum(i_nu_synth - 1))
                    max_flux = max(input_spectrum(i_nu_synth), input_spectrum(i_nu_synth - 1))

                    ! y+b*(x-a) , int from x1 .. x2
                    ! y*(x2-x1)-b*a*(x2-x1)+b/2d0*(x2**2d0-x1**2d0)
                    add_fl = (y_ax - slope * input_wavelengths(i_nu_synth - 1)) * (nu2 - nu1) &
                             + slope * (nu2 - nu1) * (nu2 + nu1) / 2d0

                    rebinned_spectrum(i_nu) = rebinned_spectrum(i_nu) + add_fl

                    del_nu = del_nu + ddel_nu

                    i_nu_synth = i_nu_synth + 1
                end do

                nu1 = max(input_wavelengths(i_nu_synth - 1), nu_obs_bin_bords(i_nu))
                nu2 = min(input_wavelengths(i_nu_synth), nu_obs_bin_bords(i_nu + 1))
                ddel_nu = nu2 - nu1

                slope = (input_spectrum(i_nu_synth) - input_spectrum(i_nu_synth - 1)) &
                        / (input_wavelengths(i_nu_synth) - input_wavelengths(i_nu_synth - 1))

                y_ax = input_spectrum(i_nu_synth - 1)

                min_flux = min(input_spectrum(i_nu_synth), input_spectrum(i_nu_synth - 1))
                max_flux = max(input_spectrum(i_nu_synth), input_spectrum(i_nu_synth - 1))

                ! y+b*(x-a) , int from x1 .. x2
                ! y*(x2-x1)-b*a*(x2-x1)+b/2d0*(x2**2d0-x1**2d0)
                add_fl = (y_ax - slope * input_wavelengths(i_nu_synth - 1)) * (nu2 - nu1) &
                         + slope * (nu2 ** 2d0 - nu1 ** 2d0) / 2d0

                rebinned_spectrum(i_nu) = rebinned_spectrum(i_nu) + add_fl

                del_nu = del_nu + ddel_nu

                rebinned_spectrum(i_nu) = rebinned_spectrum(i_nu) / del_nu
            end do
        end subroutine rebin_spectrum
end module fortran_rebin
