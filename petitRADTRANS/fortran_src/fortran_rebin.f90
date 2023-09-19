! """
! Utility function to rebin synthetic spectrum to data spectrum for the petitRADTRANS radiative transfer package.
! """


module rebin_utils
    ! """
    ! Useful functions for rebinning.
    ! """
    implicit none

    contains
        subroutine rebinning_interpolation(input_wavelengths, input_spectrum, rebin_bin_low, rebin_bin_high, &
                                           n_rebin, n_wavelengths, rebinned_spectrum)
            implicit none

            integer, intent(in) :: n_rebin, n_wavelengths
            double precision, intent(in) :: input_wavelengths(n_wavelengths), input_spectrum(n_wavelengths)
            double precision :: rebin_bin_low(n_rebin), rebin_bin_high(n_rebin)
            double precision, intent(out) :: rebinned_spectrum(n_rebin)

            integer :: i, i_input
            double precision :: nu1, nu2, y_ax, del_nu, ddel_nu, slope, add_fl, min_flux, max_flux

            i_input = 1

            do i = 1, n_rebin
                do while (input_wavelengths(i_input) < rebin_bin_low(i))
                    i_input = i_input + 1
                end do

                del_nu = 0d0

                do while (input_wavelengths(i_input) < rebin_bin_high(i))
                    nu1 = max(input_wavelengths(i_input - 1), rebin_bin_low(i))
                    nu2 = min(input_wavelengths(i_input), rebin_bin_high(i))

                    ddel_nu = nu2 - nu1

                    slope = (input_spectrum(i_input) - input_spectrum(i_input - 1)) &
                            / (input_wavelengths(i_input) - input_wavelengths(i_input - 1))

                    y_ax = input_spectrum(i_input - 1)

                    min_flux = min(input_spectrum(i_input), input_spectrum(i_input - 1))
                    max_flux = max(input_spectrum(i_input), input_spectrum(i_input - 1))

                    ! y+b*(x-a) , int from x1 .. x2
                    ! y*(x2-x1)-b*a*(x2-x1)+b/2d0*(x2**2d0-x1**2d0)
                    add_fl = (y_ax - slope * input_wavelengths(i_input - 1)) * (nu2 - nu1) &
                        + slope * (nu2 - nu1) * (nu2 + nu1) / 2d0

                    rebinned_spectrum(i) = rebinned_spectrum(i) + add_fl

                    del_nu = del_nu + ddel_nu
                    i_input = i_input + 1
                end do

                nu1 = max(input_wavelengths(i_input - 1), rebin_bin_low(i))
                nu2 = min(input_wavelengths(i_input), rebin_bin_high(i))
                ddel_nu = nu2 - nu1

                slope = (input_spectrum(i_input) - input_spectrum(i_input - 1)) &
                        / (input_wavelengths(i_input) - input_wavelengths(i_input - 1))

                y_ax = input_spectrum(i_input - 1)

                min_flux = min(input_spectrum(i_input), input_spectrum(i_input - 1))
                max_flux = max(input_spectrum(i_input), input_spectrum(i_input - 1))

                ! y+b*(x-a) , int from x1 .. x2
                ! y*(x2-x1)-b*a*(x2-x1)+b/2d0*(x2**2d0-x1**2d0)
                add_fl = (y_ax - slope * input_wavelengths(i_input - 1)) * (nu2 - nu1) &
                         + slope * (nu2 ** 2d0 - nu1 ** 2d0) / 2d0

                rebinned_spectrum(i) = rebinned_spectrum(i) + add_fl

                del_nu = del_nu + ddel_nu

                rebinned_spectrum(i) = rebinned_spectrum(i) / del_nu
            end do
        end subroutine rebinning_interpolation


        subroutine write_input_size_error(n_rebin, rebinned_spectrum)
            implicit none

            integer, intent(in) :: n_rebin
            double precision, intent(out) :: rebinned_spectrum(n_rebin)

            write(*, *) "rebin.f90 Error: output wavelength array size must be >= 2, but is of size ", &
                n_rebin

            rebinned_spectrum = -1d0
        end subroutine write_input_size_error


        subroutine write_out_of_bounds_error(input_wavelengths_min, input_wavelengths_max, &
                                             rebin_bin_min, rebin_bin_max, n_rebin, &
                                             rebinned_spectrum)
            implicit none

            integer, intent(in) :: n_rebin
            double precision, intent(in) :: input_wavelengths_min
            double precision, intent(in) :: input_wavelengths_max
            double precision, intent(in) :: rebin_bin_min
            double precision, intent(in) :: rebin_bin_max
            double precision, intent(out) :: rebinned_spectrum(n_rebin)

            write(*, *) "rebin.f90 Error: input wavelength array needs to extend at least half" // &
                 " a bin width further than output wavelength array on both sides:"
            write(*, *) "    input interval: ", input_wavelengths_min, "--", input_wavelengths_max
            write(*, *) " required interval: ", rebin_bin_min, "--", rebin_bin_max

            rebinned_spectrum = -1d0
        end subroutine write_out_of_bounds_error
end module rebin_utils


module fortran_rebin
    implicit none

    contains
        subroutine rebin_spectrum(input_wavelengths, input_spectrum, rebinned_wavelengths, &
                                  n_rebin, n_wavelengths, rebinned_spectrum)
            use rebin_utils, only: rebinning_interpolation, write_input_size_error, write_out_of_bounds_error

            implicit none

            integer, intent(in) :: n_rebin, n_wavelengths
            double precision, intent(in) :: rebinned_wavelengths(n_rebin)
            double precision, intent(in) :: input_wavelengths(n_wavelengths), input_spectrum(n_wavelengths)
            double precision, intent(out) :: rebinned_spectrum(n_rebin)

            double precision :: rebin_bin_low(n_rebin), rebin_bin_high(n_rebin)

            rebinned_spectrum = 0d0

            ! Check output array size
            if (n_rebin < 2) then
                call write_input_size_error(n_rebin, rebinned_spectrum)

                return
            end if

            ! Get bin boundaries
            rebin_bin_low(1) = rebinned_wavelengths(1) - (rebinned_wavelengths(2) - rebinned_wavelengths(1)) / 2d0
            rebin_bin_low(2:n_rebin) = rebinned_wavelengths(2:n_rebin) &
                - (rebinned_wavelengths(2:n_rebin) - rebinned_wavelengths(1:n_rebin-1)) / 2d0

            rebin_bin_high(1:n_rebin-1) = rebin_bin_low(2:n_rebin)
            rebin_bin_high(n_rebin) = rebinned_wavelengths(n_rebin) &
                + (rebinned_wavelengths(n_rebin) - rebinned_wavelengths(n_rebin-1)) / 2d0

            if ((input_wavelengths(1) >= rebin_bin_low(1)) &
                    .or. (input_wavelengths(n_wavelengths) <= rebin_bin_high(n_rebin))) then
                call write_out_of_bounds_error(&
                    input_wavelengths(1), input_wavelengths(n_wavelengths), &
                    rebin_bin_low(1), rebin_bin_high(n_wavelengths), n_rebin, &
                    rebinned_spectrum &
                )

                return
            end if

            call rebinning_interpolation(&
                input_wavelengths, input_spectrum, rebin_bin_low, rebin_bin_high, n_rebin, n_wavelengths, &
                rebinned_spectrum &
            )
        end subroutine rebin_spectrum
    
        
        subroutine rebin_spectrum_bin(input_wavelengths, input_spectrum, rebinned_wavelengths, bin_widths, &
                                      n_rebin, n_wavelengths, rebinned_spectrum)
            use rebin_utils, only: rebinning_interpolation, write_input_size_error, write_out_of_bounds_error

            implicit none
            ! I/O
            integer, intent(in) :: n_rebin, n_wavelengths
            double precision, intent(in) :: rebinned_wavelengths(n_rebin), bin_widths(n_rebin)
            double precision, intent(in) :: input_wavelengths(n_wavelengths),input_spectrum(n_wavelengths)
            double precision, intent(out) :: rebinned_spectrum(n_rebin)
            ! internal
            double precision :: rebin_bin_low(n_rebin), rebin_bin_high(n_rebin)
            
            ! Check output array size
            if (n_rebin < 2) then
                call write_input_size_error(n_rebin, rebinned_spectrum)

                return
            end if
    
            rebinned_spectrum = 0d0

            ! Get bin boundaries
            rebin_bin_low = rebinned_wavelengths - bin_widths / 2d0
            rebin_bin_high = rebinned_wavelengths + bin_widths / 2d0

            if ((input_wavelengths(1) >= rebin_bin_low(1)) &
                    .or. (input_wavelengths(n_wavelengths) <= rebin_bin_high(n_rebin))) then

                call write_out_of_bounds_error(&
                    input_wavelengths(1), input_wavelengths(n_wavelengths), &
                    rebin_bin_low(1), rebin_bin_high(n_wavelengths), n_rebin, &
                    rebinned_spectrum &
                )

                return
            end if
    
            call rebinning_interpolation(&
                input_wavelengths, input_spectrum, rebin_bin_low, rebin_bin_high, n_rebin, n_wavelengths, &
                rebinned_spectrum &
            )
        end subroutine rebin_spectrum_bin
end module fortran_rebin
