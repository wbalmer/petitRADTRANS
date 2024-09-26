! """
! Utility functions to convolve spectra for the petitRADTRANS radiative transfer package.
! """
module fortran_convolve
    implicit none

    contains
    subroutine variable_width_convolution(input_wavelength,input_flux,resolutions,&
                                          input_spectrum_length,convolved_spectrum)
        ! """
        ! Convolve a spectrum with a variable with gaussian kernel.
        ! Based on implementation from Ben Burningham for Brewster.
        ! https://github.com/fwang23nh/brewster_v2/blob/master/bbconv.f90
        !
        ! input_wavelength - units of micron
        ! input_flux - arbitrary units
        ! resolutions - spectral resolving power, R
        ! convolved_spectrum - same units as input_flux
        ! f2py intent(inout) modspec,observed_spectrum
        ! f2py intent(out) convolved_spectrum
        ! """
        implicit none

        integer,intent(in) :: input_spectrum_length
        double precision,intent(in) :: input_wavelength(input_spectrum_length), input_flux(input_spectrum_length), &
        resolutions(input_spectrum_length)

        double precision,intent(out) :: convolved_spectrum(input_spectrum_length)
        double precision :: gauss(input_spectrum_length)
        integer :: i
        double precision:: sigma


        do i = 1, input_spectrum_length
           !sigma is FWHM / 2.355
           sigma = (input_wavelength(i) / resolutions(i)) / 2.355
           gauss = exp(-(input_wavelength-input_wavelength(i))**2/(2*sigma**2))
           gauss = gauss/ sum(gauss)
           convolved_spectrum(i)= sum(gauss*input_flux)
        end do

      end subroutine variable_width_convolution
end module fortran_convolve
