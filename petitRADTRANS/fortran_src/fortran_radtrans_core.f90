! """
! Utility functions to calculate cloud opacities, optical epths, spectra, and spectral contribution functions for the
! petitRADTRANS radiative transfer package.
! """


module math
    ! """
    ! Useful mathematicl functions.
    ! """
    implicit none

    double precision, parameter :: cst_pi = 3.14159265359d0  ! TODO update 3.141592653589793d0
    
    contains
        subroutine linear_fit(x, y, ndata, a, b)
            ! """
            ! Calculate slope and y-axis intercept of x,y data, assuming zero error on data.
            ! """
            implicit none
            
            integer, intent(in) :: ndata
            double precision, intent(in) :: x(ndata), y(ndata)
            double precision, intent(out) :: a, b
            
            b = (sum(x)*sum(y)/dble(ndata) - sum(x*y))/ &
                (sum(x)**2d0/dble(ndata) - sum(x**2d0))
            a = sum(y-b*x)/dble(ndata)
        end subroutine  linear_fit
        
        
        subroutine linear_interpolate(x, y, x_out, input_len, output_len, y_out)
            ! """Implementation of linear interpolation function.
            !
            ! Takes arrays of points in x and y, together with an
            ! array of output points. Interpolates to find
            ! the output y-values.
            ! """
            implicit none

            ! inputs
            integer, intent(in) :: input_len, output_len
            double precision, intent(in) :: x(input_len), y(input_len), x_out(output_len)

            ! outputs
            double precision, intent(inout) :: y_out(output_len)

            ! internal
            integer :: i, interp_ind(output_len)
            double precision :: dx, dy, delta_x

            call find_interpolate_indices_(x, input_len, x_out, output_len, interp_ind)

            do i = 1, output_len
                dy = y(interp_ind(i) + 1) - y(interp_ind(i))
                dx = x(interp_ind(i) + 1) - x(interp_ind(i))
                delta_x = x_out(i) - x(interp_ind(i))
                y_out(i) = y(interp_ind(i)) + ((dy / dx) * delta_x)
            enddo
            
            contains
                subroutine find_interpolate_indices_(binbord,binbordlen,arr,arrlen,intpint)
                    ! """
                    ! Self-written? Too long ago... Check if not rather from numrep...
                    ! """
                  implicit none
        
                  integer            :: binbordlen, arrlen, intpint(arrlen)
                  double precision   :: binbord(binbordlen),arr(arrlen)
                  integer            :: i_arr
                  integer            :: pivot, k0, km
        
                  ! carry out a binary search for the interpolation bin borders
                  do i_arr = 1, arrlen
        
                     if (arr(i_arr) >= binbord(binbordlen)) then
                        intpint(i_arr) = binbordlen - 1
                     else if (arr(i_arr) <= binbord(1)) then
                        intpint(i_arr) = 1
                !!$        write(*,*) 'yes', arr(i_arr),binbord(1)
                     else
        
                        k0 = 1
                        km = binbordlen
                        pivot = (km+k0)/2
        
                        do while(km-k0>1)
        
                           if (arr(i_arr) >= binbord(pivot)) then
                              k0 = pivot
                              pivot = (km+k0)/2
                           else
                              km = pivot
                              pivot = (km+k0)/2
                           end if
        
                        end do
        
                        intpint(i_arr) = k0
        
                     end if
        
                  end do
        
                end subroutine find_interpolate_indices_
        end subroutine  linear_interpolate
    
        
        subroutine quicksort_swap_wrapper(length, array)
          implicit none
          integer, intent(in) :: length
          double precision, intent(inout) :: array(length, 2)
          double precision :: swapped_array(2, length)

          swapped_array(1, :) = array(:, 1)
          swapped_array(2, :) = array(:, 2)
          call quicksort_2d_swapped(length, swapped_array)
          array(:,1) = swapped_array(1,:)
          array(:,2) = swapped_array(2,:)

        end subroutine quicksort_swap_wrapper


        recursive subroutine quicksort_2d_swapped(length, array)
            implicit none
            integer, intent(in) :: length
            double precision, intent(inout) :: array(2,length)
            integer :: partition_index
            integer :: ind_up, &
               ind_down, &
               ind_down_start
            double precision :: buffer(2), compare(2)
            logical :: found
            
            found = .False.
            
            partition_index = length
            compare = array(:, partition_index)
            
            ind_down_start = length-1
            
            do ind_up = 1, length-1
                 if (array(1,ind_up) > compare(1)) then
                    found = .True.
                
                    do ind_down = ind_down_start, 1, -1
                       if (ind_down == ind_up) then
                          array(:,partition_index) = array(:,ind_down)
                          array(:,ind_down) = compare
                
                          if ((length-ind_down) > 1) then
                             call quicksort_2d_swapped(length-ind_down, array(:,ind_down+1:length))
                          end if
                          if ((ind_down-1) > 1) then
                             call quicksort_2d_swapped(ind_down-1, array(:,1:ind_down-1))
                          end if
                          return
                       else if (array(1,ind_down) < compare(1)) then
                          buffer = array(:,ind_up)
                          array(:,ind_up) = array(:,ind_down)
                          array(:,ind_down) = buffer
                          ind_down_start = ind_down
                          exit
                       end if
                    end do
                 end if
            end do
            
            if (found .EQV. .FALSE.) then
                 if ((length-1) > 1 ) then
                    call quicksort_2d_swapped(length-1, array(:,1:length-1))
                 end if
            end if
        end subroutine quicksort_2d_swapped


        subroutine solve_tridiagonal_system(a, b, c, res, length, solution)
            ! """
            ! Solves tridiagonal systems of linear equations. Source: some numerical recipes book.
            ! """
            implicit none

            double precision, parameter :: sqrt_hugest = sqrt(huge(0d0)), tiniest = tiny(0d0)
            ! I/O
            integer, intent(in) :: length
            double precision, intent(in) :: a(length), &
                b(length), &
                c(length), &
                res(length)
            double precision, intent(out) :: solution(length)

            ! Internal variables
            logical :: huge_value_warning_trigger
            integer :: ind
            double precision :: buffer_scalar, buffer_vector(length), solution_pre

            huge_value_warning_trigger = .false.

            ! Test if b(1) == 0:
            if (abs(b(1)) < tiniest) then
                stop "Error in tridag routine, b(1) must not be zero!"  ! TODO remove fortran stops and replace with error output
            end if

            ! Begin inversion
            buffer_scalar = b(1)
            solution(1) = res(1) / buffer_scalar

            do ind = 2, length
                buffer_vector(ind) = c(ind - 1) / buffer_scalar
                buffer_scalar = b(ind) - a(ind) * buffer_vector(ind)  ! TODO might be possible to gain time here by pre-calculating buffer_scalar

                if (abs(buffer_scalar) < tiniest) then
                    write(*, *) "Error: tridag routine failed!"

                    solution = 0d0

                    return
                end if

                solution(ind) = res(ind) / buffer_scalar
                solution_pre = solution(ind - 1) / buffer_scalar

                if(solution_pre <= sqrt_hugest) then ! less accurate than a proper overflow detection, but less costly
                    solution(ind) = max(solution(ind) - solution_pre * a(ind), tiniest)  ! max() prevents < 0 solutions
                else  ! overflow prevention
                    if(.not. huge_value_warning_trigger) then
                        write(&
                            *, &
                            '("solve_tridiagonal_system: Warning: &
                            &very high value (> ", ES11.3, ") found during inversion, capping solution...")'&
                        ) sqrt_hugest
                        huge_value_warning_trigger = .true.
                    end if

                    solution(ind) = sqrt_hugest  ! capping value to avoid infinity
                end if
            end do

            do ind = length - 1, 1, -1
                solution(ind) = solution(ind) - buffer_vector(ind + 1) * solution(ind + 1)
            end do
        end subroutine solve_tridiagonal_system
end module math


module physics
    ! """
    ! Physical constants defintion and useful physical functions.
    ! """
    implicit none

    double precision, parameter :: cst_c = 2.99792458d10  ! (cm.s-1) speed of light in vacuum
    double precision, parameter :: cst_k = 1.3806488d-16  ! (g.cm2.s-2.K-1) Boltzmann constant TODO update 1.380649d-16
    double precision, parameter :: cst_h = 6.62606957d-27  ! (g.cm2.s-1) Planck constant  ! TODO update 6.62607015d-27
    double precision, parameter :: cst_amu = 1.66053892d-24  ! (g) atomic mass constant  ! TODO update 1.6605390666e-24
    double precision, parameter :: cst_sneep_ubachs_n = 25.47d18  ! TODO what is this?

    contains
        subroutine compute_planck_function(temperatures, frequencies, n_layers, planck_flux)
            ! """
            ! Calculate the Planck source function.
            ! """
            implicit none
            integer, intent(in) :: n_layers
            double precision, intent(in) :: temperatures(n_layers), frequencies
            double precision, intent(out) :: planck_flux(n_layers)
            double precision :: buffer

            !~~~~~~~~~~~~~

            planck_flux = 0d0
            buffer = 2d0*cst_h*frequencies**3d0/cst_c**2d0
            planck_flux = buffer / (exp(cst_h*frequencies/cst_k/temperatures)-1d0)
        end subroutine compute_planck_function


        subroutine compute_planck_function_temperature_derivative(temperatures, frequencies, n_frequencies, &
                                                                  B_nu_dT)
            implicit none

            integer, intent(in) :: n_frequencies
            double precision, intent(in) :: temperatures, frequencies(n_frequencies)
            double precision, intent(out) :: B_nu_dT(n_frequencies-1)
            double precision :: buffer(n_frequencies-1), nu_use(n_frequencies-1)
            integer :: i

            do i = 1, n_frequencies-1
             nu_use(i) = (frequencies(i)+frequencies(i+1))/2d0
            end do

            buffer = 2d0*cst_h**2d0*nu_use**4d0/cst_c**2d0
            B_nu_dT = buffer / (&
                (exp(cst_h*nu_use/cst_k/temperatures/2d0)-exp(-cst_h*nu_use/cst_k/temperatures/2d0))**2d0 &
            ) / cst_k / temperatures**2d0
        end subroutine compute_planck_function_temperature_derivative


        subroutine compute_planck_function_integral(temperatures, frequencies, n_layers, n_frequencies, &
                                                    planck_flux)
            ! """
            ! Compute mean using Boole's method
            ! """
            implicit none

            double precision, parameter :: cst_c2 = cst_c ** 2d0
            double precision, parameter :: cst_h_c2 = cst_h / cst_c2
            double precision, parameter :: cst_h_k = cst_h / cst_k
            double precision, parameter :: factor1 = 1d0 / 90d0
            double precision, parameter :: factor2 = 14d0 * cst_h_c2 ! (7 * 2)
            double precision, parameter :: factor3 = 64d0 * cst_h_c2 ! (32 * 2)
            double precision, parameter :: factor4 = 24d0 * cst_h_c2 ! (12 * 2)

            integer, intent(in) :: n_frequencies
            integer, intent(in) :: n_layers
            double precision, intent(in) :: frequencies(n_frequencies)
            double precision, intent(in) :: temperatures(n_layers)
            double precision, intent(out) :: planck_flux(n_layers, n_frequencies - 1)

            integer :: i
            double precision :: nu2, nu3, nu4, diff_nu, nu_large, nu_small
            double precision :: cst_h_kt(n_layers)

            planck_flux(:, :) = 0d0

            if (frequencies(2) > frequencies(1)) then
                write(*, *) "Error: frequencies must be in decreasing order"
                stop
            end if

            cst_h_kt = cst_h_k / temperatures

            do i = 1, n_frequencies - 1
                nu_large = frequencies(i)
                nu_small = frequencies(i + 1)

                diff_nu = nu_large - nu_small

                nu2 = nu_small + diff_nu * 0.25d0
                nu3 = nu_small + diff_nu * 0.5d0
                nu4 = nu_small + diff_nu * 0.75d0

                planck_flux(:, i) = planck_flux(:, i) + factor1 * ( &
                    factor2 * nu_small ** 3d0 / (exp(cst_h_kt * nu_small) - 1d0) &
                    + factor3 * nu2 ** 3d0 / (exp(cst_h_kt * nu2) - 1d0) &
                    + factor4 * nu3 ** 3d0 / (exp(cst_h_kt * nu3) - 1d0) &
                    + factor3 * nu4 ** 3d0 / (exp(cst_h_kt * nu4) - 1d0) &
                    + factor2 * nu_large ** 3d0 / (exp(cst_h_kt * nu_large) - 1d0) &
                )
            end do
        end subroutine compute_planck_function_integral


        subroutine compute_rosseland_opacities_core(clouds_final_absorption_opacities, frequencies_bin_edges, &
                                                    temperature, weights_gauss, n_frequencies_bins, n_g, &
                                                    opacities_rosseland)
            implicit none

            integer, intent(in) :: n_g, n_frequencies_bins
            double precision, intent(in) :: frequencies_bin_edges(n_frequencies_bins)
            double precision, intent(in) :: clouds_final_absorption_opacities(n_g, n_frequencies_bins-1)
            double precision, intent(in) :: temperature, weights_gauss(n_g)
            double precision, intent(out) :: opacities_rosseland

            integer :: i
            double precision :: B_nu_dT(n_frequencies_bins-1), numerator

            call compute_planck_function_temperature_derivative(&
                temperature, frequencies_bin_edges, n_frequencies_bins, B_nu_dT &
            )

            opacities_rosseland = 0d0
            numerator = 0d0

            do i = 1, n_frequencies_bins-1
                opacities_rosseland = opacities_rosseland + &
                B_nu_dT(i) * sum(weights_gauss/clouds_final_absorption_opacities(:,i)) &
                    * (frequencies_bin_edges(i)-frequencies_bin_edges(i+1))
                numerator = numerator &
                    + B_nu_dT(i) &
                    * (frequencies_bin_edges(i)-frequencies_bin_edges(i+1))
            end do

            opacities_rosseland = numerator / opacities_rosseland
        end subroutine compute_rosseland_opacities_core


        subroutine compute_star_planck_function_integral(n_frequencies,temperatures,frequencies,planck_flux)
            implicit none
            integer                         :: n_frequencies
            double precision                :: temperatures,planck_flux(n_frequencies-1), frequencies(n_frequencies)

            double precision :: t_tmp(1), b_nu_tmp(1, n_frequencies-1)

            t_tmp(1) = temperatures
            b_nu_tmp(:, :) = 0d0

            call compute_planck_function_integral(&
                t_tmp, frequencies, 1, n_frequencies, b_nu_tmp &
            )

            planck_flux(:) = b_nu_tmp(1, :)

        end subroutine compute_star_planck_function_integral
end module physics


module cloud_utils
    ! """
    ! Useful functions for cloud simulation.
    ! """
    implicit none
    
    contains
        subroutine compute_turbulent_settling_speed(x, surface_gravity, rho, clouds_particles_densities,&
                                                    temperatures, mean_molar_masses, &
                                                    turbulent_settling_speed_ret)
            use math, only: cst_pi
            use physics, only: cst_amu, cst_k

            implicit none

            double precision, parameter :: d = 2.827d-8, epsilon = 59.7*cst_k

            double precision, intent(in) :: &
                x, surface_gravity, rho, clouds_particles_densities, temperatures, mean_molar_masses
            double precision, intent(out) :: turbulent_settling_speed_ret

            double precision    :: N_Knudsen, psi, eta, CdNreSq, Nre, Cd, v_settling_visc


            N_Knudsen = mean_molar_masses*cst_amu/(cst_pi*rho*d**2d0*x)
            psi = 1d0 + N_Knudsen*(1.249d0+0.42d0*exp(-0.87d0*N_Knudsen))
            eta = 15d0/16d0*sqrt(cst_pi*2d0*cst_amu*cst_k*temperatures)/(cst_pi*d**2d0) &
            *(cst_k*temperatures/epsilon)**0.16d0/1.22d0
            CdNreSq = 32d0*rho*surface_gravity*x**3d0*(clouds_particles_densities-rho)/(3d0*eta**2d0)
            Nre = exp(-2.7905d0+0.9209d0*log(CdNreSq)-0.0135d0*log(CdNreSq)**2d0)

            if (Nre < 1d0) then
                Cd = 24d0
            else if (Nre > 1d3) then
                Cd = 0.45d0
            else
                Cd = CdNreSq/Nre**2d0
            end if

            v_settling_visc = 2d0*x**2d0*(clouds_particles_densities-rho)*psi*surface_gravity/(9d0*eta)
            turbulent_settling_speed_ret = psi*sqrt(8d0*surface_gravity*x*(clouds_particles_densities-rho)/(3d0*Cd*rho))

            if ((Nre < 1d0) .and. (v_settling_visc < turbulent_settling_speed_ret)) then
                turbulent_settling_speed_ret = v_settling_visc
            end if
        end subroutine compute_turbulent_settling_speed


        function particle_radius(x1, x2, surface_gravity, rho, clouds_particles_densities, temperatures, &
                                 mean_molar_masses, w_star)
            ! """
            ! Find the particle radius, using a simple bisection method.
            ! """
          implicit none

          integer, parameter :: ITMAX = 1000

          double precision, intent(in) :: &
              surface_gravity, rho, clouds_particles_densities, temperatures, mean_molar_masses, w_star
          double precision, intent(in) :: x1, x2
          double precision :: particle_radius

          integer :: iter
          double precision :: a,b,c,fa,fb,fc,del

          a=x1
          b=x2
          call compute_turbulent_settling_speed(&
              a,surface_gravity,rho,clouds_particles_densities,temperatures,mean_molar_masses,fa&
          )
          fa = fa - w_star
          call compute_turbulent_settling_speed(&
              b,surface_gravity,rho,clouds_particles_densities,temperatures,mean_molar_masses,fb&
          )
          fb = fb - w_star

          if((fa>0..and.fb>0.).or.(fa<0..and.fb<0.)) then
             !write(*,*) 'warning: root must be bracketed for zbrent'
             particle_radius = 1d-17
             return
          end if

          do iter=1,ITMAX

             if (abs(log10(a/b)) > 1d0) then
                c = 1e1**(log10(a*b)/2d0)
             else
                c = (a+b)/2d0
             end if

             call compute_turbulent_settling_speed(&
                 c,surface_gravity,rho,clouds_particles_densities,temperatures,mean_molar_masses,fc&
             )
             fc = fc - w_star

             if (((fc > 0d0) .and. (fa > 0d0)) .or. ((fc < 0d0) .and. (fa < 0d0))) then
                del = 2d0*abs(a-c)/(a+b)
                a = c
                fa = fc
             else
                del = 2d0*abs(b-c)/(a+b)
                b = c
                fb = fc
             end if

             if (abs(del) < 1d-9) then
                exit
             end if

          end do

          if (iter == ITMAX) then
             write(*,*) 'warning: maximum number of bisection root iterations reached!'
          end if

          particle_radius = c
          return

        end function particle_radius
end module cloud_utils


module fortran_radtrans_core
    ! """
    ! Functions used in the Radtrans object.
    ! """
    implicit none

    contains
        subroutine combine_ck_opacities(opacities, g_gauss, weights_gauss, &
                                        n_g, n_frequencies, n_species, n_layers, opacities_out)
            ! """Subroutine to completely mix the c-k opacities.
            ! """
            use math, only: linear_interpolate, quicksort_swap_wrapper

            implicit none

            integer, intent(in)          :: n_g, n_frequencies, n_species, n_layers
            double precision, intent(in) :: opacities(n_g, n_frequencies, &
                    n_species, n_layers), g_gauss(n_g), weights_gauss(n_g)
            double precision, intent(out) :: opacities_out(n_g, n_frequencies, n_layers)

            double precision, parameter :: threshold_coefficient = 1d-3

            integer :: i_freq, i_spec, i_struc, i_samp, i_g, j_g, n_sample
            double precision :: weights_use(n_g)
            double precision :: cum_sum, k_min(n_frequencies, n_layers), k_max(n_frequencies, n_layers), &
                    g_out(n_g ** 2 + 1), k_out(n_g ** 2 + 1), &
                    g_presort(n_g ** 2 + 1), &
                    sampled_opa_weights(n_g ** 2, 2), &
                    spec2(n_g)

            double precision :: threshold(n_frequencies, n_layers)

            n_sample = n_g ** 2

            k_min = 0d0
            k_max = 0d0
            weights_use = weights_gauss
            weights_use(1:8) = weights_use(1:8) / 3d0

            ! In every layer and frequency bin:
            ! Find the species with the largest kappa(g=0) value, and use it to get the kappas threshold.
            do i_struc = 1, n_layers
                do i_freq = 1, n_frequencies
                    threshold(i_freq, i_struc) = maxval(opacities(1, i_freq, :, i_struc)) &
                        * threshold_coefficient
                end do
            end do

            ! This is for the everything-with-everything combination, if only two species
            ! get combined. Only need to do this here once.
            do i_g = 1, n_g
                do j_g = 1, n_g
                    g_presort((i_g - 1) * n_g + j_g) = weights_gauss(i_g) * weights_gauss(j_g)
                end do
            end do

            ! Here we'll loop over every entry, mix and add the kappa values,
            ! calculate the g-weights and then interpolate back to the standard
            ! g-grid.

            opacities_out = opacities(:, :, 1, :)

            if (n_species > 1) then
                do i_struc = 1, n_layers
                    do i_freq = 1, n_frequencies
                        do i_spec = 2, n_species
                            ! Neglect kappas below the threshold
                            if (opacities(n_g, i_freq, i_spec, i_struc) < threshold(i_freq, i_struc)) then
                                cycle
                            endif

                            spec2 = opacities(:, i_freq, i_spec, i_struc)

                            k_out = 0d0

                            do i_g = 1, n_g
                                do j_g = 1, n_g
                                    k_out((i_g-1) * n_g + j_g) = opacities_out(i_g, i_freq, i_struc) &
                                        + spec2(j_g)
                                end do
                            end do

                            sampled_opa_weights(:, 1) = k_out(1:n_sample)
                            sampled_opa_weights(:, 2) = g_presort(1:n_sample)

                            call quicksort_swap_wrapper(n_sample, sampled_opa_weights)

                            sampled_opa_weights(:, 2) = sampled_opa_weights(:, 2) / sum(sampled_opa_weights(:, 2))

                            g_out = 0d0
                            cum_sum = 0d0

                            do i_samp = 1, n_sample
                                g_out(i_samp) = sampled_opa_weights(i_samp, 2) * 0.5d0 + cum_sum
                                cum_sum = cum_sum + sampled_opa_weights(i_samp, 2)
                            end do

                            g_out(n_sample + 1) = 1d0

                            k_out(1:n_sample) = sampled_opa_weights(:, 1)
                            k_out(1) = opacities_out(1, i_freq, i_struc) + spec2(1)
                            k_out(n_sample+1) = opacities_out(n_g, i_freq, i_struc) + spec2(n_g)

                            ! Linearly interpolate back to the 16-point grid, storing in the output array
                            call  linear_interpolate(g_out, k_out, g_gauss, n_sample + 1, n_g, &
                                                     opacities_out(:, i_freq, i_struc))
                        end do
                    end do
                end do
            endif
        end subroutine combine_ck_opacities


        subroutine compute_ck_flux(frequencies, optical_depths, temperatures, &
                                   emission_cos_angles, emission_cos_angles_weights, weights_gauss, &
                                   return_contribution, n_frequencies, n_layers, n_angles, n_g, n_species, &
                                   flux, emission_contribution)
            ! """
            ! Calculate the radiative transport, using the mean transmission method.
            ! """
          use math, only: cst_pi
          use physics, only: compute_planck_function
          implicit none

          ! I/O
          integer, intent(in)                         :: n_frequencies, n_layers,n_g, n_species
          double precision, intent(in)                :: frequencies(n_frequencies)
          double precision, intent(in)                :: temperatures(n_layers)
          double precision, intent(in)                :: optical_depths(n_g,n_frequencies,n_species,n_layers)

          integer, intent(in)                         :: n_angles
          double precision, intent(in)                :: emission_cos_angles(n_angles)
          double precision, intent(in)                :: emission_cos_angles_weights(n_angles)
          double precision, intent(in)                :: weights_gauss(n_g)
          logical, intent(in)                         :: return_contribution
          double precision, intent(out)               :: flux(n_frequencies)
          double precision, intent(out)               :: emission_contribution(n_layers,n_frequencies)

          ! Internal
          integer                                     :: i_mu,i_freq,i_str,i_spec
          double precision                            :: r(n_layers)
          double precision                            :: transm_mu(n_g,n_frequencies,n_species,n_layers), &
               mean_transm(n_frequencies,n_species,n_layers), transm_all(n_frequencies,n_layers), &
               transm_all_loc(n_layers), flux_mu(n_frequencies)

          flux = 0d0

          if (return_contribution) then
             emission_contribution = 0d0
          end if

          do i_mu = 1, n_angles

             ! will contain species' product of g-space integrated transmissions
             transm_all = 1d0
             ! Transmissions for a given incidence angle
             transm_mu = exp(-optical_depths/emission_cos_angles(i_mu))
             ! Flux contribution from given mu-angle
             flux_mu = 0d0

             do i_str = 1, n_layers
                do i_spec = 1, n_species
                   do i_freq = 1, n_frequencies
                      ! Integrate transmission over g-space
                      mean_transm(i_freq,i_spec,i_str) = sum(transm_mu(:,i_freq,i_spec,i_str)*weights_gauss)
                   end do
                end do
             end do

             ! Multiply transmissions of infdiv. species
             do i_spec = 1, n_species
                transm_all = transm_all*mean_transm(:,i_spec,:)
             end do

             ! Do the actual radiative transport
             do i_freq = 1, n_frequencies
                ! Get source function
                r = 0
                call compute_planck_function(temperatures, frequencies(i_freq), n_layers, r)
                ! Spatial transmissions at given wavelength
                transm_all_loc = transm_all(i_freq,:)
                ! Calc Eq. 9 of manuscript (em_deriv.pdf)
                do i_str = 1, n_layers-1
                   flux_mu(i_freq) = flux_mu(i_freq)+ &
                        (r(i_str)+r(i_str+1))*(transm_all_loc(i_str)-transm_all_loc(i_str+1))/2d0
                   if (return_contribution) then
                      emission_contribution(i_str,i_freq) = emission_contribution(i_str,i_freq) &
                          + (r(i_str)+r(i_str+1)) * &
                           (transm_all_loc(i_str)-transm_all_loc(i_str+1)) &
                           *emission_cos_angles(i_mu)*emission_cos_angles_weights(i_mu)
                   end if
                end do
                flux_mu(i_freq) = flux_mu(i_freq) + r(n_layers)*transm_all_loc(n_layers)
                if (return_contribution) then
                   emission_contribution(n_layers,i_freq) = emission_contribution(n_layers,i_freq) + 2d0*r(n_layers)* &
                        transm_all_loc(n_layers)*emission_cos_angles(i_mu)*emission_cos_angles_weights(i_mu)
                end if
             end do
             ! angle integ, factor 1/2 needed for flux calc. from upward pointing intensity
             flux = flux + flux_mu/2d0*emission_cos_angles(i_mu)*emission_cos_angles_weights(i_mu)

          end do
          ! Normalization
          flux = flux*4d0*cst_pi

          if (return_contribution) then
             do i_freq = 1, n_frequencies
                emission_contribution(:,i_freq) = emission_contribution(:,i_freq)/sum(emission_contribution(:,i_freq))
             end do
          end if

        end subroutine compute_ck_flux


        subroutine compute_cloud_hansen_opacities(atmospheric_densities, clouds_particles_densities, &
                                                  clouds_species_mass_fractions, clouds_a_hansen, clouds_b_hansen, &
                                                  clouds_particles_radii_bins, clouds_particles_radii, &
                                                  clouds_absorption_opacities, clouds_scattering_opacities, &
                                                  clouds_particles_asymmetry_parameters, &
                                                  n_layers, n_clouds, n_particles_radii, n_cloud_wavelengths, &
                                                  clouds_total_absorption_opacities, &
                                                  clouds_total_scattering_opacities, &
                                                  clouds_total_red_fac_aniso)
            ! """
            ! Calculate cloud opacities.
            ! """
            use math, only: cst_pi

            implicit none

            integer, intent(in) :: n_layers, n_clouds, n_particles_radii, n_cloud_wavelengths
            double precision, intent(in) :: atmospheric_densities(n_layers), clouds_particles_densities(n_clouds)
            double precision, intent(in) :: clouds_species_mass_fractions(n_layers,n_clouds), &
                clouds_a_hansen(n_layers,n_clouds), clouds_b_hansen(n_layers,n_clouds)
            double precision, intent(in) :: clouds_particles_radii_bins(n_particles_radii+1), &
                clouds_particles_radii(n_particles_radii)
            double precision, intent(in) :: clouds_absorption_opacities(n_particles_radii, &
                n_cloud_wavelengths,n_clouds), &
                clouds_scattering_opacities(n_particles_radii,n_cloud_wavelengths,n_clouds), &
                clouds_particles_asymmetry_parameters(n_particles_radii,n_cloud_wavelengths,n_clouds)
            double precision, intent(out) :: clouds_total_absorption_opacities(n_cloud_wavelengths,n_layers), &
                clouds_total_scattering_opacities(n_cloud_wavelengths,n_layers), &
                clouds_total_red_fac_aniso(n_cloud_wavelengths,n_layers)

            integer :: i_struc, i_spec, i_lamb, i_cloud
            double precision :: N, dndr(n_particles_radii), integrand_abs(n_particles_radii), mass_to_vol, &
                integrand_scat(n_particles_radii), add_abs, add_scat, integrand_aniso(n_particles_radii), add_aniso, &
                dndr_scale

            clouds_total_absorption_opacities = 0d0
            clouds_total_scattering_opacities = 0d0
            clouds_total_red_fac_aniso = 0d0

            do i_struc = 1, n_layers
                do i_spec = 1, n_clouds
                    do i_lamb = 1, n_cloud_wavelengths
                        mass_to_vol = 0.75d0 * clouds_species_mass_fractions(i_struc, i_spec) &
                            * atmospheric_densities(i_struc) / cst_pi / clouds_particles_densities(i_spec)

                        N = mass_to_vol / (&
                            clouds_a_hansen(i_struc, i_spec) ** 3d0 * (clouds_b_hansen(i_struc,i_spec) - 1d0) &
                            * (2d0 * clouds_b_hansen(i_struc,i_spec) - 1d0) &
                        )
                        dndr_scale = &
                            log(N) + log(clouds_a_hansen(i_struc, i_spec) * clouds_b_hansen(i_struc, i_spec)) &
                                * (&
                                    (2d0 * (clouds_b_hansen(i_struc, i_spec)) - 1d0) &
                                    / clouds_b_hansen(i_struc, i_spec) &
                                ) &
                                - log(&
                                    gamma((1d0 - 2d0 * clouds_b_hansen(i_struc, i_spec)) &
                                    / clouds_b_hansen(i_struc, i_spec)) &
                                )

                        do i_cloud = 1, n_particles_radii
                            dndr(i_cloud) = dndr_scale + hansen_size_nr(&
                                clouds_particles_radii(i_cloud), &
                                clouds_a_hansen(i_struc,i_spec), &
                                clouds_b_hansen(i_struc,i_spec) &
                            )
                        end do

                        dndr = exp(dndr)

                        integrand_abs = 0.75d0 * cst_pi * clouds_particles_radii ** 3d0 &
                            * clouds_particles_densities(i_spec) * dndr &
                            * clouds_absorption_opacities(:,i_lamb,i_spec)
                        integrand_scat = 0.75d0 * cst_pi * clouds_particles_radii ** 3d0 &
                            * clouds_particles_densities(i_spec) * dndr &
                            * clouds_scattering_opacities(:,i_lamb,i_spec)
                        integrand_aniso = integrand_scat &
                            * (1d0 - clouds_particles_asymmetry_parameters(:, i_lamb, i_spec))

                        add_abs = sum(&
                            integrand_abs &
                            * ( &
                                clouds_particles_radii_bins(2:n_particles_radii + 1) &
                                - clouds_particles_radii_bins(1:n_particles_radii)&
                            ) &
                        )
                        clouds_total_absorption_opacities(i_lamb,i_struc) = &
                            clouds_total_absorption_opacities(i_lamb, i_struc) + add_abs

                        add_scat = sum(&
                            integrand_scat &
                            * (&
                            clouds_particles_radii_bins(2:n_particles_radii + 1) &
                            - clouds_particles_radii_bins(1:n_particles_radii) &
                            ) &
                        )
                        clouds_total_scattering_opacities(i_lamb, i_struc) = &
                            clouds_total_scattering_opacities(i_lamb, i_struc) + add_scat

                        add_aniso = sum(&
                            integrand_aniso &
                            * (clouds_particles_radii_bins(2:n_particles_radii+1) &
                               - clouds_particles_radii_bins(1:n_particles_radii)) &
                        )
                        clouds_total_red_fac_aniso(i_lamb, i_struc) = &
                            clouds_total_red_fac_aniso(i_lamb, i_struc) + add_aniso
                    end do
                end do

                do i_lamb = 1, n_cloud_wavelengths
                    if (clouds_total_scattering_opacities(i_lamb,i_struc) > 1d-200) then
                        clouds_total_red_fac_aniso(i_lamb, i_struc) = &
                            clouds_total_red_fac_aniso(i_lamb, i_struc) &
                            / clouds_total_scattering_opacities(i_lamb, i_struc)
                    else
                        clouds_total_red_fac_aniso(i_lamb, i_struc) = 0d0
                    end if
                end do

                clouds_total_absorption_opacities(:, i_struc) = &
                    clouds_total_absorption_opacities(:, i_struc) / atmospheric_densities(i_struc)
                clouds_total_scattering_opacities(:, i_struc) = &
                    clouds_total_scattering_opacities(:, i_struc) / atmospheric_densities(i_struc)
            end do

            contains
                function hansen_size_nr(r,a,b)
                    implicit none

                    double precision :: r, a, b
                    double precision :: hansen_size_nr

                    hansen_size_nr = log(r) * (1d0 - 3d0 * b) / b - 1d0 * r / (a * b)
                end function hansen_size_nr
        end subroutine compute_cloud_hansen_opacities


        subroutine compute_cloud_optical_depths(surface_gravity, pressures, opacities, &
                                                n_layers, n_frequencies, n_g, n_species, &
                                                optical_depths)
            ! """
            ! Calculate optical depths with 2nd order accuracy for clouds.
            ! """

          !use physics
          implicit none

          ! I/O
          integer, intent(in)                          :: n_layers, n_frequencies, n_g, n_species
          double precision, intent(in)                 :: opacities(n_g,n_frequencies,n_species,n_layers)
          double precision, intent(in)                 :: surface_gravity, pressures(n_layers)
          double precision, intent(out)                :: optical_depths(n_g,n_frequencies,n_species,n_layers)
          ! internal
          integer                                      :: i_struc, i_freq, i_g, i_spec
          double precision                             :: del_tau_lower_ord, &
               gamma_second(n_g,n_frequencies,n_species), f_second, kappa_i(n_g,n_frequencies,n_species), &
               kappa_im(n_g,n_frequencies,n_species), kappa_ip(n_g,n_frequencies,n_species)
          logical                                      :: second_order
          !~~~~~~~~~~~~~

          optical_depths = 0d0
          second_order = .FALSE.

          if (second_order) then
             do i_struc = 2, n_layers
                if (i_struc == n_layers) then
                   optical_depths(:,:,:,i_struc) = optical_depths(:,:,:,i_struc-1) + &
                        (opacities(:,:,:,i_struc)+opacities(:,:,:,i_struc-1)) &
                        /2d0/surface_gravity*(pressures(i_struc)-pressures(i_struc-1))
                else
                   f_second = (pressures(i_struc+1)-pressures(i_struc))/(pressures(i_struc)-pressures(i_struc-1))
                   kappa_i = opacities(:,:,:,i_struc)
                   kappa_im = opacities(:,:,:,i_struc-1)
                   kappa_ip = opacities(:,:,:,i_struc+1)
                   gamma_second = (kappa_ip-(1d0+f_second)*kappa_i+f_second*kappa_im) / &
                        (f_second*(1d0+f_second))
                   optical_depths(:,:,:,i_struc) = optical_depths(:,:,:,i_struc-1) + &
                        ((kappa_i+kappa_im)/2d0-gamma_second/6d0) &
                        /surface_gravity*(pressures(i_struc)-pressures(i_struc-1))
                   do i_spec = 1, n_species
                      do i_freq = 1, n_frequencies
                         do i_g = 1, n_g
                            if (optical_depths(i_g,i_freq,i_spec,i_struc) &
                                < optical_depths(i_g,i_freq,i_spec,i_struc-1)) then
                               if (i_struc <= 2) then
                                  optical_depths(i_g,i_freq,i_spec,i_struc) = &
                                       optical_depths(i_g,i_freq,i_spec,i_struc-1)*1.01d0
                               else
                                  optical_depths(i_g,i_freq,i_spec,i_struc) = &
                                       optical_depths(i_g,i_freq,i_spec,i_struc-1) + &
                                       (optical_depths(i_g,i_freq,i_spec,i_struc-1)- &
                                       optical_depths(i_g,i_freq,i_spec,i_struc-2))*0.01d0
                               end if
                            end if
                            del_tau_lower_ord = (kappa_i(i_g,i_freq,i_spec)+ &
                                 kappa_im(i_g,i_freq,i_spec))/2d0/surface_gravity* &
                                 (pressures(i_struc)-pressures(i_struc-1))
                            if ((optical_depths(i_g,i_freq,i_spec,i_struc) - &
                                 optical_depths(i_g,i_freq,i_spec,i_struc-1)) > del_tau_lower_ord) then
                               optical_depths(i_g,i_freq,i_spec,i_struc) = &
                                    optical_depths(i_g,i_freq,i_spec,i_struc-1) + del_tau_lower_ord
                            end if
                         end do
                      end do
                   end do
                end if
             end do
          else
             do i_struc = 2, n_layers
                optical_depths(:,:,:,i_struc) = optical_depths(:,:,:,i_struc-1) + &
                     (opacities(:,:,:,i_struc)+opacities(:,:,:,i_struc-1)) &
                     /2d0/surface_gravity*(pressures(i_struc)-pressures(i_struc-1))
             end do
          end if

        end subroutine compute_cloud_optical_depths


        subroutine compute_cloud_particles_mean_radius(surface_gravity, atmospheric_densities, &
                                                       clouds_particles_densities, temperatures, &
                                                       mean_molar_masses, f_seds, &
                                                       cloud_particle_radius_distribution_std, &
                                                       eddy_diffusion_coefficients, &
                                                       n_layers, n_clouds, &
                                                       clouds_particles_mean_radii)
            use cloud_utils, only: compute_turbulent_settling_speed, particle_radius
            use math, only: linear_fit
            use physics, only: cst_amu, cst_k

            implicit none
          ! I/O
          integer, intent(in)  :: n_layers, n_clouds
          double precision, intent(in) :: surface_gravity, atmospheric_densities(n_layers), &
                clouds_particles_densities(n_clouds), &
                temperatures(n_layers), &
                mean_molar_masses(n_layers), f_seds(n_clouds), &
                cloud_particle_radius_distribution_std, eddy_diffusion_coefficients(n_layers)
          double precision, intent(out) :: clouds_particles_mean_radii(n_layers,n_clouds)
          ! Internal
          integer, parameter :: N_fit = 100
          integer          :: i_str, i_spec, i_rad
          double precision :: w_star(n_layers), H(n_layers)
          double precision :: r_w(n_layers,n_clouds), alpha(n_layers,n_clouds)
          double precision :: rad(N_fit), vel(N_fit), f_fill(n_clouds)
          double precision :: a, b

          H = cst_k*temperatures/(mean_molar_masses*cst_amu*surface_gravity)
          w_star = eddy_diffusion_coefficients/H

          f_fill = 1d0

          do i_str = 1, n_layers
             do i_spec = 1, n_clouds
                r_w(i_str,i_spec) = particle_radius(&
                    1d-16,1d2,surface_gravity,atmospheric_densities(i_str), &
                    clouds_particles_densities(i_spec),temperatures(i_str),mean_molar_masses(i_str),w_star(i_str))
                if (r_w(i_str,i_spec) > 1d-16) then
                   if (f_seds(i_spec) > 1d0) then
                      do i_rad = 1, N_fit
                         rad(i_rad) = r_w(i_str,i_spec)/max(cloud_particle_radius_distribution_std,1.0001d0) &
                            + (r_w(i_str,i_spec)&
                            - r_w(i_str,i_spec)/max(cloud_particle_radius_distribution_std,1.0001d0))&
                            * dble(i_rad-1)/dble(N_fit-1)
                         call compute_turbulent_settling_speed(&
                             rad(i_rad),surface_gravity,atmospheric_densities(i_str),&
                             clouds_particles_densities(i_spec),&
                             temperatures(i_str), &
                            mean_molar_masses(i_str),vel(i_rad) &
                         )
                      end do
                   else
                      do i_rad = 1, N_fit
                         rad(i_rad) = r_w(i_str,i_spec) &
                             + (r_w(i_str,i_spec)*max(cloud_particle_radius_distribution_std,1.0001d0) &
                             - r_w(i_str,i_spec)) &
                             * dble(i_rad-1)/dble(N_fit-1)
                         call compute_turbulent_settling_speed(&
                            rad(i_rad),surface_gravity,atmospheric_densities(i_str),&
                             clouds_particles_densities(i_spec),&
                             temperatures(i_str), &
                            mean_molar_masses(i_str),vel(i_rad) &
                         )
                      end do
                   end if

                   call  linear_fit(log(rad), log(vel/w_star(i_str)), N_fit, a, b)

                   alpha(i_str,i_spec) = b
                   r_w(i_str,i_spec) = exp(-a/b)
                   clouds_particles_mean_radii(i_str,i_spec) = &
                       r_w(i_str,i_spec) * f_seds(i_spec)**(1d0/alpha(i_str,i_spec))&
                       * exp(-(alpha(i_str,i_spec)+6d0)/2d0*log(cloud_particle_radius_distribution_std)**2d0)
                else
                   clouds_particles_mean_radii(i_str,i_spec) = 1d-17
                   alpha(i_str,i_spec) = 1d0
                end if
             end do

          end do

        end subroutine compute_cloud_particles_mean_radius


        subroutine compute_cloud_particles_mean_radius_hansen(surface_gravity, rho, clouds_particles_densities,&
                                                              temperatures,mean_molar_masses,&
                                                              f_seds, clouds_b_hansen, eddy_diffusion_coefficients, &
                                                              n_layers, n_clouds, &
                                                              clouds_a_hansen)
            use cloud_utils, only: compute_turbulent_settling_speed, particle_radius
            use math, only: linear_fit
            use physics, only: cst_amu, cst_k

            implicit none

            integer, intent(in)  :: n_layers, n_clouds
            double precision, intent(in) :: surface_gravity, rho(n_layers), clouds_particles_densities(n_clouds), &
                temperatures(n_layers), mean_molar_masses(n_layers), f_seds(n_clouds), &
                clouds_b_hansen(n_layers,n_clouds), eddy_diffusion_coefficients(n_layers)
            double precision, intent(out) :: clouds_a_hansen(n_layers,n_clouds)

            integer, parameter :: N_fit = 100
            doubleprecision, parameter :: x_gamma_max = 170d0  ! gamma(x_gamma_max) >~ huge(0d0)

            integer          :: i_str, i_spec, i_rad
            double precision :: w_star(n_layers), H(n_layers)
            double precision :: r_w(n_layers,n_clouds), alpha(n_layers, n_clouds), &
                x_gamma(n_layers, n_clouds)
            double precision :: rad(N_fit), vel(N_fit)
            double precision :: a, b

            H = cst_k * temperatures / (mean_molar_masses * cst_amu * surface_gravity)
            w_star = eddy_diffusion_coefficients / H
            x_gamma = 1d0 + 1d0 / clouds_b_hansen  ! argument of the gamma function

            do i_str = 1, n_layers
                do i_spec = 1, n_clouds
                    r_w(i_str,i_spec) = particle_radius(&
                        1d-16,&
                        1d2, &
                        surface_gravity, &
                        rho(i_str), &
                        clouds_particles_densities(i_spec), &
                        temperatures(i_str), &
                        mean_molar_masses(i_str), &
                        w_star(i_str) &
                    )

                    if (r_w(i_str,i_spec) > 1d-16) then
                        if (f_seds(i_spec) > 1d0) then
                            do i_rad = 1, N_fit
                                rad(i_rad) = &
                                    r_w(i_str, i_spec) * clouds_b_hansen(i_str, i_spec) &
                                    + (r_w(i_str, i_spec) - r_w(i_str, i_spec) * clouds_b_hansen(i_str, i_spec)) &
                                    * dble(i_rad - 1) / dble(N_fit - 1)

                                call compute_turbulent_settling_speed(&
                                    rad(i_rad), &
                                    surface_gravity, &
                                    rho(i_str), &
                                    clouds_particles_densities(i_spec), &
                                    temperatures(i_str), &
                                    mean_molar_masses(i_str), &
                                    vel(i_rad) &
                                )
                            end do
                        else
                            do i_rad = 1, N_fit
                                rad(i_rad) = &
                                    r_w(i_str,i_spec) &
                                    + (r_w(i_str,i_spec) / clouds_b_hansen(i_str,i_spec) - r_w(i_str,i_spec)) &
                                    * dble(i_rad - 1) / dble(N_fit - 1)

                                call compute_turbulent_settling_speed(&
                                    rad(i_rad), &
                                    surface_gravity, &
                                    rho(i_str), &
                                    clouds_particles_densities(i_spec), &
                                    temperatures(i_str), &
                                    mean_molar_masses(i_str), &
                                    vel(i_rad) &
                                )
                            end do
                        end if

                        call  linear_fit(log(rad), log(vel / w_star(i_str)), N_fit, a, b)

                        alpha(i_str, i_spec) = b
                        r_w(i_str,i_spec) = exp(-a / b)

                        if (x_gamma(i_str, i_spec) + alpha(i_str, i_spec) < x_gamma_max) then
                            clouds_a_hansen(i_str, i_spec) = &
                                (&
                                    clouds_b_hansen(i_str, i_spec) ** (-1d0 * alpha(i_str, i_spec)) &
                                    * r_w(i_str, i_spec) ** alpha(i_str, i_spec) * f_seds(i_spec) &
                                    * (&
                                        clouds_b_hansen(i_str, i_spec) ** 3d0 &
                                        * clouds_b_hansen(i_str, i_spec) ** alpha(i_str, i_spec) &
                                        - clouds_b_hansen(i_str,i_spec) + 1d0 &
                                    )&
                                    * gamma(x_gamma(i_str, i_spec)) &
                                    / (&
                                        (&
                                            clouds_b_hansen(i_str,i_spec) * alpha(i_str,i_spec) &
                                            + 2d0 * clouds_b_hansen(i_str, i_spec) + 1d0 &
                                        )&
                                        * gamma(x_gamma(i_str, i_spec) + alpha(i_str, i_spec))&
                                    )&
                                ) ** (1d0 / alpha(i_str, i_spec))
                        else  ! to avoid overflow, approxiate gamma(x) / gamma(x + a) by x ** -a (from Stirling formula)
                            clouds_a_hansen(i_str, i_spec) = &
                                (&
                                    clouds_b_hansen(i_str, i_spec) ** (-1d0 * alpha(i_str, i_spec)) &
                                    * r_w(i_str, i_spec) ** alpha(i_str, i_spec) * f_seds(i_spec) &
                                    * (&
                                        clouds_b_hansen(i_str, i_spec) ** 3d0 &
                                        * clouds_b_hansen(i_str, i_spec) ** alpha(i_str, i_spec) &
                                        - clouds_b_hansen(i_str,i_spec) + 1d0 &
                                    )&
                                    * x_gamma(i_str, i_spec) ** (-alpha(i_str, i_spec)) &
                                    / (&
                                        (&
                                            clouds_b_hansen(i_str,i_spec) * alpha(i_str,i_spec) &
                                            + 2d0 * clouds_b_hansen(i_str, i_spec) + 1d0 &
                                        )&
                                    )&
                                ) ** (1d0 / alpha(i_str, i_spec))
                        end if
                    else
                        clouds_a_hansen(i_str, i_spec) = 1d-17
                        alpha(i_str, i_spec) = 1d0
                    end if
                end do
            end do
        end subroutine compute_cloud_particles_mean_radius_hansen


        subroutine compute_feautrier_radiative_transfer(frequencies_bin_edges, optical_depths, temperatures, &
                                                        emission_cos_angles, emission_cos_angles_weights, &
                                                        weights_gauss, photon_destruction_probabilities, &
                                                        return_contribution, reflectances, emissivities, &
                                                        stellar_intensity, emission_geometry, &
                                                        star_irradiation_cos_angle, &
                                                        n_frequencies_bin_edges, n_layers, n_angles, n_g, &
                                                        flux, emission_contribution)
            use math, only: solve_tridiagonal_system, cst_pi
            use physics, only: compute_planck_function_integral

            implicit none

            integer, parameter :: iter_scat = 1000
            double precision, parameter :: tiniest = tiny(0d0), pi_4 = 4d0 * cst_pi

            integer, intent(in)             :: n_frequencies_bin_edges, n_layers, n_angles, n_g
            double precision, intent(in)    :: star_irradiation_cos_angle
            double precision, intent(in)    :: &
                reflectances(n_frequencies_bin_edges-1), emissivities(n_frequencies_bin_edges-1) !ELALEI
            double precision, intent(in)    :: stellar_intensity(n_frequencies_bin_edges-1) !ELALEI
            double precision, intent(in)    :: frequencies_bin_edges(n_frequencies_bin_edges)
            double precision, intent(in)    :: optical_depths(n_g,n_frequencies_bin_edges-1,n_layers)
            double precision, intent(in)    :: temperatures(n_layers)
            double precision, intent(in)    :: emission_cos_angles(n_angles)
            double precision, intent(in)    :: emission_cos_angles_weights(n_angles), weights_gauss(n_g)
            double precision, intent(in)    :: photon_destruction_probabilities(n_g,n_frequencies_bin_edges-1,n_layers)
            logical, intent(in)             :: return_contribution
            double precision, intent(out)   :: flux(n_frequencies_bin_edges-1)
            double precision, intent(out)   :: emission_contribution(n_layers,n_frequencies_bin_edges-1)
            character(len=20), intent(in)   :: emission_geometry

            integer                         :: j,i,k,l
            double precision                :: I_J(n_layers,n_angles), I_H(n_layers,n_angles)
            double precision                :: source(n_layers, n_g, n_frequencies_bin_edges - 1)
            double precision                :: source_tmp(n_g, n_frequencies_bin_edges - 1, n_layers), &
                J_planet_scat(n_layers, n_g, n_frequencies_bin_edges - 1), &
                photon_destruction_probabilities_(n_layers, n_g, n_frequencies_bin_edges-1), &
                source_planet_scat_n(n_g,n_frequencies_bin_edges-1,n_layers), &
                source_planet_scat_n1(n_g,n_frequencies_bin_edges-1,n_layers), &
                source_planet_scat_n2(n_g,n_frequencies_bin_edges-1,n_layers), &
                source_planet_scat_n3(n_g,n_frequencies_bin_edges-1,n_layers)
            double precision                :: J_star_ini(n_g,n_frequencies_bin_edges-1,n_layers)
            double precision                :: I_star_calc(n_g,n_angles,n_layers,n_frequencies_bin_edges-1)
            double precision                :: flux_old(n_frequencies_bin_edges-1), conv_val
            double precision                :: I_surface_reflection
            double precision                :: I_surface_no_scattering(n_g, n_frequencies_bin_edges - 1)
            double precision                :: I_surface_emission
            double precision                :: surf_refl_2(n_frequencies_bin_edges - 1)
            double precision                :: mu_weight(n_angles)
            ! tridag variables
            double precision                :: a(n_layers, n_angles, n_g, n_frequencies_bin_edges - 1),&
                b(n_layers, n_angles, n_g, n_frequencies_bin_edges - 1),&
                c(n_layers, n_angles, n_g, n_frequencies_bin_edges - 1),&
                r(n_layers, n_frequencies_bin_edges - 1), &
                planck(n_layers, n_frequencies_bin_edges - 1)
            double precision                :: f1,f2,f3, deriv1, deriv2, I_minus
            double precision                :: f2_struct(n_layers, n_angles, n_g, n_frequencies_bin_edges - 1),&
                                               f3_struct(n_layers, n_angles, n_g, n_frequencies_bin_edges - 1)

            ! quantities for P-T structure iteration
            double precision                :: J_bol(n_layers)
            double precision                :: J_bol_a(n_layers)
            double precision                :: J_bol_g(n_layers)

            ! ALI
            double precision                :: lambda_loc(n_layers, n_g, n_frequencies_bin_edges - 1)

            ! control
            double precision                :: inv_del_tau_min, inv_del_tau_min_half
            integer                         :: i_iter_scat

            ! GCM species calc
            logical                         :: GCM_read
            double precision                :: I_GCM(n_angles,n_frequencies_bin_edges-1)

            ! Variables for the contribution function calculation
            integer :: i_mu, i_str, i_freq
            double precision :: transm_mu(n_g,n_frequencies_bin_edges-1,n_layers), &
                         transm_all(n_frequencies_bin_edges-1,n_layers), transm_all_loc(n_layers)

            ! Variables for surface scattering
            double precision                :: I_plus_surface(n_angles, n_g, n_frequencies_bin_edges-1), &
                                               mu_double(n_angles)
!            double precision                :: t0,tf, t1, ti, ttri,tder,ttot,ti2,tstuff,t2,t3,tx
! TODO clean debug
!call cpu_time(t0)
!print*,'fixed'
            I_plus_surface = 0d0
            I_minus = 0d0

            GCM_read = .FALSE.
            source_tmp = 0d0
            source = 0d0
            flux_old = 0d0
            flux = 0d0

            source_planet_scat_n = 0d0
            source_planet_scat_n1 = 0d0
            source_planet_scat_n2 = 0d0
            source_planet_scat_n3 = 0d0

            ! Optimize shape for loops
            photon_destruction_probabilities_ = &
                reshape(photon_destruction_probabilities, shape(photon_destruction_probabilities_), order=[2, 3, 1])

            ! DO THE STELLAR ATTENUATION CALCULATION
            J_star_ini = 0d0

            f2_struct = 0d0
            f3_struct = 0d0
            r = 0d0

            do i = 1, n_frequencies_bin_edges-1
                ! Irradiation treatment
                ! Dayside ave: multiply flux by 1/2.
                ! Planet ave: multiply flux by 1/4
                do i_mu = 1, n_angles
                    if (trim(adjustl(emission_geometry)) == 'dayside_ave') then
                        I_star_calc(:,i_mu,:,i) = 0.5* abs(stellar_intensity(i)) &
                            * exp(-optical_depths(:,i,:)/emission_cos_angles(i_mu))
                        J_star_ini(:,i,:) = J_star_ini(:,i,:) &
                            + 0.5d0*I_star_calc(:,i_mu,:,i)*emission_cos_angles_weights(i_mu)
                    else if (trim(adjustl(emission_geometry)) == 'planetary_ave') then
                        I_star_calc(:,i_mu,:,i) = 0.25* abs(stellar_intensity(i)) &
                            * exp(-optical_depths(:,i,:)/emission_cos_angles(i_mu))
                        J_star_ini(:,i,:) = J_star_ini(:,i,:) &
                            +0.5d0*I_star_calc(:,i_mu,:,i)*emission_cos_angles_weights(i_mu)
                    else if (trim(adjustl(emission_geometry)) == 'non-isotropic') then
                        J_star_ini(:,i,:) = abs(&
                            stellar_intensity(i)/4.*exp(-optical_depths(:,i,:)/star_irradiation_cos_angle)&
                        )
                    else
                        write(*,*) 'Invalid geometry'
                    end if
                end do
            end do

            inv_del_tau_min = 1d10
            inv_del_tau_min_half = inv_del_tau_min * 0.5d0

            mu_double(:) = emission_cos_angles(:) * 2d0

            ! Initialize the parameters that will be constant through the iterations
            call init_iteration_parameters()

            surf_refl_2 = 2d0 * reflectances
            mu_weight = emission_cos_angles * emission_cos_angles_weights

            do i = 1, n_frequencies_bin_edges - 1
                I_surface_emission = emissivities(i) * planck(n_layers, i)

                do l = 1, n_g
                    if  (trim(adjustl(emission_geometry)) /= 'non-isotropic') then
                        I_surface_reflection = surf_refl_2(i) &
                            * sum(I_star_calc(l, :, n_layers, i) * mu_weight)
                    else
                        I_surface_reflection = reflectances(i) &
                            * J_star_ini(l, i, n_layers) * 4d0 * star_irradiation_cos_angle
                    end if

                    I_surface_no_scattering(l, i) = I_surface_emission + I_surface_reflection
                end do
            end do
!call cpu_time(tf)
!print*, 'init: ', tf-t0
            do i_iter_scat = 1, iter_scat
!print*,i_iter_scat

                flux_old = flux
                J_planet_scat = 0d0

                J_bol(1) = 0d0
                I_GCM = 0d0
!ti=0d0
!ttri = 0d0
!tder = 0d0
!ti2 = 0d0
!tstuff = 0d0
!t1 = 0d0
!t2 = 0d0
!t3 = 0d0
!call cpu_time(ttot)
                do i = 1, n_frequencies_bin_edges - 1
                    flux(i) = 0d0
                    J_bol_a = 0d0

                    r(:, i) = planck(:, i)

                    do l = 1, n_g
                        if (i_iter_scat == 1) then
                            source(:, l, i) = photon_destruction_probabilities_(:, l, i) * r(:, i) &
                                + (1d0 - photon_destruction_probabilities_(:, l, i)) * J_star_ini(l, i, :)
                        else
                            r(:, i) = source(:, l, i)
                        end if

                        do j = 1, n_angles
!call cpu_time(t0)
                            ! r(n_layers) = I_J(n_layers) = 0.5[I_plus + I_minus]
                            ! where I_plus is the light that goes downwards and
                            ! I_minus is the light that goes upwards.
                            !!!!!!!!!!!!!!!!!! ALWAYS NEEDED !!!!!!!!!!!!!!!!!!
                            !I_plus = I_plus_surface(j, l, i)

                            !!!!!!!!!!!!!!! EMISSION ONLY TERM !!!!!!!!!!!!!!!!
                            I_minus = I_surface_no_scattering(l, i) &
                            !!!!!!!!!!!!!!! SURFACE SCATTERING !!!!!!!!!!!!!!!!
                            ! ----> of the emitted/scattered atmospheric light
                            ! + reflectances(i) * sum(I_plus_surface(:, l, i) * emission_cos_angles_weights) ! OLD PRE 091220
                                + surf_refl_2(i) * sum(I_plus_surface(:, l, i) * mu_weight) !&

                            !sum to get I_J
                            r(n_layers, i) = 0.5d0 * (I_plus_surface(j, l, i) + I_minus)
!call cpu_time(tf)
!ti = ti + tf - t0
!call cpu_time(t0)

                            call solve_tridiagonal_system(&
                                a(:, j, l, i), b(:, j, l, i), c(:, j, l, i), r(:, i), n_layers, I_J(:,j) &
                            )
!call cpu_time(tf)
!ttri = ttri + tf - t0
!call cpu_time(t0)
                            I_H(1, j) = -I_J(1, j)

                            do k = 2, n_layers - 1
                                deriv1 = f2_struct(k, j, l, i) * (I_J(k + 1, j) - I_J(k, j))
                                deriv2 = f3_struct(k, j, l, i) * (I_J(k, j) - I_J(k - 1, j))
                                I_H(k, j) = -(deriv1 + deriv2) * 0.5d0

                                ! TEST PAUL SCAT
                                if (k == n_layers - 1) then
                                    I_plus_surface(j, l, i) = I_J(n_layers, j) - deriv1
                                end if
                                ! END TEST PAUL SCAT
                            end do

                            I_H(n_layers, j) = 0d0
!call cpu_time(tf)
!tder = tder + tf - t0
                        end do  ! emission_cos_angles
!call cpu_time(t0)
                        J_bol_g = 0d0

                        do j = 1, n_angles
                            J_bol_g = J_bol_g + I_J(:, j) * emission_cos_angles_weights(j)
                            flux(i) = flux(i) - I_H(1, j) * pi_4 * weights_gauss(l) * mu_weight(j)
                        end do

                        ! Save angle-dependent surface flux
                        if (GCM_read) then
                            do j = 1, n_angles
                                I_GCM(j,i) = I_GCM(j,i) - 2d0 * I_H(1, j) * weights_gauss(l)
                            end do
                        end if

                        J_planet_scat(:, l, i) = J_bol_g
!call cpu_time(tf)
!tstuff = tstuff + tf - t0
                    end do  ! g
                end do  ! frequencies
!call cpu_time(tf)
!ttot = tf - ttot
!print*,'res3',ti,ttri,tder,ti2,tstuff,ttot,ti+ttri+tder+ti2+tstuff
!print*,'ti',ti,t1,t2,t3,t1+t2+t3
                do i = 1, n_frequencies_bin_edges-1
                    do l = 1, n_g
                        do k = 1, n_layers
                            if (photon_destruction_probabilities_(k, l, i) < 1d-10) then
                                photon_destruction_probabilities_(k, l, i) = 1d-10
                            end if
                        end do
                    end do
                end do

                call compute_planck_function_integral(&
                    temperatures, frequencies_bin_edges, n_layers, n_frequencies_bin_edges, r &
                )

                do i = 1, n_frequencies_bin_edges-1
                    do l = 1, n_g
                        source(:, l, i) = (&
                                photon_destruction_probabilities_(:, l, i) * r(:, i) &
                                + (1d0 - photon_destruction_probabilities_(:, l, i)) &
                                * (J_star_ini(l, i, :)+J_planet_scat(:, l, i)-lambda_loc(:, l, i) * source(:, l, i))&
                            ) / &
                            (1d0 - (1d0 - photon_destruction_probabilities_(:, l, i)) * lambda_loc(:, l, i))
                    end do
                end do

                source_planet_scat_n3 = source_planet_scat_n2
                source_planet_scat_n2 = source_planet_scat_n1
                source_planet_scat_n1 = source_planet_scat_n
                source_tmp = reshape(source, shape=shape(source_tmp), order=[3, 1, 2])
                source_planet_scat_n  = source_tmp

                if (mod(i_iter_scat, 4) == 0) then
                    !write(*,*) 'Ng acceleration!'

                    call m_approximate_ng_source(source_planet_scat_n,source_planet_scat_n1, &
                        source_planet_scat_n2,source_planet_scat_n3,source_tmp, &
                        n_g,n_frequencies_bin_edges,n_layers)

                    source = reshape(source_tmp, shape=shape(source), order=[2, 3, 1])
                end if

                ! Test if the flux has converged
                conv_val = maxval(abs((flux - flux_old) / flux))

                if ((conv_val < 1d-3) .and. (i_iter_scat > 9)) then
                    exit
                end if
            end do  ! iterations

            ! Calculate the contribution function.
            ! Copied from compute_ck_flux, here using "source" as the source function
            ! (before it was the Planck function).

            emission_contribution = 0d0

            if (return_contribution) then
                do i_mu = 1, n_angles
                    ! Transmissions for a given incidence angle
                    transm_mu = exp(-optical_depths/emission_cos_angles(i_mu))

                    do i_str = 1, n_layers
                        do i_freq = 1, n_frequencies_bin_edges-1
                            ! Integrate transmission over g-space
                            transm_all(i_freq,i_str) = sum(transm_mu(:,i_freq,i_str)*weights_gauss)
                        end do
                    end do

                    ! Do the actual radiative transport
                    do i_freq = 1, n_frequencies_bin_edges-1
                        ! Spatial transmissions at given wavelength
                        transm_all_loc = transm_all(i_freq,:)
                        ! Calc Eq. 9 of manuscript (em_deriv.pdf)
                        do i_str = 1, n_layers
                            r(i_str, i_freq) = sum(source(i_str,:,i_freq)*weights_gauss)
                        end do

                        do i_str = 1, n_layers-1
                            emission_contribution(i_str,i_freq) = emission_contribution(i_str, i_freq)&
                                + (r(i_str, i_freq) + r(i_str + 1, i_freq)) &
                                * (transm_all_loc(i_str) - transm_all_loc(i_str + 1)) &
                                * emission_cos_angles(i_mu) * emission_cos_angles_weights(i_mu)
                        end do

                        emission_contribution(n_layers,i_freq) = emission_contribution(n_layers,i_freq)+ &

                        2d0*I_minus*transm_all_loc(n_layers)*emission_cos_angles(i_mu)*emission_cos_angles_weights(i_mu)
                    end do
                end do

                do i_freq = 1, n_frequencies_bin_edges-1
                    emission_contribution(:,i_freq) = &
                        emission_contribution(:,i_freq)/sum(emission_contribution(:,i_freq))
                end do
            end if

            contains
                subroutine init_iteration_parameters()
                    implicit none

                    double precision :: &
                        dtau_1_level(n_layers - 1), &
                        dtau_2_level(n_layers - 2)

                    b(n_layers, :, :, :) = 1d0
                    c(n_layers, :, :, :) = 0d0
                    a(n_layers, :, :, :) = 0d0

                    lambda_loc = 0d0

                    call compute_planck_function_integral(&
                        temperatures, frequencies_bin_edges, n_layers, n_frequencies_bin_edges, planck &
                    )

                    do i = 1, n_frequencies_bin_edges - 1
                        do l = 1, n_g
                            dtau_1_level(1) = optical_depths(l, i, 2) - optical_depths(l, i, 1)

                            if(abs(dtau_1_level(1)) < tiniest) then
                                dtau_1_level(1) = tiniest
                            end if

                            do k = 2, n_layers - 1
                                dtau_1_level(k) = optical_depths(l, i, k + 1) - optical_depths(l, i, k)

                                if(abs(dtau_1_level(k)) < tiniest) then
                                    dtau_1_level(k) = tiniest
                                end if

                                dtau_2_level(k - 1) = optical_depths(l, i, k + 1) - optical_depths(l, i, k - 1)

                               if(abs(dtau_2_level(k - 1)) < tiniest) then
                                    dtau_2_level(k - 1) = tiniest
                               end if
                            end do

!if(i == 800 .and. l == 8) print*,dtau_1_level(50),dtau_2_level(50)

                            do j = 1, n_angles
                                ! Own boundary treatment
                                ! Frist level (top)
                                f1 = emission_cos_angles(j) / dtau_1_level(1)

                                ! own test against instability
                                if (f1 > inv_del_tau_min) then
                                    f1 = inv_del_tau_min
                                end if

                                b(1, j, l, i) = 1d0 + 2d0 * f1 * (1d0 + f1)
                                c(1, j, l, i) = -2d0 * f1 ** 2d0
                                a(1, j, l, i) = 0d0

                                ! Calculate the local approximate lambda iterator
                                lambda_loc(1, l, i) = &
                                    lambda_loc(1, l, i) + emission_cos_angles_weights(j) / (1d0 + 2d0 * f1 * (1d0 + f1))

                                ! Mid levels
                                do k = 2, n_layers - 1
                                    f1 = mu_double(j) / dtau_2_level(k - 1)
                                    f2 = emission_cos_angles(j) / dtau_1_level(k)
                                    f3 = emission_cos_angles(j) / dtau_1_level(k - 1)

                                    ! own test against instability
                                    if (f1 > inv_del_tau_min_half) then
                                        f1 = inv_del_tau_min_half
                                    end if

                                    if (f2 > inv_del_tau_min) then
                                        f2 = inv_del_tau_min
                                    end if

                                    if (f3 > inv_del_tau_min) then
                                            f3 = inv_del_tau_min
                                    end if

                                    b(k, j, l, i) = 1d0 + f1 * (f2 + f3)
                                    c(k, j, l, i) = -f1 * f2
                                    a(k, j, l, i) = -f1 * f3

                                    f2_struct(k, j, l, i) = f2
                                    f3_struct(k, j, l, i) = f3

                                    ! Calculate the local approximate lambda iterator
                                    lambda_loc(k, l, i) = lambda_loc(k, l, i) &
                                        + emission_cos_angles_weights(j) / (1d0 + f1 * (f2 + f3))
                                end do

                                ! Own boundary treatment
                                ! Last level (surface)
                                f1 = emission_cos_angles(j) / dtau_1_level(n_layers - 1)

                                if (f1 > inv_del_tau_min) then
                                    f1 = inv_del_tau_min
                                end if

                                lambda_loc(n_layers, l, i) = &
                                    lambda_loc(n_layers, l, i) &
                                    + emission_cos_angles_weights(j) / (1d0 + 2d0 * f1 ** 2d0)
                            end do
                        end do
                    end do
                end subroutine init_iteration_parameters

                subroutine m_approximate_ng_source(source_n,source_n1,source_n2,source_n3,source, &
                                         n_g,n_frequencies_bin_edges,n_layers)
                    implicit none

                    double precision, parameter :: huge_q = sqrt(huge(0d0)), tiniest = tiny(0d0)

                    integer :: n_layers, n_frequencies_bin_edges, n_g, i_ng, i_freq
                    double precision :: tn(n_layers), tn1(n_layers), tn2(n_layers), &
                            tn3(n_layers), temp_buff(n_layers), &
                            source_n(n_g,n_frequencies_bin_edges-1,n_layers), &
                            source_n1(n_g,n_frequencies_bin_edges-1,n_layers), &
                            source_n2(n_g,n_frequencies_bin_edges-1,n_layers), &
                            source_n3(n_g,n_frequencies_bin_edges-1,n_layers), &
                            source(n_g,n_frequencies_bin_edges-1,n_layers), &
                            source_buff(n_g,n_frequencies_bin_edges-1,n_layers)
                    double precision :: Q1(n_layers), Q2(n_layers), Q3(n_layers)
                    double precision :: A1, A2, B1, B2, C1, C2, AB_denominator
                    double precision :: a, b

                    do i_freq = 1, n_frequencies_bin_edges-1
                        do i_ng = 1, n_g
                            tn = source_n(i_ng,i_freq,1:n_layers)
                            tn1 = source_n1(i_ng,i_freq,1:n_layers)
                            tn2 = source_n2(i_ng,i_freq,1:n_layers)
                            tn3 = source_n3(i_ng,i_freq,1:n_layers)

                            Q1 = tn - 2d0 * tn1 + tn2
                            Q2 = tn - tn1 - tn2 + tn3
                            Q3 = tn - tn1

                            ! test
                            Q1(1) = 0d0
                            Q2(1) = 0d0
                            Q3(1) = 0d0

                            A1 = min(sum(Q1 * Q1), huge_q)
                            A2 = min(sum(Q2 * Q1), huge_q)
                            B1 = min(sum(Q1 * Q2), huge_q)
                            B2 = min(sum(Q2 * Q2), huge_q)
                            C1 = min(sum(Q1 * Q3), huge_q)
                            C2 = min(sum(Q2 * Q3), huge_q)

                            AB_denominator = A1 * B2 - A2 * B1

                            ! Nan handling
                            if(AB_denominator < tiniest) then
                                return
                            end if

                            if (abs(A1) >= 1d-250 .and. &
                                    abs(A2) >= 1d-250 .and. &
                                    abs(B1) >= 1d-250 .and. &
                                    abs(B2) >= 1d-250 .and. &
                                    abs(C1) >= 1d-250 .and. &
                                    abs(C2) >= 1d-250) then
                                a = (C1 * B2 - C2 * B1) / AB_denominator
                                b = (C2 * A1 - C1 * A2) / AB_denominator

                                temp_buff = max((1d0 - a - b) * tn + a * tn1 + b * tn2, 0d0)

                                source_buff(i_ng,i_freq,1:n_layers) = temp_buff
                            else
                                source_buff(i_ng,i_freq,1:n_layers) = source(i_ng,i_freq,1:n_layers)
                            end if
                        end do
                    end do

                    source = source_buff
                end subroutine m_approximate_ng_source
        end subroutine compute_feautrier_radiative_transfer


        subroutine compute_optical_depths(surface_gravity, pressures, opacities, scattering_in_emission, &
                                          continuum_opacities_scattering, n_layers, n_frequencies, n_g, &
                                          optical_depths, photon_destruction_probabilities)
            ! """
            ! Calculate tau_scat with 2nd order accuracy.
            ! """
            implicit none
            
            ! I/O
            integer, parameter                           :: n_species = 1
            integer, intent(in)                          :: n_layers, n_frequencies, n_g
            double precision, intent(in)                 :: opacities(n_g,n_frequencies,n_species,n_layers)
            double precision, intent(in)                 :: surface_gravity, pressures(n_layers)
            logical, intent(in)                          :: scattering_in_emission
            double precision, intent(in)                 :: continuum_opacities_scattering(n_frequencies,n_layers)
            double precision, intent(out)                :: optical_depths(n_g,n_frequencies,n_species,n_layers), &
               photon_destruction_probabilities(n_g,n_frequencies,n_layers)
            ! internal
            integer                                      :: i_struc, i_freq, i_g, i_spec
            double precision                             :: del_tau_lower_ord, &
               gamma_second(n_g,n_frequencies,n_species), f_second, kappa_i(n_g,n_frequencies,n_species), &
               kappa_im(n_g,n_frequencies,n_species), kappa_ip(n_g,n_frequencies,n_species)
            double precision                             :: opacities_(n_g,n_frequencies,n_species,n_layers)
            logical                                      :: second_order
            
            optical_depths = 0d0
            second_order = .FALSE.
            
            opacities_ = opacities
            
            if (scattering_in_emission) then
                do i_g = 1, n_g
                    opacities_(i_g,:,1,:) = opacities_(i_g,:,1,:) &
                        + continuum_opacities_scattering(:,:)
                    photon_destruction_probabilities(i_g,:,:) = &
                        continuum_opacities_scattering(:,:) / opacities_(i_g,:,1,:)
                end do
                
                photon_destruction_probabilities = 1d0 - photon_destruction_probabilities
            else
                photon_destruction_probabilities = 1d0
            end if
            
            if (second_order) then
                do i_struc = 2, n_layers
                    if (i_struc == n_layers) then
                        optical_depths(:,:,:,i_struc) = optical_depths(:,:,:,i_struc-1) &
                            + (opacities_(:,:,:,i_struc)+opacities_(:,:,:,i_struc-1)) &
                            /2d0/surface_gravity*(pressures(i_struc)-pressures(i_struc-1))
                    else
                        f_second = (pressures(i_struc+1)-pressures(i_struc))/(pressures(i_struc)-pressures(i_struc-1))
                        kappa_i = opacities_(:,:,:,i_struc)
                        kappa_im = opacities_(:,:,:,i_struc-1)
                        kappa_ip = opacities_(:,:,:,i_struc+1)
                        gamma_second = (kappa_ip-(1d0+f_second)*kappa_i+f_second*kappa_im) &
                            / (f_second*(1d0+f_second))
                        optical_depths(:,:,:,i_struc) = optical_depths(:,:,:,i_struc-1) &
                            + ((kappa_i+kappa_im)/2d0-gamma_second/6d0) &
                            /surface_gravity*(pressures(i_struc)-pressures(i_struc-1))
                        
                        do i_spec = 1, n_species
                            do i_freq = 1, n_frequencies
                            do i_g = 1, n_g
                                if (optical_depths(i_g,i_freq,i_spec,i_struc) &
                                        < optical_depths(i_g,i_freq,i_spec,i_struc-1)) then
                                    if (i_struc <= 2) then
                                        optical_depths(i_g,i_freq,i_spec,i_struc) = &
                                            optical_depths(i_g,i_freq,i_spec,i_struc-1)*1.01d0
                                    else
                                        optical_depths(i_g,i_freq,i_spec,i_struc) = &
                                            optical_depths(i_g,i_freq,i_spec,i_struc-1) &
                                            + (optical_depths(i_g,i_freq,i_spec,i_struc-1) &
                                            - optical_depths(i_g,i_freq,i_spec,i_struc-2))*0.01d0
                                    end if
                                end if
                                
                                del_tau_lower_ord = (kappa_i(i_g,i_freq,i_spec)+ &
                                kappa_im(i_g,i_freq,i_spec))/2d0/surface_gravity* &
                                (pressures(i_struc)-pressures(i_struc-1))
                                
                                if ((optical_depths(i_g,i_freq,i_spec,i_struc) - &
                                        optical_depths(i_g,i_freq,i_spec,i_struc-1)) > del_tau_lower_ord) then
                                    optical_depths(i_g,i_freq,i_spec,i_struc) = &
                                    optical_depths(i_g,i_freq,i_spec,i_struc-1) + del_tau_lower_ord
                                end if
                            end do
                            end do
                        end do
                    end if
                end do
            else
                do i_struc = 2, n_layers
                    optical_depths(:,:,:,i_struc) = optical_depths(:,:,:,i_struc-1) + &
                         (opacities_(:,:,:,i_struc)+opacities_(:,:,:,i_struc-1)) &
                         /2d0/surface_gravity*(pressures(i_struc)-pressures(i_struc-1))
                end do
            end if
        end subroutine compute_optical_depths


        subroutine compute_planck_opacities(opacities, temperatures, weights_gauss, frequencies_bin_edges, &
                                            scattering_in_emission, continuum_opacities_scattering, &
                                            n_g, n_frequencies, n_layers, n_frequencies_bin_edges, &
                                            opacities_planck)

            implicit none

            integer,          intent(in)  :: n_g, n_frequencies, n_layers, n_frequencies_bin_edges
            double precision, intent(in)  :: opacities(n_g, n_frequencies, n_layers)
            double precision, intent(in)  :: frequencies_bin_edges(n_frequencies_bin_edges)
            double precision, intent(in)  :: temperatures(n_layers), weights_gauss(n_g)
            logical, intent(in)           :: scattering_in_emission
            double precision, intent(in)  :: continuum_opacities_scattering(n_frequencies,n_layers)
            double precision, intent(out) :: opacities_planck(n_layers)

            double precision              :: total_kappa_use(n_g, n_frequencies, n_layers)

            integer                       :: i_struc, i_g

            if (scattering_in_emission) then
                do i_g = 1, n_g
                    total_kappa_use(i_g,:,:) = opacities(i_g,:,:) + continuum_opacities_scattering
                end do
            else
                total_kappa_use = opacities
            end if

            do i_struc = 1, n_layers
                call compute_planck_opacities_(&
                    total_kappa_use(:,:,i_struc), &
                    frequencies_bin_edges, &
                    temperatures(i_struc), &
                    n_g, &
                    n_frequencies+1, &
                    opacities_planck(i_struc), &
                    weights_gauss &
                )
            end do

            contains
                subroutine compute_planck_opacities_(clouds_final_absorption_opacities, frequencies_bin_edges, &
                                                     temperatures, n_g, n_frequencies_bins, &
                                                     opacities_planck, weights_gauss)
                    use physics, only: compute_star_planck_function_integral

                    implicit none

                    integer                         :: n_g,n_frequencies_bins
                    double precision                :: frequencies_bin_edges(n_frequencies_bins)
                    double precision                :: clouds_final_absorption_opacities(n_g,n_frequencies_bins-1)
                    double precision                :: temperatures, opacities_planck, weights_gauss(n_g), &
                    planck_flux(n_frequencies_bins-1), norm

                    integer                         :: i

                    call compute_star_planck_function_integral(n_frequencies_bins,temperatures,frequencies_bin_edges,&
                                                            planck_flux)

                    opacities_planck = 0d0
                    norm = 0d0

                    do i = 1, n_frequencies_bins-1
                        opacities_planck = opacities_planck + &
                        planck_flux(i) * sum(clouds_final_absorption_opacities(:,i)*weights_gauss) * &
                        (frequencies_bin_edges(i)-frequencies_bin_edges(i+1))
                        norm = norm + &
                        planck_flux(i) * (frequencies_bin_edges(i)-frequencies_bin_edges(i+1))
                    end do

                    opacities_planck = opacities_planck / norm

                end subroutine compute_planck_opacities_
        end subroutine compute_planck_opacities


        subroutine compute_radius_hydrostatic_equilibrium(pressures, surface_gravity, atmosphere_densities, &
                                                          reference_pressure, planet_radius, variable_gravity, &
                                                          n_layers, &
                                                          radius_hydrostatic_equilibrium)
            ! """
            ! Calculate the radius at hydrostatic equilibrium from the pressure grid.
            ! """
            implicit none
            ! I/O
            integer, intent(in)                         :: n_layers
            double precision, intent(in)                :: reference_pressure
            double precision, intent(in)                :: pressures(n_layers), atmosphere_densities(n_layers)
            double precision, intent(in)                :: surface_gravity, planet_radius
            logical, intent(in)                         :: variable_gravity
            double precision, intent(out)               :: radius_hydrostatic_equilibrium(n_layers)

            ! Internal
            integer                                     :: i_str
            double precision                            :: R0, inv_rho(n_layers), coefficient

            inv_rho = 1d0 / atmosphere_densities

            radius_hydrostatic_equilibrium = 0d0
            R0 = 0d0

            if (variable_gravity) then
                coefficient = 1d0 / surface_gravity / planet_radius ** 2d0
            else
                coefficient = -1d0 / surface_gravity
            end if

            ! Calculate radius with vertically varying gravity, set up such that at P=P0, i.e. R=planet_radius
            ! the planet has the predefined scalar gravity value
            radius_hydrostatic_equilibrium(n_layers - 1) = radius_hydrostatic_equilibrium(n_layers) &
                - (inv_rho(n_layers - 1) + inv_rho(n_layers)) &
                * (pressures(n_layers) - pressures(n_layers - 1)) * coefficient * 0.5d0

            do i_str = n_layers - 2, 1, -1
                radius_hydrostatic_equilibrium(i_str) = radius_hydrostatic_equilibrium(i_str + 1) - integ_parab(&
                    pressures(i_str), &
                    pressures(i_str + 1), &
                    pressures(i_str + 2), &
                    inv_rho(i_str), &
                    inv_rho(i_str + 1), &
                    inv_rho(i_str + 2), &
                    pressures(i_str), pressures(i_str+1) &
                ) * coefficient
            end do

            ! Find R0
            do i_str = n_layers - 1, 1, -1
                if ((pressures(i_str + 1) > reference_pressure) .and. (pressures(i_str) <= reference_pressure)) then
                    if (i_str <= n_layers - 2) then
                        R0 = radius_hydrostatic_equilibrium(i_str + 1) - integ_parab(&
                            pressures(i_str), &
                            pressures(i_str + 1), &
                            pressures(i_str + 2), &
                            inv_rho(i_str), &
                            inv_rho(i_str + 1), &
                            inv_rho(i_str + 2), &
                            reference_pressure, &
                            pressures(i_str + 1) &
                        ) * coefficient

                        exit
                    else
                        R0 = radius_hydrostatic_equilibrium(i_str + 1) - (inv_rho(i_str) + inv_rho(i_str+1)) &
                            * (pressures(i_str + 1) - reference_pressure) * coefficient * 0.5d0

                        exit
                    end if
                end if
            end do

            if (variable_gravity) then
                R0 = 1d0 / planet_radius - R0
                radius_hydrostatic_equilibrium = radius_hydrostatic_equilibrium + R0
                radius_hydrostatic_equilibrium = 1d0 / radius_hydrostatic_equilibrium
            else
                R0 = planet_radius - R0
                radius_hydrostatic_equilibrium = radius_hydrostatic_equilibrium + R0
            end if

            contains
                function integ_parab(x, y, z, fx, fy, fz, a, b)
                    ! Function to calc higher order integ.
                    implicit none
                    ! I/O
                    double precision :: x, y, z, fx, fy, fz, a, b
                    double precision :: integ_parab
                    ! Internal
                    double precision :: c1, c2, c3

                    c3 = ((fz - fy) / (z - y) - (fz - fx) / (z - x)) / (y - x)
                    c2 = (fz - fx) / (z - x) - c3 * (z + x)
                    c1 = fx - c2 * x - c3 * x ** 2d0

                    integ_parab = c1 * (b - a) + c2 * (b ** 2d0 - a ** 2d0) / 2d0 + c3 * (b ** 3d0 - a ** 3d0) / 3d0
                end function integ_parab
        end subroutine compute_radius_hydrostatic_equilibrium


        subroutine compute_rayleigh_scattering_opacities(species, mass_fractions, wavelengths_angstroem, &
                                                         mean_molar_masses, temperatures, pressures, &
                                                         n_layers, n_frequencies, &
                                                         rayleigh_scattering_opacities)
            ! """
            ! Add Rayleigh scattering.
            ! """
            use math, only: cst_pi
            use physics, only: cst_amu, cst_k, cst_sneep_ubachs_n
            implicit none

            integer, intent(in) :: n_frequencies, n_layers
            character(len=20), intent(in) :: species
            double precision, intent(in) :: wavelengths_angstroem(n_frequencies)
            double precision, intent(in) :: mass_fractions(n_layers), mean_molar_masses(n_layers), &
                temperatures(n_layers), pressures(n_layers)
            double precision, intent(out) :: rayleigh_scattering_opacities(n_frequencies, n_layers)

            double precision, parameter :: abundance_threshold = 1d-60

            integer :: i_str, i_freq
            double precision :: lambda_cm(n_frequencies), lamb_inv(n_frequencies), lamb_inv_use
            double precision :: a0, a1, a2, a3, a4, a5, a6, a7, luv, lir, l(n_frequencies), d(n_layers), &
                temperatures_(n_layers), &
                retVal, retValMin, retValMax, mass_h2o, nm1, fk, scale, mass_co2, mass_o2, mass_n2, A, B, C, mass_co, &
                nfr_co, mass_ch4, nfr_ch4!, rayleigh_kappa_tmp(n_frequencies, n_layers)

            rayleigh_scattering_opacities = 0d0
            !rayleigh_kappa_tmp = 0d0

            if (trim(adjustl(species)) == 'H2') then
                call h2_rayleigh()
            else if (trim(adjustl(species)) == 'He') then
                call he_rayleigh()
            else if (trim(adjustl(species)) == 'H2O') then
             ! For H2O Rayleigh scattering according to Harvey et al. (1998)
             a0 = 0.244257733
             a1 = 9.74634476d-3
             a2 = -3.73234996d-3
             a3 = 2.68678472d-4
             a4 = 1.58920570d-3
             a5 = 2.45934259d-3
             a6 = 0.900704920
             a7 = -1.66626219d-2
             luv = 0.2292020d0
             lir = 5.432937d0
             mass_h2o = 18d0*cst_amu

             lambda_cm = wavelengths_angstroem*1d-8
             lamb_inv = 1d0/lambda_cm

             l = lambda_cm/1d-4/0.589d0
             d = mean_molar_masses*cst_amu*pressures/cst_k/temperatures*mass_fractions
             temperatures_ = temperatures/273.15d0

             do i_str = 1, n_layers
                if (mass_fractions(i_str) > 1d-60) then
                   do i_freq = 1, n_frequencies

                      retVal = (&
                            a0 &
                            + a1 * d(i_str) &
                            + a2 * temperatures_(i_str) &
                            + a3 * l(i_freq) ** 2d0 * temperatures_(i_str) &
                            + a4 / l(i_freq) ** 2d0 &
                            + a5 / (l(i_freq)**2d0-luv**2d0) &
                            + a6/(l(i_freq)**2d0-lir**2d0) &
                            + a7 * d(i_str)**2d0) * d(i_str)

                      retValMin = (&
                          a0 &
                          + a1 * d(i_str) &
                          + a2 * temperatures_(i_str) &
                          + a3 * (0.2d0/0.589d0)**2d0*temperatures_(i_str) &
                          + a4 / (0.2d0/0.589d0)**2d0 &
                          + a5/((0.2d0/0.589d0)**2d0-luv**2d0) &
                          + a6/((0.2d0/0.589d0)**2d0-lir**2d0) &
                          + a7*d(i_str)**2d0 &
                      ) * d(i_str)

                      retValMax = (&
                          a0 &
                          + a1 * d(i_str) &
                          + a2 * temperatures_(i_str) &
                          + a3 * (1.1d0/0.589d0)**2d0*temperatures_(i_str) &
                          + a4 / (1.1d0/0.589d0)**2d0 &
                          + a5 / ((1.1d0/0.589d0)**2d0-luv**2d0) &
                          + a6 / ((1.1d0/0.589d0)**2d0-lir**2d0) &
                          + a7 * d(i_str)**2d0 &
                      )*d(i_str)

                      if ((lambda_cm(i_freq)/1d-4 > 0.2d0) .and. (lambda_cm(i_freq)/1d-4 < 1.1d0)) then
                         nm1 = sqrt((1d0+2d0*retVal)/(1d0-retVal))
                      else if (lambda_cm(i_freq)/1d-4 >= 1.1d0) then
                         nm1 = sqrt((1.+2.*retValMax)/(1.-retValMax))
                      else
                         nm1 = sqrt((1.+2.*retValMin)/(1.-retValMin))
                      end if

                      nm1 = nm1 - 1d0
                      fk = 1.0

                      retVal = 24d0*cst_pi**3d0*lamb_inv(i_freq)**4d0/(d(i_str)/18d0/cst_amu)**2d0* &
                           (((nm1+1d0)**2d0-1d0)/((nm1+1d0)**2d0+2d0))**2d0*fk / mass_h2o * &
                           mass_fractions(i_str)

                      if (.NOT. ISNAN(retVal)) then
                         rayleigh_scattering_opacities(i_freq,i_str) = rayleigh_scattering_opacities(i_freq,i_str) &
                              + retVal
                      end if

                   end do
                end if
             end do
            else if (trim(adjustl(species)) == 'CO2') then

             ! CO2 Rayleigh scattering according to Sneep & Ubachs (2004)
             d = mean_molar_masses*cst_amu*pressures/cst_k/temperatures*mass_fractions

             lambda_cm = wavelengths_angstroem*1d-8
             lamb_inv = 1d0/lambda_cm
             mass_co2 = 44d0*cst_amu

             do i_str = 1, n_layers
                if (mass_fractions(i_str) > 1d-60) then
                   scale = d(i_str)/44d0/cst_amu/cst_sneep_ubachs_n
                   do i_freq = 1, n_frequencies

                      nm1 = 1d-3*1.1427d6*( 5799.25d0/max(20d0**2d0,128908.9d0**2d0-lamb_inv(i_freq)**2d0) + &
                           120.05d0/max(20d0**2d0,89223.8d0**2d0-lamb_inv(i_freq)**2d0) + &
                           5.3334d0/max(20d0**2d0,75037.5d0**2d0-lamb_inv(i_freq)**2d0) + &
                           4.3244/max(20d0**2d0,67837.7d0**2d0-lamb_inv(i_freq)**2d0) + &
                           0.1218145d-4/max(20d0**2d0,2418.136d0**2d0-lamb_inv(i_freq)**2d0))
                      nm1 = nm1 * scale
                      fk = 1.1364+25.3d-12*lamb_inv(i_freq)**2d0
                      rayleigh_scattering_opacities(i_freq,i_str) = rayleigh_scattering_opacities(i_freq,i_str) &
                           + 24d0*cst_pi**3d0*lamb_inv(i_freq)**4d0/(scale*cst_sneep_ubachs_n)**2d0* &
                           (((nm1+1d0)**2d0-1d0)/((nm1+1d0)**2d0+2d0))**2d0*fk / mass_co2 * &
                           mass_fractions(i_str)

                   end do
                end if
             end do
            else if (trim(adjustl(species)) == 'O2') then

             ! O2 Rayleigh scattering according to Thalman et al. (2014).
             ! Also see their erratum!
             d = mean_molar_masses*cst_amu*pressures/cst_k/temperatures*mass_fractions

             lambda_cm = wavelengths_angstroem*1d-8
             lamb_inv = 1d0/lambda_cm
             mass_o2 = 32d0*cst_amu

             do i_str = 1, n_layers
                if (mass_fractions(i_str) > 1d-60) then
                   scale = d(i_str)/mass_o2/2.68678d19

                   do i_freq = 1, n_frequencies

                      if (lamb_inv(i_freq) > 18315d0) then
                         A = 20564.8d0
                         B = 2.480899d13
                      else
                         A = 21351.1d0
                         B = 2.18567d13
                      end if
                      C = 4.09d9

                      nm1 = 1d-8*(A+B/(C-lamb_inv(i_freq)**2d0))
                      nm1 = nm1 !* scale
                      fk = 1.096d0+1.385d-11*lamb_inv(i_freq)**2d0+1.448d-20*lamb_inv(i_freq)**4d0
                      rayleigh_scattering_opacities(i_freq,i_str) = rayleigh_scattering_opacities(i_freq,i_str) &
                           + 24d0*cst_pi**3d0*lamb_inv(i_freq)**4d0/(2.68678d19)**2d0* & !(d(i_str)/mass_o2)**2d0* &
                           (((nm1+1d0)**2d0-1d0)/((nm1+1d0)**2d0+2d0))**2d0*fk / mass_o2 * &
                           mass_fractions(i_str)

                   end do
                end if
             end do
            else if (trim(adjustl(species)) == 'N2') then

             ! N2 Rayleigh scattering according to Thalman et al. (2014).
             ! Also see their erratum!
             d = mean_molar_masses*cst_amu*pressures/cst_k/temperatures*mass_fractions

             lambda_cm = wavelengths_angstroem*1d-8
             lamb_inv = 1d0/lambda_cm
             mass_n2 = 34d0*cst_amu

             do i_str = 1, n_layers
                if (mass_fractions(i_str) > 1d-60) then
                   scale = d(i_str)/mass_n2/2.546899d19

                   do i_freq = 1, n_frequencies

                      if (lamb_inv(i_freq) > 4860d0) then

                         if (lamb_inv(i_freq) > 21360d0) then
                            A = 5677.465d0
                            B = 318.81874d12
                            C = 14.4d9
                         else
                            A = 6498.2d0
                            B = 307.43305d12
                            C = 14.4d9
                         end if

                         nm1 = 1d-8*(A+B/(C-lamb_inv(i_freq)**2d0))
                         nm1 = nm1 !* scale
                         fk = 1.034d0+3.17d-12*lamb_inv(i_freq)**2d0
                         rayleigh_scattering_opacities(i_freq,i_str) = rayleigh_scattering_opacities(i_freq,i_str) &
                              + 24d0*cst_pi**3d0*lamb_inv(i_freq)**4d0/(2.546899d19)**2d0* & !(d(i_str)/mass_n2)**2d0* &
                              (((nm1+1d0)**2d0-1d0)/((nm1+1d0)**2d0+2d0))**2d0*fk / mass_n2 * &
                              mass_fractions(i_str)

                      end if

                   end do
                end if
             end do
            else if (trim(adjustl(species)) == 'CO') then
             ! CO Rayleigh scattering according to Sneep & Ubachs (2004)

             lambda_cm = wavelengths_angstroem*1d-8
             lamb_inv = 1d0/lambda_cm

             d = mean_molar_masses*cst_amu*pressures/cst_k/temperatures*mass_fractions

             do i_str = 1, n_layers

                if (mass_fractions(i_str) > 1d-60) then

                   scale = d(i_str)/28d0/cst_amu/cst_sneep_ubachs_n
                   nfr_co = d(i_str)/28d0/cst_amu
                   mass_co = 28d0*cst_amu

                   do i_freq = 1, n_frequencies

                      lamb_inv_use = lamb_inv(i_freq)
                      if (lambda_cm(i_freq)/1e-4 < 0.168d0) then
                         lamb_inv_use = 1d0/0.168d-4
                      end if
                      nm1 = (22851d0 + 0.456d12/(71427d0**2d0-lamb_inv_use**2d0))*1d-8
                      nm1 = nm1 * scale
                      fk = 1.016d0

                      rayleigh_scattering_opacities(i_freq,i_str) = rayleigh_scattering_opacities(i_freq,i_str) &
                           + 24d0*cst_pi**3d0*lamb_inv(i_freq)**4d0/(nfr_co)**2d0* &
                           (((nm1+1d0)**2d0-1d0)/((nm1+1d0)**2d0+2d0))**2d0*fk / mass_co * &
                           mass_fractions(i_str)

                   end do
                end if

             end do
            else if (trim(adjustl(species)) == 'CH4') then
             ! CH4 Rayleigh scattering according to Sneep & Ubachs (2004)

             lambda_cm = wavelengths_angstroem*1d-8
             lamb_inv = 1d0/lambda_cm

             d = mean_molar_masses*cst_amu*pressures/cst_k/temperatures*mass_fractions

             do i_str = 1, n_layers

                if (mass_fractions(i_str) > 1d-60) then

                   scale = d(i_str)/16d0/cst_amu/cst_sneep_ubachs_n
                   nfr_ch4 = d(i_str)/16d0/cst_amu
                   mass_ch4 = 16d0*cst_amu

                   do i_freq = 1, n_frequencies

                      nm1 = (46662d0 + 4.02d-6*lamb_inv(i_freq)**2d0)*1d-8
                      nm1 = nm1 * scale
                      fk = 1.0
                      rayleigh_scattering_opacities(i_freq,i_str) = rayleigh_scattering_opacities(i_freq,i_str) &
                           + 24d0*cst_pi**3d0*lamb_inv(i_freq)**4d0/(nfr_ch4)**2d0* &
                           (((nm1+1d0)**2d0-1d0)/((nm1+1d0)**2d0+2d0))**2d0*fk / mass_ch4 * &
                           mass_fractions(i_str)

                   end do
                end if

             end do
            end if

            contains
                subroutine h2_rayleigh()
                    ! H2 Rayleigh according to dalgarno & williams (1962)
                    implicit none

                    integer :: i
                    doubleprecision :: factor(n_frequencies)

                    factor = (&
                        8.14d-13 / wavelengths_angstroem ** 4 &
                        + 1.28d-6 / wavelengths_angstroem ** 6 &
                        + 1.61d0 / wavelengths_angstroem ** 8 &
                    ) * 0.5d0 / 1.66053892d-24

                    rayleigh_scattering_opacities = 0d0

                    do i = 1, n_layers
                        rayleigh_scattering_opacities(:, i) = rayleigh_scattering_opacities(:, i) + factor * mass_fractions(i)
                    end do

                    do i = 1, n_layers
                        if (mass_fractions(i) < abundance_threshold) then
                            rayleigh_scattering_opacities(:, i) = 0d0
                        end if
                    end do
                end subroutine h2_rayleigh

                subroutine he_rayleigh()
                    ! He Rayleigh scattering according to Chan & Dalgarno alphas (1965)
                    implicit none

                    integer :: i
                    double precision :: alpha_polarization(n_frequencies)

                    rayleigh_scattering_opacities = 0d0

                    lambda_cm = wavelengths_angstroem * 1d-8
                    l = lambda_cm * 1d4

                    alpha_polarization = 2.265983321d0 - 3.721350022d0 * l + 3.016150391d0 * l ** 2d0

                    do i_freq = 1, n_frequencies
                        if (lambda_cm(i_freq) >= 0.9110d-4) then
                            alpha_polarization(i_freq) = 1.379
                        end if
                    end do

                    ! Use alpha_polarization to store mass_fractions multiplicative factor
                    alpha_polarization = 128d0 * cst_pi ** 5d0 / 3d0 / lambda_cm ** 4d0 &
                        * (alpha_polarization * 1.482d-25) ** 2d0 / 4d0 / 1.66053892d-24

                    do i = 1, n_layers
                        rayleigh_scattering_opacities(:, i) = rayleigh_scattering_opacities(:, i) &
                            + alpha_polarization * mass_fractions(i)
                    end do

                    do i = 1, n_layers
                        if (mass_fractions(i) < abundance_threshold) then
                            rayleigh_scattering_opacities(:, i) = 0d0
                        end if
                    end do
                end subroutine he_rayleigh
        end subroutine compute_rayleigh_scattering_opacities


        subroutine compute_rosseland_opacities(opacities, temperatures, weights_gauss, frequencies_bin_edges, &
                                               scattering_in_emission, continuum_opacities_scattering, &
                                               n_g, n_frequencies, n_layers, n_frequencies_bin_edges, &
                                               opacities_rosseland)
            use physics, only: compute_rosseland_opacities_core

            implicit none

            integer,          intent(in)  :: n_g, n_frequencies, n_layers, n_frequencies_bin_edges
            double precision, intent(in)  :: opacities(n_g, n_frequencies, n_layers)
            double precision, intent(in)  :: frequencies_bin_edges(n_frequencies_bin_edges)
            double precision, intent(in)  :: temperatures(n_layers), weights_gauss(n_g)
            logical, intent(in)           :: scattering_in_emission
            double precision, intent(in)  :: continuum_opacities_scattering(n_frequencies,n_layers)
            double precision, intent(out) :: opacities_rosseland(n_layers)

            double precision              :: total_kappa_use(n_g, n_frequencies, n_layers)

            integer                       :: i_struc, i_g

            if (scattering_in_emission) then
                do i_g = 1, n_g
                    total_kappa_use(i_g,:,:) = opacities(i_g,:,:) + continuum_opacities_scattering
                end do
            else
                total_kappa_use = opacities
            end if

            do i_struc = 1, n_layers
                call compute_rosseland_opacities_core( &
                    total_kappa_use(:,:,i_struc), &
                    frequencies_bin_edges, &
                    temperatures(i_struc), &
                    weights_gauss, &
                    n_frequencies+1, n_g, &
                    opacities_rosseland(i_struc) &
                )
            end do
        end subroutine compute_rosseland_opacities


        subroutine compute_transit_radii(opacities, temperatures, pressures, surface_gravity, mean_molar_masses, &
                                         reference_pressure, planet_radius, weights_gauss, &
                                         scattering_in_transmission, continuum_opacities_scattering, variable_gravity, &
                                         n_frequencies, n_layers, n_g, n_species, &
                                         transit_radii, radius_hydrostatic_equilibrium)
            ! """
            ! Calculate the transmission spectrum
            ! """
            use physics, only: cst_amu, cst_k

            implicit none

            ! I/O
            integer, intent(in)                         :: n_frequencies, n_layers, n_g, n_species
            double precision, intent(in)                :: reference_pressure, planet_radius
            double precision, intent(in)                :: temperatures(n_layers), pressures(n_layers), &
                mean_molar_masses(n_layers)
            double precision, intent(in)                :: opacities(n_g,n_frequencies,n_species,n_layers)

            double precision, intent(in)                :: surface_gravity
            double precision, intent(in)                :: weights_gauss(n_g), &
                continuum_opacities_scattering(n_frequencies,n_layers)
            logical, intent(in)                         :: scattering_in_transmission
            logical, intent(in)                         :: variable_gravity

            double precision, intent(out)               :: transit_radii(n_frequencies), &
                radius_hydrostatic_equilibrium(n_layers)

            ! Internal
            double precision                            :: P0_cgs, rho(n_layers), radius_squared(n_layers), &
                opacities_(n_g,n_frequencies,n_species,n_layers), d_radius(n_layers)
            integer                                     :: i_str, i_freq, i_g, i_spec, j_str, sizet2
            logical                                     :: rad_neg
            double precision                            :: alpha_t2(n_g,n_frequencies,n_species,n_layers)
            double precision                            :: t_graze(n_g,n_frequencies,n_species,n_layers), &
               t_graze_wlen_int(n_frequencies, n_layers), &
               t_graze_wlen_int_1(n_frequencies, n_layers), &
               t_graze_wlen_int_2(n_frequencies, n_layers), &
               t_graze_wlen_int_t(n_layers, n_frequencies), &
               alpha_t2_scat(n_frequencies,n_layers-1), t_graze_scat(n_frequencies,n_layers), &
                radius_squared2(n_layers, n_layers)

            rad_neg = .False.
            sizet2 = n_g * n_frequencies * n_species

            radius_hydrostatic_equilibrium = 0d0
            transit_radii = 0d0

            t_graze = 0d0
            t_graze_scat = 0d0
            alpha_t2 = 0d0
            alpha_t2_scat = 0d0
            t_graze_wlen_int_t = 0d0

            opacities_ = opacities

            ! Some cloud opas can be < 0 sometimes, apparently.
            do i_str = 1, n_layers
                do i_spec = 1, n_species
                    do i_freq = 1, n_frequencies
                        do i_g = 1, n_g
                            if (opacities_(i_g, i_freq, i_spec, i_str) < 0d0) then
                                opacities_(i_g, i_freq, i_spec, i_str) = 0d0
                            end if
                        end do
                    end do
                end do
            end do

            ! Convert reference pressure to cgs
            P0_cgs = reference_pressure * 1d6

            ! Calculate density
            rho = mean_molar_masses * cst_amu * pressures / cst_k / temperatures

            ! Calculate planetary radius_hydrostatic_equilibrium (in cm), assuming hydrostatic equilibrium
            call compute_radius_hydrostatic_equilibrium(&
                pressures, surface_gravity, rho, P0_cgs, planet_radius, variable_gravity, n_layers, &
                radius_hydrostatic_equilibrium &
            )

            do i_str = n_layers, 1, -1
                if (radius_hydrostatic_equilibrium(i_str) < 0d0) then
                    rad_neg = .True.
                    radius_hydrostatic_equilibrium(i_str) = radius_hydrostatic_equilibrium(i_str + 1)
                end if
            end do

            if (rad_neg) then
                write(*, *) 'pRT: negative planet radius corretion applied!'
            end if

            ! Calculate mean free paths across grazing distances
            do i_str = 1, n_layers - 1
                alpha_t2(:, :, :, i_str) = opacities_(:, :, :, i_str) * rho(i_str)
            end do

            alpha_t2(:, :, :, 1:n_layers - 2) = &
                alpha_t2(:, :, :, 1:n_layers - 2) + alpha_t2(:, :, :, 2:n_layers - 1)
            alpha_t2(:, :, :, n_layers - 1) = &
                alpha_t2(:, :, :, n_layers - 1) + opacities_(:, :, :, n_layers) * rho(n_layers)

            if (scattering_in_transmission) then
                do i_str = 1, n_layers - 1
                    alpha_t2_scat(:, i_str) = continuum_opacities_scattering(:, i_str) * rho(i_str)
                end do

                alpha_t2_scat(:, 1:n_layers - 2) = &
                    alpha_t2_scat(:, 1:n_layers - 2) + alpha_t2_scat(:, 2:n_layers - 1)
                alpha_t2_scat(:, n_layers - 1) = &
                    alpha_t2_scat(:, n_layers - 1) + continuum_opacities_scattering(:, n_layers) * rho(n_layers)
            end if

            ! Calculate grazing rays optical depths
            radius_squared(:) = radius_hydrostatic_equilibrium(:) ** 2d0
            radius_squared2(:, :) = 0d0

            do i_str = 1, n_layers
                do j_str = 1, i_str - 1
                    radius_squared2(j_str, i_str) = radius_squared(j_str) - radius_squared(i_str)
                end do
            end do

            radius_squared2 = sqrt(radius_squared2)
            radius_squared2(1:n_layers - 1, :) = radius_squared2(1:n_layers - 1, :) - radius_squared2(2:n_layers, :)

            ! Bottleneck is here
            do i_str = 2, n_layers
                do j_str = 1, i_str - 1
                    t_graze(:, :, :, i_str) = &
                        t_graze(:, :, :, i_str) + alpha_t2(:, :, :, j_str) * radius_squared2(j_str, i_str)
                    t_graze_scat(:, i_str) = &
                        t_graze_scat(:, i_str) + alpha_t2_scat(:, j_str) * radius_squared2(j_str, i_str)
                end do
            end do

            ! Safeguard
            if (.not. scattering_in_transmission) then
                t_graze_scat = 0d0
            end if

            ! Calculate transmissions, update optical depths array to store these
            t_graze = exp(-t_graze)

            if (scattering_in_transmission) then
                t_graze_scat = exp(-t_graze_scat)
            end if

            t_graze_wlen_int = 1d0
            ! Wlen (in g-space) integrate transmissions
            do i_str = 2, n_layers ! i_str=1 t_grazes are 1 anyways
                do i_spec = 1, n_species
                    do i_freq = 1, n_frequencies
                        t_graze_wlen_int(i_freq, i_str) = t_graze_wlen_int(i_freq, i_str) * &
                            sum(t_graze(:, i_freq, i_spec, i_str) * weights_gauss)

                        if (scattering_in_transmission .and. (i_spec == 1)) then
                            t_graze_wlen_int(i_freq, i_str) = t_graze_wlen_int(i_freq, i_str) * &
                               t_graze_scat(i_freq, i_str)
                        end if
                    end do
                end do
            end do

            t_graze_wlen_int = 1d0 - t_graze_wlen_int

            do i_str = 1, n_layers
                t_graze_wlen_int(:, i_str) = t_graze_wlen_int(:, i_str) * radius_hydrostatic_equilibrium(i_str)
            end do

            d_radius(1) = 0d0

            do i_str = 2, n_layers
                d_radius(i_str) = radius_hydrostatic_equilibrium(i_str - 1) - radius_hydrostatic_equilibrium(i_str)
            end do

            t_graze_wlen_int_1(:, 1) = 0d0
            t_graze_wlen_int_2(:, 1) = 0d0

            do i_str = 2, n_layers
                t_graze_wlen_int_1(:, i_str) = t_graze_wlen_int(:, i_str - 1) * d_radius(i_str)
                t_graze_wlen_int_2(:, i_str) = t_graze_wlen_int(:, i_str) * d_radius(i_str)
            end do

            t_graze_wlen_int = t_graze_wlen_int_1 + t_graze_wlen_int_2
            t_graze_wlen_int_t = transpose(t_graze_wlen_int)

            ! Caculate planets effectice area (leaving out cst_pi, because we want the radius in the end)
            do i_freq = 1, n_frequencies
                transit_radii(i_freq) = sum(transit_radii(i_freq) + t_graze_wlen_int_t(:, i_freq))
            end do

            ! Get radius_hydrostatic_equilibrium
            transit_radii = sqrt(transit_radii + radius_hydrostatic_equilibrium(n_layers) ** 2d0)
        end subroutine compute_transit_radii


        subroutine compute_transmission_spectrum_contribution(opacities, temperatures, pressures, surface_gravity,&
                                                              mean_molar_masses, reference_pressure, planet_radius, &
                                                              weights_gauss, transit_radii_squared, &
                                                              scattering_in_transmission, &
                                                              continuum_opacities_scattering, variable_gravity, &
                                                              n_frequencies, n_layers, n_g, n_species, &
                                                              transmission_contribution, &
                                                              radius_hydrostatic_equilibrium)
            ! """
            ! Calculate the contribution function of the transmission spectrum.
            ! """
            use physics, only: cst_amu, cst_k
            
            implicit none
            
            logical, intent(in)                         :: scattering_in_transmission
            logical, intent(in)                         :: variable_gravity
            integer, intent(in)                         :: n_frequencies, n_layers, n_g, n_species
            double precision, intent(in)                :: reference_pressure, planet_radius
            double precision, intent(in)                :: temperatures(n_layers), pressures(n_layers), &
                mean_molar_masses(n_layers)
            double precision, intent(in)                :: opacities(n_g,n_frequencies,n_species,n_layers)
            double precision, intent(in)                :: surface_gravity
            double precision, intent(in)                :: weights_gauss(n_g), &
                continuum_opacities_scattering(n_frequencies,n_layers)
            double precision, intent(in)                :: transit_radii_squared(n_frequencies)
            double precision, intent(out)               :: transmission_contribution(n_layers,n_frequencies), &
                radius_hydrostatic_equilibrium(n_layers)
            
            integer                                     :: i_str, i_freq,  i_spec, j_str, i_leave_str
            double precision                            :: P0_cgs, rho(n_layers)
            double precision                            :: alpha_t2(n_g,n_frequencies,n_species,n_layers-1)
            double precision                            :: t_graze(n_g,n_frequencies,n_species,n_layers), s_1, s_2, &
               t_graze_wlen_int(n_layers,n_frequencies), alpha_t2_scat(n_frequencies,n_layers-1), &
               t_graze_scat(n_frequencies,n_layers), total_kappa_use(n_g,n_frequencies,n_species,n_layers), &
               continuum_opa_scat_use(n_frequencies,n_layers), transit_radii(n_frequencies)
            
            ! Convert reference pressure to cgs
            P0_cgs = reference_pressure * 1d6
            
            ! Calculate density
            rho = mean_molar_masses * cst_amu * pressures / cst_k / temperatures
            
            ! Calculate planetary radius (in cm), assuming hydrostatic equilibrium
            call compute_radius_hydrostatic_equilibrium(&
                pressures, surface_gravity, rho, P0_cgs, planet_radius, variable_gravity, n_layers, &
                radius_hydrostatic_equilibrium &
            )
            
            do i_leave_str = 1, n_layers
                transit_radii = 0d0
                t_graze = 0d0
                t_graze_scat = 0d0
                
                continuum_opa_scat_use = continuum_opacities_scattering
                total_kappa_use = opacities
                total_kappa_use(:, :, :, i_leave_str) = 0d0
                continuum_opa_scat_use(:, i_leave_str) = 0d0
                
                ! Calc. mean free paths across grazing distances
                do i_str = 1, n_layers-1
                    alpha_t2(:, :, :, i_str) = &
                        total_kappa_use(:, :, :, i_str) * rho(i_str) &
                        + total_kappa_use(:, :, :, i_str + 1) * rho(i_str + 1)
                end do
                
                if (scattering_in_transmission) then
                    do i_str = 1, n_layers - 1
                        alpha_t2_scat(:,i_str) = continuum_opa_scat_use(:, i_str) * rho(i_str) &
                            + continuum_opa_scat_use(:, i_str + 1) * rho(i_str + 1)
                    end do
                end if
                
                ! Cacuclate grazing rays optical depths
                do i_str = 2, n_layers
                    s_1 = sqrt(radius_hydrostatic_equilibrium(1) ** 2d0 - radius_hydrostatic_equilibrium(i_str) ** 2d0)
                    
                    do j_str = 1, i_str-1
                        if (j_str > 1) then
                            s_1 = s_2
                        end if
                       
                        s_2 = sqrt(radius_hydrostatic_equilibrium(j_str + 1) ** 2d0 &
                            - radius_hydrostatic_equilibrium(i_str) ** 2d0)
                        t_graze(:, :, :, i_str) = t_graze(:, :, :, i_str) + alpha_t2(:, :, :, j_str) * (s_1 - s_2)
                    end do
                end do
                
                if (scattering_in_transmission) then
                    do i_str = 2, n_layers
                        s_1 = sqrt(radius_hydrostatic_equilibrium(1) ** 2d0 &
                            - radius_hydrostatic_equilibrium(i_str) ** 2d0)
                       
                        do j_str = 1, i_str - 1
                            if (j_str > 1) then
                                s_1 = s_2
                            end if
                        
                            s_2 = sqrt(radius_hydrostatic_equilibrium(j_str + 1) ** 2d0 &
                                - radius_hydrostatic_equilibrium(i_str) ** 2d0)
                            t_graze_scat(:, i_str) = t_graze_scat(:, i_str) + alpha_t2_scat(:, j_str) * (s_1 - s_2)
                        end do
                    end do
                end if
                
                ! Calculate transmissions, update optical depths array to store these
                t_graze = exp(-t_graze)
                
                if (scattering_in_transmission) then
                    t_graze_scat = exp(-t_graze_scat)
                end if
                
                t_graze_wlen_int = 1d0
                
                ! Wlen (in g-space) integrate transmissions
                do i_str = 2, n_layers ! i_str=1 t_grazes are 1 anyways
                    do i_spec = 1, n_species
                        do i_freq = 1, n_frequencies
                            t_graze_wlen_int(i_str,i_freq) = t_graze_wlen_int(i_str,i_freq)&
                                * sum(t_graze(:, i_freq, i_spec, i_str) * weights_gauss)
                          
                            if (scattering_in_transmission .and. (i_spec == 1)) then
                                t_graze_wlen_int(i_str,i_freq) = t_graze_wlen_int(i_str, i_freq) &
                                    * t_graze_scat(i_freq, i_str)
                            end if
                        end do
                    end do
                end do
                
                ! Get effective area fraction from transmission
                t_graze_wlen_int = 1d0 - t_graze_wlen_int
                
                ! Caculate planets effectice area (leaving out cst_pi, because we want the radius in the end)
                do i_freq = 1, n_frequencies
                    do i_str = 2, n_layers
                        transit_radii(i_freq) = &
                            transit_radii(i_freq) &
                            + (&
                                t_graze_wlen_int(i_str - 1, i_freq) * radius_hydrostatic_equilibrium(i_str-1) &
                                + t_graze_wlen_int(i_str, i_freq) * radius_hydrostatic_equilibrium(i_str) &
                            ) &
                            * (radius_hydrostatic_equilibrium(i_str - 1) - radius_hydrostatic_equilibrium(i_str))
                    end do
                end do

                ! Get radius at hydrostatic equilibrium
                transit_radii = transit_radii+radius_hydrostatic_equilibrium(n_layers) ** 2d0
                transmission_contribution(i_leave_str, :) = transit_radii_squared - transit_radii
            end do
            
            do i_freq = 1, n_frequencies
                transmission_contribution(:, i_freq) = transmission_contribution(:, i_freq) &
                    / sum(transmission_contribution(:, i_freq))
            end do
        end subroutine compute_transmission_spectrum_contribution


        subroutine interpolate_cloud_opacities(clouds_total_absorption_opacities, clouds_total_scattering_opacities, &
                                               clouds_total_red_fac_aniso, cloud_wavelengths, frequencies_bin_edges, &
                                               n_cloud_wavelengths, n_layers, n_frequencies_bins, &
                                               clouds_final_absorption_opacities, cloud_abs_plus_scat_anisotropic, &
                                               clouds_final_red_fac_aniso, cloud_abs_plus_scat_no_anisotropic)
            ! """
            ! Interpolate cloud opacities to actual radiative transfer wavelength grid.
            ! """
          use physics, only: cst_c

          implicit none
          ! I/O
          integer, intent(in)           :: n_cloud_wavelengths,n_layers,n_frequencies_bins
          double precision, intent(in)  :: clouds_total_absorption_opacities(n_cloud_wavelengths,n_layers), &
               clouds_total_scattering_opacities(n_cloud_wavelengths,n_layers), &
               clouds_total_red_fac_aniso(n_cloud_wavelengths,n_layers), cloud_wavelengths(n_cloud_wavelengths), &
               frequencies_bin_edges(n_frequencies_bins)
          double precision, intent(out) :: clouds_final_absorption_opacities(n_frequencies_bins-1,n_layers), &
               cloud_abs_plus_scat_anisotropic(n_frequencies_bins-1,n_layers), &
               clouds_final_red_fac_aniso(n_frequencies_bins-1,n_layers), &
               cloud_abs_plus_scat_no_anisotropic(n_frequencies_bins-1,n_layers)

          ! internal
          double precision :: kappa_integ(n_layers), kappa_scat_integ(n_layers), red_fac_aniso_integ(n_layers), &
               kappa_tot_integ(n_frequencies_bins-1,n_layers), kappa_tot_scat_integ(n_frequencies_bins-1,n_layers)
          integer          :: HIT_i_lamb
          double precision :: HIT_border_lamb(n_frequencies_bins)
          integer          :: intp_index_small_min, intp_index_small_max, &
               new_small_ind

          clouds_final_absorption_opacities = 0d0
          cloud_abs_plus_scat_anisotropic = 0d0
          cloud_abs_plus_scat_no_anisotropic = 0d0


          HIT_border_lamb = cst_c/frequencies_bin_edges
          clouds_final_red_fac_aniso = 0d0

          kappa_tot_integ = 0d0
          kappa_tot_scat_integ = 0d0

          do HIT_i_lamb = 1, n_frequencies_bins-1

             intp_index_small_min = MIN(MAX(INT((log10(HIT_border_lamb(HIT_i_lamb))-log10(cloud_wavelengths(1))) / &
                  log10(cloud_wavelengths(n_cloud_wavelengths)/cloud_wavelengths(1))*dble(n_cloud_wavelengths-1) &
                  +1d0),1),n_cloud_wavelengths-1)

             intp_index_small_max = MIN(MAX(INT((log10(HIT_border_lamb(HIT_i_lamb+1))-log10(cloud_wavelengths(1))) / &
                  log10(cloud_wavelengths(n_cloud_wavelengths)/cloud_wavelengths(1))*dble(n_cloud_wavelengths-1) &
                  +1d0),1),n_cloud_wavelengths-1)

             kappa_integ = 0d0
             kappa_scat_integ = 0d0
             red_fac_aniso_integ = 0d0

             if ((intp_index_small_max-intp_index_small_min) == 0) then

                call intergrate_opacities_(intp_index_small_min,n_cloud_wavelengths,n_layers,&
                    clouds_total_absorption_opacities, &
                    cloud_wavelengths,HIT_border_lamb(HIT_i_lamb),HIT_border_lamb(HIT_i_lamb+1),kappa_integ)

                call intergrate_opacities_(intp_index_small_min,n_cloud_wavelengths,n_layers,&
                    clouds_total_scattering_opacities, &
                    cloud_wavelengths,HIT_border_lamb(HIT_i_lamb),HIT_border_lamb(HIT_i_lamb+1),kappa_scat_integ)

                call intergrate_opacities_(intp_index_small_min,n_cloud_wavelengths,n_layers,&
                    clouds_total_red_fac_aniso, &
                    cloud_wavelengths,HIT_border_lamb(HIT_i_lamb),HIT_border_lamb(HIT_i_lamb+1),red_fac_aniso_integ)

             else if ((intp_index_small_max-intp_index_small_min) == 1) then

                call intergrate_opacities_(intp_index_small_min,n_cloud_wavelengths,n_layers,&
                    clouds_total_absorption_opacities, &
                    cloud_wavelengths,HIT_border_lamb(HIT_i_lamb),cloud_wavelengths(intp_index_small_min+1),kappa_integ)

                call intergrate_opacities_(intp_index_small_min,n_cloud_wavelengths,n_layers,&
                    clouds_total_scattering_opacities, &
                    cloud_wavelengths,HIT_border_lamb(HIT_i_lamb),cloud_wavelengths(intp_index_small_min+1),&
                    kappa_scat_integ)

                call intergrate_opacities_(intp_index_small_min,n_cloud_wavelengths,n_layers,&
                    clouds_total_red_fac_aniso, &
                    cloud_wavelengths,HIT_border_lamb(HIT_i_lamb),cloud_wavelengths(intp_index_small_min+1),&
                    red_fac_aniso_integ)

                call intergrate_opacities_(intp_index_small_max,n_cloud_wavelengths,n_layers,&
                    clouds_total_absorption_opacities, &
                    cloud_wavelengths,cloud_wavelengths(intp_index_small_max),HIT_border_lamb(HIT_i_lamb+1),kappa_integ)

                call intergrate_opacities_(intp_index_small_max,n_cloud_wavelengths,n_layers,&
                    clouds_total_scattering_opacities, &
                    cloud_wavelengths,cloud_wavelengths(intp_index_small_max),HIT_border_lamb(HIT_i_lamb+1),&
                    kappa_scat_integ)

                call intergrate_opacities_(intp_index_small_max,n_cloud_wavelengths,n_layers,&
                    clouds_total_red_fac_aniso, &
                    cloud_wavelengths,cloud_wavelengths(intp_index_small_max),HIT_border_lamb(HIT_i_lamb+1),&
                    red_fac_aniso_integ)

             else

                call intergrate_opacities_(intp_index_small_min,n_cloud_wavelengths,n_layers,&
                    clouds_total_absorption_opacities, &
                    cloud_wavelengths,HIT_border_lamb(HIT_i_lamb),cloud_wavelengths(intp_index_small_min+1),kappa_integ)

                call intergrate_opacities_(intp_index_small_min,n_cloud_wavelengths,n_layers,&
                    clouds_total_scattering_opacities, &
                    cloud_wavelengths,HIT_border_lamb(HIT_i_lamb),cloud_wavelengths(intp_index_small_min+1),&
                    kappa_scat_integ)

                call intergrate_opacities_(intp_index_small_min,n_cloud_wavelengths,n_layers,&
                    clouds_total_red_fac_aniso, &
                    cloud_wavelengths,HIT_border_lamb(HIT_i_lamb),cloud_wavelengths(intp_index_small_min+1),&
                    red_fac_aniso_integ)

                new_small_ind = intp_index_small_min+1
                do while (intp_index_small_max-new_small_ind /= 0)

                   call intergrate_opacities_(new_small_ind,n_cloud_wavelengths,n_layers,&
                        clouds_total_absorption_opacities, &
                        cloud_wavelengths,cloud_wavelengths(new_small_ind),cloud_wavelengths(new_small_ind+1),&
                        kappa_integ)

                   call intergrate_opacities_(new_small_ind,n_cloud_wavelengths,n_layers,&
                        clouds_total_scattering_opacities, &
                        cloud_wavelengths,cloud_wavelengths(new_small_ind),cloud_wavelengths(new_small_ind+1),&
                        kappa_scat_integ)

                   call intergrate_opacities_(new_small_ind,n_cloud_wavelengths,n_layers,&
                        clouds_total_red_fac_aniso, &
                        cloud_wavelengths,cloud_wavelengths(new_small_ind),cloud_wavelengths(new_small_ind+1),&
                        red_fac_aniso_integ)

                   new_small_ind = new_small_ind+1

                end do

                call intergrate_opacities_(intp_index_small_max,n_cloud_wavelengths,n_layers,&
                    clouds_total_absorption_opacities, &
                    cloud_wavelengths,cloud_wavelengths(intp_index_small_max),HIT_border_lamb(HIT_i_lamb+1),kappa_integ)

                call intergrate_opacities_(intp_index_small_max,n_cloud_wavelengths,n_layers,&
                    clouds_total_scattering_opacities, &
                    cloud_wavelengths,cloud_wavelengths(&
                        intp_index_small_max),HIT_border_lamb(HIT_i_lamb+1),kappa_scat_integ)

                call intergrate_opacities_(intp_index_small_max,n_cloud_wavelengths,n_layers,&
                    clouds_total_red_fac_aniso, &
                    cloud_wavelengths,cloud_wavelengths(intp_index_small_max),HIT_border_lamb(HIT_i_lamb+1),&
                    red_fac_aniso_integ)

             end if

             kappa_integ = kappa_integ/(HIT_border_lamb(HIT_i_lamb+1)-HIT_border_lamb(HIT_i_lamb))
             kappa_scat_integ = kappa_scat_integ/(HIT_border_lamb(HIT_i_lamb+1)-HIT_border_lamb(HIT_i_lamb))
             red_fac_aniso_integ = red_fac_aniso_integ/(HIT_border_lamb(HIT_i_lamb+1)-HIT_border_lamb(HIT_i_lamb))

             kappa_tot_integ(HIT_i_lamb,:) = kappa_integ
             kappa_tot_scat_integ(HIT_i_lamb,:) = kappa_scat_integ

             clouds_final_absorption_opacities(HIT_i_lamb,:) = clouds_final_absorption_opacities(HIT_i_lamb,:) + &
                  kappa_integ
             cloud_abs_plus_scat_anisotropic(HIT_i_lamb,:) = cloud_abs_plus_scat_anisotropic(HIT_i_lamb,:) + &
                  kappa_integ + kappa_scat_integ * red_fac_aniso_integ
             cloud_abs_plus_scat_no_anisotropic(HIT_i_lamb,:) = cloud_abs_plus_scat_no_anisotropic(HIT_i_lamb,:) + &
                  kappa_integ + kappa_scat_integ

             clouds_final_red_fac_aniso(HIT_i_lamb,:) = clouds_final_red_fac_aniso(HIT_i_lamb,:) + red_fac_aniso_integ

          end do

            contains
                subroutine intergrate_opacities_(intp_ind,n_cloud_wavelengths,n_layers,kappa,lambda,l_bord1,l_bord2,&
                                                 kappa_integ)
                    ! """
                    ! Calculate the integral of a linearly interpolated function kappa.
                    ! """
                    implicit none

                    integer, intent(in) :: intp_ind,n_cloud_wavelengths,n_layers
                    double precision, intent(in) :: lambda(n_cloud_wavelengths), kappa(n_cloud_wavelengths,n_layers)
                    double precision, intent(in) :: l_bord1,l_bord2
                    double precision, intent(out) :: kappa_integ(n_layers)

                    kappa_integ = kappa_integ + kappa(intp_ind,:)*(l_bord2-l_bord1) &
                        + (kappa(intp_ind+1,:)-kappa(intp_ind,:))/ &
                    (lambda(intp_ind+1)-lambda(intp_ind))* &
                    0.5d0*((l_bord2-lambda(intp_ind))**2d0-(l_bord1-lambda(intp_ind))**2d0)
                end subroutine intergrate_opacities_
        end subroutine interpolate_cloud_opacities



        subroutine unused_compute_olson_kunasz_line_by_line_radiative_transfer(n_frequencies_bins,frequencies_bin_edges, &
                tau_approx,n_layers,pressures,temperatures,flux,&
                j_deep,n_angles,emission_cos_angles,emission_cos_angles_weights,clouds_final_absorption_opacities, &
                kappa_H,kappa_J,eddington_F,eddington_Psi,H_star,mu_star,H_star_0, &
                abs_S,t_irr,HIT_N_g_eff,J_bol,H_bol,K_bol, &
                w_gauss_ck,surface_gravity,dayside_ave,planetary_ave,j_for_zbrent,jstar_for_zbrent, &
                range_int,range_write,I_minus_out_Olson,bord_ind_1dm6,I_J_out_Feautrier)
            ! TODO never used, keeping just in case
            use math, only: cst_pi
            use physics, only: compute_planck_function, compute_rosseland_opacities_core
            
            implicit none

            logical, intent(in)             :: dayside_ave, planetary_ave
            integer, intent(in)             :: n_frequencies_bins, n_layers, HIT_N_g_eff, bord_ind_1dm6
            integer, intent(in)             :: n_angles, range_int, range_write
            double precision, intent(in)    :: frequencies_bin_edges(n_frequencies_bins), surface_gravity, &
                clouds_final_absorption_opacities(HIT_N_g_eff,n_frequencies_bins-1,n_layers)
            double precision, intent(in)    :: pressures(n_layers), temperatures(n_layers)
            double precision, intent(in)    :: tau_approx(HIT_N_g_eff,n_frequencies_bins-1,n_layers)
            double precision, intent(in)    :: emission_cos_angles(n_angles), emission_cos_angles_weights(n_angles),&
                w_gauss_ck(HIT_N_g_eff)
            double precision, intent(in)    :: H_star_0(n_frequencies_bins-1), t_irr, &
                                               I_minus_out_Olson(n_angles,HIT_N_g_eff,n_frequencies_bins-1)
            double precision, intent(inout) :: J_bol(n_layers), H_bol(n_layers), K_bol(n_layers)
            double precision, intent(inout) :: kappa_J(n_layers), kappa_H(n_layers)
            double precision, intent(inout) :: jstar_for_zbrent(HIT_N_g_eff,n_frequencies_bins-1,n_layers)
            double precision, intent(out)   :: eddington_F(n_layers), eddington_Psi
            double precision, intent(out)   :: flux(n_frequencies_bins-1), &
                                               j_deep(n_frequencies_bins-1), &
                                               I_J_out_Feautrier(n_angles,HIT_N_g_eff,n_frequencies_bins-1), &
                                               abs_S(n_layers), &
                                               j_for_zbrent(HIT_N_g_eff,n_frequencies_bins-1,n_layers)
            double precision, intent(out)   :: H_star(n_frequencies_bins-1,n_layers), mu_star

            double precision                :: del_tau(n_layers-1), atten_factors(n_layers-1), &
                                               mean_source(n_layers-1), atten_factors_no_exp(n_layers-1), &
                                               atten_factors_no_exp_p1(n_layers-2)
            double precision                :: plus_val_j(n_layers), plus_val_h(n_layers), &
                                               plus_val_k(n_layers)
            double precision                :: Q, Q_max, u, v, w, f, dt1, edt1,&
                                               dt2, e0, e1, e2, r(n_layers)
            double precision                :: j_out_test(n_layers,n_frequencies_bins-1), &
                                               kappa_out_test(n_layers,n_frequencies_bins-1), &
                                               kappa_S(n_layers), kappa_S_nu(n_layers,n_frequencies_bins-1), &
                                               H_tot_layer(n_layers), I_star_calc(HIT_N_g_eff,n_angles,n_layers)

            double precision                :: J_bol_a(n_layers), H_bol_a(n_layers), K_bol_a(n_layers)
            double precision                :: kappa_J_a(n_layers), kappa_H_a(n_layers)
            double precision                :: abs_S_nu(n_layers)
            double precision                :: H_star_calc(HIT_N_g_eff,n_layers)
            double precision                :: gamma

            character(len=8)                :: fmt
            integer                         :: j,i,k,l
            integer                         :: i_mu, range_int_use, range_write_use
            double precision                :: I_plus(n_layers,n_angles), I_minus(n_layers,n_angles)

            fmt = '(I5.5)'
            
            if (range_int < 0) then
                range_int_use = n_layers
            else
                range_int_use = range_int
            end if
            
            if (range_write < 0) then
                range_write_use = n_layers
            else
                range_write_use = range_write
            end if
            
            ! calculate how the stellar light is attenuated
            abs_S = 0d0
            
            kappa_out_test = 0d0
            j_out_test = 0d0
            kappa_S = 0d0
            kappa_S_nu = 0d0
            H_tot_layer = 0d0
            I_star_calc = 0d0

            ! Guillot test
            gamma = 6d-1 * sqrt(T_irr / 2d3)
            
            do i = 1, n_frequencies_bins-1
                if (dayside_ave .or. planetary_ave) then
                    do i_mu = 1, n_angles
                        I_star_calc(:,i_mu,:) = -4d0*H_star_0(i)*exp(-tau_approx(:,i,:)/emission_cos_angles(i_mu))
                    end do

                    H_star_calc = 0d0

                    do i_mu = 1, n_angles
                        H_star_calc(:,:) = H_star_calc(:,:)&
                            - 0.5d0*I_star_calc(:,i_mu,:)*emission_cos_angles(i_mu)*emission_cos_angles_weights(i_mu)
                    end do
                else
                    H_star_calc(:,:) = H_star_0(i)*exp(-tau_approx(:,i,:)/mu_star)
                end if

                H_star(i,:) = 0d0
                abs_S_nu = 0d0

                do j = 1, HIT_N_g_eff
                    H_star(i,:) = H_star(i,:)+H_star_calc(j,:)*w_gauss_ck(j)

                    ! Guillot test
                    if (dayside_ave .or. planetary_ave) then
                        do i_mu = 1, n_angles
                            abs_S_nu(:) = abs_S_nu(:) &
                                + 0.5d0 * w_gauss_ck(j) * clouds_final_absorption_opacities(j,i,:) &
                                * I_star_calc(j,i_mu,:) &
                                * emission_cos_angles_weights(i_mu)
                            jstar_for_zbrent(j,i,:) = jstar_for_zbrent(j,i,:) &
                                + 0.5d0 * I_star_calc(j,i_mu,:) * emission_cos_angles_weights(i_mu)
                        end do
                    else
                        abs_S_nu(:) = &
                            abs_S_nu(:)-H_star_calc(j,:)*w_gauss_ck(j)/mu_star*clouds_final_absorption_opacities(j,i,:)
                    end if

                    kappa_S = kappa_S+H_star_calc(j,:)*clouds_final_absorption_opacities(j,i,:)*w_gauss_ck(j)* &
                    (frequencies_bin_edges(i)-frequencies_bin_edges(i+1))

                    kappa_S_nu(:,i) = &
                        kappa_S_nu(:,i)+H_star_calc(j,:)*clouds_final_absorption_opacities(j,i,:)*w_gauss_ck(j)* &
                        (frequencies_bin_edges(i)-frequencies_bin_edges(i+1))

                    H_tot_layer = H_tot_layer + H_star_calc(j,:)*w_gauss_ck(j)*&
                        (frequencies_bin_edges(i)-frequencies_bin_edges(i+1))
                end do

                abs_S = abs_S + abs_S_nu*(frequencies_bin_edges(i)-frequencies_bin_edges(i+1))
            end do
            
            kappa_S = kappa_S / H_tot_layer

            do i = 1, n_frequencies_bins-1
                kappa_S_nu(:,i) = kappa_S_nu(:,i) / H_tot_layer
            end do
            ! ***0
            
            ! do the full RT step
            K_bol = 0d0
            
            do i = 1, n_frequencies_bins - 1
                flux(i) = 0d0
                j_deep(i) = 0d0
                J_bol_a = 0d0
                H_bol_a = 0d0
                K_bol_a = 0d0
                kappa_J_a = 0d0
                kappa_H_a = 0d0

                r = 0

                call compute_planck_function(&
                    temperatures, (frequencies_bin_edges(i) + frequencies_bin_edges(i + 1)) * 0.5d0, n_layers, r &
                )

                do l = 1, HIT_N_g_eff
                    ! Calculate the optical thickness of a layer
                    del_tau = 0d0
                    mean_source = 0d0

                    do k = 2, range_int_use
                        del_tau(k - 1) = tau_approx(l,i,k) - tau_approx(l,i,k - 1)
                        mean_source(k - 1) = (r(k - 1) + r(k)) * 0.5d0
                    end do

                    do j = 1, n_angles
                        ! Calculate the attenuation factors
                        atten_factors_no_exp = del_tau / emission_cos_angles(j)
                        atten_factors_no_exp_p1 = atten_factors_no_exp(2:n_layers-1)
                        atten_factors = exp(-atten_factors_no_exp)

                        ! Boundary conditions
                        I_plus(1, j) = 0d0

                        if (range_int_use == n_layers) then
                            I_minus(n_layers,j) = r(n_layers)
                        else
                            I_minus(range_int_use,j) = I_minus_out_Olson(j,l,i)
                        end if

                        ! integrate
                        do k = 2, range_int_use
                            ! I_plus calculation
                            if (k < range_int_use) then
                                dt1 = atten_factors_no_exp(k - 1)
                                f = atten_factors_no_exp(k) / dt1

                                edt1 = exp(-dt1)
                                e0 = 1d0 - edt1
                                e1 = dt1 - e0
                                e2 = dt1 ** 2d0 - 2d0 * e1
                                dt2 = f * dt1

                                if (dt1 > 1d-4) then
                                    u = e0 + (e2 - (2d0 * dt1 + dt2) * e1) / (dt1 * (dt1 + dt2))
                                    v = ((dt1 + dt2) * e1 - e2) / (dt1 * dt2)
                                    w = (e2-dt1*e1)/(dt2*(dt1+dt2))
                                else
                                    u = -(1d0/24d0)*dt1*(-8d0-12d0*f+6d0*dt1+8d0*f*dt1-2d0*dt1**2d0- &
                                        3d0*f*dt1**2+dt1**3d0+f*dt1**3d0)/(1d0+f)
                                    v = (1d0/24d0)*dt1*(4d0-2d0*dt1+dt1**2d0+12d0*f-4d0*f*dt1+f*dt1**2d0)/f
                                    w = (1d0/120d0)*dt1*(-20d0+10d0*dt1-3d0*dt1**2d0+dt1**3d0)/(f*(1d0+f))
                                end if

                                Q = u * r(k - 1) + v * r(k) + w * r(k + 1)
                                Q_max = 0.5d0 * (r(k - 1) * clouds_final_absorption_opacities(l, i, k - 1) &
                                    + r(k) * clouds_final_absorption_opacities(l, i, k)) &
                                    * (pressures(k) - pressures(k - 1)) &
                                    / surface_gravity / emission_cos_angles(j)  ! use MP/RT for rho?
                                Q = max(min(Q, Q_max), 0d0)
                                I_plus(k, j) = I_plus(k - 1, j)*atten_factors(k - 1) + Q
                            else
                                I_plus(k,j) = I_plus(k-1,j)*atten_factors(k-1)+(1d0-atten_factors(k-1))*mean_source(k-1)
                            end if

                            ! I_minus calculation
                            if (k < range_int_use) then  ! 2 times the same thing?
                                dt1 = atten_factors_no_exp(range_int_use - k + 1)
                                f = atten_factors_no_exp(range_int_use - k) / dt1

                                edt1 = exp(-dt1)
                                e0 = 1d0-edt1
                                e1 = dt1 -e0
                                e2 = dt1**2d0 - 2d0*e1
                                dt2 = f * dt1

                                if (dt1 > 1d-4) then
                                    u = e0 + (e2 - (2d0 * dt1 + dt2) * e1) / (dt1 * (dt1 + dt2))
                                    v = ((dt1+dt2)*e1-e2)/(dt1*dt2)
                                    w = (e2-dt1*e1)/(dt2*(dt1+dt2))
                                else
                                    u = -(1d0/24d0)*dt1*(-8d0-12d0*f+6d0*dt1+8d0*f*dt1-2d0*dt1**2d0- &
                                        3d0*f*dt1**2+dt1**3d0+f*dt1**3d0)/(1d0+f)
                                    v = (1d0/24d0)*dt1*(4d0-2d0*dt1+dt1**2d0+12d0*f-4d0*f*dt1+f*dt1**2d0)/f
                                    w = (1d0/120d0)*dt1*(-20d0+10d0*dt1-3d0*dt1**2d0+dt1**3d0)/(f*(1d0+f))
                                end if

                                Q = u * r(range_int_use - k + 2) + v * r(range_int_use - k + 1) &
                                    + w * r(range_int_use - k)
                                Q_max = 0.5d0*(r(range_int_use-k+2)&
                                    *clouds_final_absorption_opacities(l,i,range_int_use-k+2)+ &
                                    r(range_int_use-k+1)*clouds_final_absorption_opacities(l,i,range_int_use-k+1))* &
                                    (pressures(range_int_use-k+2)-pressures(range_int_use-k+1)) &
                                    /surface_gravity/emission_cos_angles(j)
                                Q = max(min(Q,Q_max),0d0)
                                I_minus(range_int_use-k+1,j) = I_minus(range_int_use-k+2,j)&
                                    * atten_factors(range_int_use-k+1) + Q
                            else
                                I_minus(range_int_use-k+1,j) = I_minus(range_int_use-k+2,j) &
                                    * atten_factors(range_int_use-k+1)&
                                    + (1d0-atten_factors(range_int_use-k+1))*mean_source(range_int_use-k+1)
                            end if
                        end do

                        I_J_out_Feautrier(j,l,i) = 0.5d0*(I_minus(bord_ind_1dm6,j)+I_plus(bord_ind_1dm6,j))
                    end do

                    ! OLD NON FEAUTRIER PART

                    plus_val_j = 0d0 ! flux going in
                    plus_val_h = 0d0 ! flux going in
                    plus_val_k = 0d0 ! flux going in

                    do j = 1, n_angles
                        !plus_val_j = plus_val_j + I_plus(:,j) * emission_cos_angles_weights(j)
                        plus_val_j = plus_val_j + (I_plus(:,j)+I_minus(:,j))/2d0 * emission_cos_angles_weights(j)
                        !neg_val_j = neg_val_j + I_minus(:,j) * emission_cos_angles_weights(j)

                        !plus_val_h = plus_val_h + I_plus(:,j) * emission_cos_angles(j) * emission_cos_angles_weights(j)
                        plus_val_h = plus_val_h + (I_plus(:,j)-I_minus(:,j))/2d0 &
                            * emission_cos_angles(j) * emission_cos_angles_weights(j)
                        !neg_val_h = neg_val_h - I_minus(:,j) * emission_cos_angles(j) * emission_cos_angles_weights(j)

                        !plus_val_k = plus_val_k + I_plus(:,j) * emission_cos_angles(j)**2d0 * emission_cos_angles_weights(j)
                        plus_val_k = plus_val_k + (I_plus(:,j)+I_minus(:,j))/2d0 &
                            * emission_cos_angles(j)**2d0 * emission_cos_angles_weights(j)
                        !neg_val_k = neg_val_k + I_minus(:,j) * emission_cos_angles(j)**2d0 * emission_cos_angles_weights(j)
                    end do

                    J_bol_a = J_bol_a + w_gauss_ck(l) * plus_val_j
                    H_bol_a = H_bol_a + w_gauss_ck(l) * plus_val_h
                    K_bol_a = K_bol_a + w_gauss_ck(l) * plus_val_k

                    if (range_write_use == n_layers) then
                        j_for_zbrent(l,i,1:range_write_use) = plus_val_j(1:range_write_use)
                    else
                        j_for_zbrent(l,i,1:range_write_use-1) = plus_val_j(1:range_write_use-1)
                    end if

                    kappa_J_a = kappa_J_a + clouds_final_absorption_opacities(l,i,:) * w_gauss_ck(l) * plus_val_j
                    kappa_H_a = kappa_H_a + clouds_final_absorption_opacities(l,i,:) * w_gauss_ck(l) * plus_val_h

                    kappa_out_test(:,i) = kappa_out_test(:,i) + clouds_final_absorption_opacities(l,i,:) * w_gauss_ck(l)
                end do

                j_out_test(:,i) = J_bol_a

                if (range_int_use == n_layers) then
                    j_deep(i) = J_bol_a(n_layers-1)
                end if

                flux(i) = H_bol_a(1) * 4d0 * cst_pi

                if (range_write_use == n_layers) then
                    J_bol(1:range_write_use) = J_bol(1:range_write_use) + &
                        J_bol_a(1:range_write_use)*(frequencies_bin_edges(i)-frequencies_bin_edges(i+1))
                    H_bol(1:range_write_use) = H_bol(1:range_write_use) + &
                        H_bol_a(1:range_write_use)*(frequencies_bin_edges(i)-frequencies_bin_edges(i+1))

                    K_bol = K_bol + K_bol_a*(frequencies_bin_edges(i)-frequencies_bin_edges(i+1))
                    kappa_J(1:range_write_use) = kappa_J(1:range_write_use) + kappa_J_a(1:range_write_use)* &
                        (frequencies_bin_edges(i)-frequencies_bin_edges(i+1))
                    kappa_H(1:range_write_use) = kappa_H(1:range_write_use) + kappa_H_a(1:range_write_use)* &
                        (frequencies_bin_edges(i)-frequencies_bin_edges(i+1))
                else
                    J_bol(1:range_write_use-1) = J_bol(1:range_write_use-1) + &
                        J_bol_a(1:range_write_use-1)*(frequencies_bin_edges(i)-frequencies_bin_edges(i+1))
                    H_bol(1:range_write_use-1) = H_bol(1:range_write_use-1) + &
                        H_bol_a(1:range_write_use-1)*(frequencies_bin_edges(i)-frequencies_bin_edges(i+1))

                    K_bol = K_bol + K_bol_a*(frequencies_bin_edges(i)-frequencies_bin_edges(i+1))
                    kappa_J(1:range_write_use-1) = kappa_J(1:range_write_use-1) + kappa_J_a(1:range_write_use-1)* &
                        (frequencies_bin_edges(i)-frequencies_bin_edges(i+1))
                    kappa_H(1:range_write_use-1) = kappa_H(1:range_write_use-1) + kappa_H_a(1:range_write_use-1)* &
                        (frequencies_bin_edges(i)-frequencies_bin_edges(i+1))
                end if
            end do
            
            if (range_write_use == n_layers) then
                kappa_J(1:range_write_use) = kappa_J(1:range_write_use) / J_bol(1:range_write_use)
                kappa_H(1:range_write_use) = kappa_H(1:range_write_use) / H_bol(1:range_write_use)
                eddington_F(1:range_write_use) = K_bol(1:range_write_use)/J_bol(1:range_write_use)
            else
                kappa_J(1:range_write_use-1) = kappa_J(1:range_write_use-1) / J_bol(1:range_write_use-1)
                kappa_H(1:range_write_use-1) = kappa_H(1:range_write_use-1) / H_bol(1:range_write_use-1)
                eddington_F(1:range_write_use-1) = K_bol(1:range_write_use-1)/J_bol(1:range_write_use-1)
            end if

            eddington_Psi = H_bol(1)/J_bol(1)

            ! Use that we are optically thick at the bottom (hopefully)
            if (range_int_use == n_layers) then
                call compute_rosseland_opacities_core(&
                    clouds_final_absorption_opacities(:,:,n_layers), &
                    frequencies_bin_edges, temperatures(n_layers), &
                    w_gauss_ck, &
                    n_frequencies_bins, HIT_N_g_eff, &
                    kappa_H(n_layers) &
                )
            end if
        end subroutine unused_compute_olson_kunasz_line_by_line_radiative_transfer


        subroutine unused_compute_olson_kunasz_ck_radiative_transfer(&
                frequencies_bin_edges, &
                tau_approx_scat, &
                temperatures, &
                emission_cos_angles, &
                emission_cos_angles_weights, &
                w_gauss_ck, &
                photon_destruct_in, &
                surf_refl, &
                surf_emi, &
                I_star_0, &
                geom, &
                mu_star, &
                flux, &
                n_frequencies_bin_edges, &
                n_layers, &
                n_angles, &
                n_g &
             )
            ! """Calculate the radiance from a planetary atmosphere, based on the method of Olson and Kunasz 1987.
            ! """
            ! TODO never used, keeping just in case
            use math, only: cst_pi
            use physics, only: compute_planck_function_integral

            implicit none

            character(len=*)                :: geom
            integer, intent(in)             :: n_frequencies_bin_edges, n_layers, n_g
            integer, intent(in)             :: n_angles
            double precision, intent(in)    :: photon_destruct_in(n_g,n_frequencies_bin_edges-1,n_layers)
            double precision, intent(in)    :: mu_star
            double precision, intent(in)    :: frequencies_bin_edges(n_frequencies_bin_edges), &
                                               surf_refl(n_frequencies_bin_edges-1),&
                                               surf_emi(n_frequencies_bin_edges-1), &
                                               temperatures(n_layers)
            double precision, intent(in)    :: tau_approx_scat(n_g,n_frequencies_bin_edges-1,n_layers)
            double precision, intent(in)    :: emission_cos_angles(n_angles), emission_cos_angles_weights(n_angles), &
                                               w_gauss_ck(n_g)
            double precision, intent(in)    :: I_star_0(n_frequencies_bin_edges-1)
            double precision, intent(out)   :: flux(n_frequencies_bin_edges-1)

            double precision                :: del_tau(n_layers-1), atten_factors(n_layers-1), &
                                               mean_source(n_layers-1), atten_factors_no_exp(n_layers-1), &
                                               atten_factors_no_exp_p1(n_layers-2)
            double precision                :: tmp_j, j_iter(n_layers, n_angles, n_g, n_frequencies_bin_edges - 1)
            double precision                :: Q, Q_max, u(n_layers-1), v(n_layers-1), w(n_layers-1), r(n_layers)
            double precision                :: &
                                               I_star_calc(n_g,n_angles,n_layers,n_frequencies_bin_edges-1), &
                                               photon_destruct(n_layers, n_g, n_frequencies_bin_edges-1)
            double precision                :: H_bol_a
            double precision                :: source(n_layers, n_g, n_frequencies_bin_edges - 1)
            double precision                :: J_star_ini(n_g,n_frequencies_bin_edges-1,n_layers)

            character(len=8)                :: fmt
            integer                         :: j,i,k,l
            integer                         :: i_mu
            double precision                :: I_plus(n_layers,n_angles), I_minus(n_layers,n_angles), &
                I_surface_emission, I_surface_reflection, I_surface_no_scattering(n_g, n_frequencies_bin_edges - 1), &
                surf_refl_2(n_frequencies_bin_edges-1), mu_weight(n_angles), planck(n_layers, n_frequencies_bin_edges-1)

            fmt = '(I5.5)'

            ! calculate how the stellar light is attenuated
            I_star_calc = 0d0

            surf_refl_2 = 2d0 * surf_refl
            mu_weight = emission_cos_angles * emission_cos_angles_weights
            photon_destruct = photon_destruct_in

            call compute_planck_function_integral(&
                    temperatures, frequencies_bin_edges, n_layers, n_frequencies_bin_edges, planck &
                )

            do i = 1, n_frequencies_bin_edges-1
                ! Irradiation treatment
                ! Dayside ave: multiply flux by 1/2.
                ! Planet ave: multiply flux by 1/4
                do i_mu = 1, n_angles
                    if (trim(adjustl(geom)) == 'dayside_ave') then
                        I_star_calc(:,i_mu,:,i) = 0.5 * abs(I_star_0(i)) &
                            * exp(-tau_approx_scat(:,i,:)/emission_cos_angles(i_mu))
                        J_star_ini(:,i,:) = J_star_ini(:,i,:) &
                            + 0.5d0*I_star_calc(:,i_mu,:,i)*emission_cos_angles_weights(i_mu)
                    else if (trim(adjustl(geom)) == 'planetary_ave') then
                        I_star_calc(:,i_mu,:,i) = 0.25 * abs(I_star_0(i)) &
                            * exp(-tau_approx_scat(:,i,:)/emission_cos_angles(i_mu))
                        J_star_ini(:,i,:) = J_star_ini(:,i,:) &
                            + 0.5d0*I_star_calc(:,i_mu,:,i)*emission_cos_angles_weights(i_mu)
                    else if (trim(adjustl(geom)) == 'non-isotropic') then
                        J_star_ini(:,i,:) = abs(I_star_0(i)/4.*exp(-tau_approx_scat(:,i,:)/mu_star))
                    else
                        write(*,*) 'Invalid geometry'
                    end if
                end do
            end do

            do i = 1, n_frequencies_bin_edges - 1
                I_surface_emission = surf_emi(i) * planck(n_layers, i)

                do l = 1, n_g
                    if  (trim(adjustl(geom)) /= 'non-isotropic') then
                        I_surface_reflection = surf_refl_2(i) &
                            * sum(I_star_calc(l, :, n_layers, i) * mu_weight)
                    else
                        I_surface_reflection = surf_refl(i) &
                            * J_star_ini(l, i, n_layers) * 4d0 * mu_star
                    end if

                    I_surface_no_scattering(l, i) = I_surface_emission + I_surface_reflection
                end do
            end do

            ! do the full RT step
            do i = 1, n_frequencies_bin_edges - 1
                flux(i) = 0d0
                H_bol_a = 0d0

                !r = planck(:, i)

                do l = 1, n_g
                    ! Calculate the optical thickness of a layer
                    del_tau = 0d0
                    mean_source = 0d0

                    !if (i_iter_scat == 1) then  ! initial guess
                        source(:, l, i) = photon_destruct(:, l, i) * r &
                            + (1d0 - photon_destruct(:, l, i)) * J_star_ini(l, i, :)
                    !else
                        r = source(:, l, i)
                    !end if

                    do k = 2, n_layers
                        del_tau(k - 1) = tau_approx_scat(l,i,k) - tau_approx_scat(l,i,k - 1)
                        mean_source(k - 1) = (r(k - 1) + r(k)) * 0.5d0
                    end do

                    do j = 1, n_angles
                        ! Calculate the attenuation factors
                        atten_factors_no_exp = del_tau / emission_cos_angles(j)
                        atten_factors_no_exp_p1 = atten_factors_no_exp(2:n_layers-1)
                        atten_factors = exp(-atten_factors_no_exp)

                        call olson_kunasz_integration(atten_factors_no_exp, n_layers, u, v, w)

                        ! Boundary conditions
                        I_plus(1, j) = 0d0
                        I_minus(n_layers, j) = I_surface_no_scattering(l, i)  ! TODO add scattering part

                        ! integrate
                        ! I_plus calculation
!                        do k = 2, n_layers - 1
!                            ! Q_max = 0.5 * (j_k-1 + j_k) * delta_s
!                            ! j_k = alpha_k * S_k = delta_tau_k / delta_s * S_k
!                            ! Q_max = 0.5 * (delta_tau_k-1 * S_k-1 + delta_tau_k * S_k)
!                            Q = u(k) * r(k - 1) + v(k) * r(k) + w(k) * r(k + 1)
!                            Q_max = 0.5d0 * (r(k - 1) * atten_factors_no_exp(k - 1) + r(k) * atten_factors_no_exp(k))
!                            Q = max(min(Q, Q_max), 0d0)
!
!                            I_plus(k, j) = I_plus(k - 1, j) * atten_factors(k - 1) + Q
!                        end do
!
!                        I_plus(n_layers, j) = I_plus(n_layers - 1, j) * atten_factors(n_layers - 1) &
!                            + (1d0 - atten_factors(n_layers - 1)) * mean_source(n_layers - 1)
                        I_plus(2:n_layers, j) = 0d0

                        ! I_minus calculation
                        do k = n_layers - 1, 2, -1
                            Q = u(k) * r(k + 1) + v(k) * r(k) + w(k) * r(k - 1)
                            Q_max = 0.5d0 * (r(k - 1) * atten_factors_no_exp(k - 1) + r(k) * atten_factors_no_exp(k))
                            Q = max(min(Q,Q_max),0d0)

                            I_minus(k, j) = I_minus(k + 1, j) * atten_factors(k) + Q
                        end do

                        I_minus(1, j) = I_minus(2, j) * atten_factors(1) + (1d0 - atten_factors(1)) * mean_source(1)

                        j_iter(:, j, l, i) = 0.5d0 * (I_minus(:, j) + I_plus(:, j))
                    end do  ! emission_cos_angles

                    tmp_j = 0d0

                    do j = 1, n_angles
                        tmp_j = tmp_j + j_iter(1, j, l, i) * emission_cos_angles(j) * emission_cos_angles_weights(j)
                    end do

                    H_bol_a = H_bol_a + w_gauss_ck(l) * tmp_j
                end do  ! g

                flux(i) = H_bol_a * 4d0 * cst_pi
            end do  ! frequencies

            contains
                subroutine olson_kunasz_integration(delta_tau, n_cells, u, v, w)
                    implicit none

                    integer, intent(in) :: n_cells
                    double precision, intent(in) :: delta_tau(n_cells - 1)
                    double precision, intent(out) :: u(n_cells - 1), v(n_cells - 1), w(n_cells - 1)

                    integer :: k
                    double precision :: e0(n_cells - 1), e1(n_cells - 1), e2(n_cells - 1), f(n_cells - 1), &
                        exp_dtau(n_cells - 1)

                    f = delta_tau(2:n_cells) / delta_tau(1:n_cells - 1)

                    exp_dtau = exp(-delta_tau(1:n_cells - 1))

                    e0 = 1d0 - exp_dtau
                    e1 = delta_tau(1:n_cells - 1) - e0
                    e2 = delta_tau(1:n_cells - 1) ** 2d0 - 2d0 * e1

                    u(1) = 0d0
                    v(1) = 0d0
                    w(1) = 0d0

                    do k = 2, n_cells - 1
                        if (delta_tau(k) > 1d-4) then
                            u(k) = e0(k) + (e2(k) - (2d0 * delta_tau(k - 1) + delta_tau(k)) * e1(k)) &
                                / (delta_tau(k - 1) * (delta_tau(k - 1) + delta_tau(k)))
                            v(k) = ((delta_tau(k - 1) + delta_tau(k)) * e1(k) - e2(k)) &
                                / (delta_tau(k - 1) * delta_tau(k))
                            w(k) = (e2(k) - delta_tau(k - 1) * e1(k)) &
                                / (delta_tau(k) * (delta_tau(k - 1) + delta_tau(k)))
                        else  ! prevent numerical unstabilities
                            u(k) = - (1d0 / 24d0) * delta_tau(k - 1) * (&
                                    -8d0 - 12d0 * f(k) + 6d0 * delta_tau(k - 1) &
                                    + 8d0 * delta_tau(k) - 2d0 * delta_tau(k - 1) ** 2d0 &
                                    - 3d0 * delta_tau(k) ** 2d0 + delta_tau(k - 1) ** 3d0 &
                                    + delta_tau(k) ** 3d0&
                                ) / (1d0 + f(k))
                            v(k) = (1d0 / 24d0) * delta_tau(k - 1) * (&
                                    4d0 - 2d0 * delta_tau(k - 1) + delta_tau(k - 1) ** 2d0 &
                                    +12d0 * f(k) - 4d0 * delta_tau(k) + delta_tau(k) ** 2d0 &
                                ) / f(k)
                            w(k) = (1d0 / 120d0) * delta_tau(k - 1) * (&
                                    -20d0 + 10d0 * delta_tau(k - 1) - 3d0 * delta_tau(k - 1) ** 2d0 &
                                    + delta_tau(k - 1) ** 3d0&
                                ) / (f(k) * (1d0 + f(k)))
                        end if
                    end do
                end subroutine olson_kunasz_integration
        end subroutine unused_compute_olson_kunasz_ck_radiative_transfer
end module fortran_radtrans_core
