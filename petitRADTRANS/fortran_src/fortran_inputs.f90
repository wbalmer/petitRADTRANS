!!$*****************************************************************************
!!$*****************************************************************************
!!$*****************************************************************************
!!$ fortran_inputs.f90: utility functions to read, interpolate and mix opacities
!!$                 for the petitRADTRANS radiative transfer package
!!$
!!$ Copyright 2016-2018, Paul Molliere
!!$ Maintained by Paul Molliere, molliere@strw.leidenunivl.nl
!!$ Status: under development
!!$*****************************************************************************
!!$*****************************************************************************
!!$*****************************************************************************
! TODO remove labels in fortran_inputs

module fortran_inputs
    implicit none

    integer, parameter :: path_string_max_length = 1024  ! Windows path length soft limit is 256
    integer, parameter :: species_string_max_length = 64  ! longest molelcule formula ("tintin") is 30 characters long

    contains
        subroutine compute_file_size(file_path, arr_len)
            ! """
            ! Subroutine to get a kappa array size in the high-res case.
            ! """
            implicit none

            character(len=*), intent(in)     :: file_path
            integer, intent(out) :: arr_len

            integer          :: io
            doubleprecision :: dump

            open(unit=49, file=trim(adjustl(file_path)), form = 'unformatted', ACCESS='stream')

            arr_len = 0

            do
                read(49, iostat=io) dump
                arr_len = arr_len + 1

                if (mod(arr_len, 500000) == 0) then
                    write(*, *) arr_len, dump, io  ! check that we are reading the file and not stalling
                end if

                if(io < 0) then
                    exit
                end if
            end do

            close(49)

            arr_len = arr_len - 1
        end subroutine compute_file_size


        subroutine compute_total_opacities(abundances,opa_struc_kappas,continuum_opa, &
                                           N_species,freq_len,struc_len,g_len,opa_struc_kappas_out)
            ! """
            ! Subroutine to get the abundance weightes opas for ck, and for adding the continuum opas.
            ! """
            implicit none

            integer, intent(in) :: N_species,freq_len,struc_len,g_len
            double precision, intent(in) :: abundances(struc_len,N_species), &
            continuum_opa(freq_len,struc_len)
            double precision, intent(in) :: opa_struc_kappas(g_len,freq_len,N_species,struc_len)
            double precision, intent(out) :: opa_struc_kappas_out(g_len,freq_len,N_species,struc_len)

            integer :: i_spec, i_struc, i_freq

            do i_struc = 1, struc_len
                do i_spec = 1, N_species
                    opa_struc_kappas_out(:,:,i_spec,i_struc) = abundances(i_struc,i_spec) &
                        * opa_struc_kappas(:,:,i_spec,i_struc)
                end do
            end do

            do i_struc = 1, struc_len
                do i_freq = 1, freq_len
                    opa_struc_kappas_out(:,i_freq,1,i_struc) = &
                        opa_struc_kappas_out(:,i_freq,1,i_struc) &
                        + continuum_opa(i_freq,i_struc)
                end do
            end do
        end subroutine compute_total_opacities


        subroutine find_interpolate_indices(binbord,binbordlen,arr,arrlen,intpint)
            ! """
            ! TODO add docstring in find_interpolate_indices
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
                else
                    k0 = 1
                    km = binbordlen
                    pivot = (km + k0) / 2

                    do while(km - k0 > 1)
                        if (arr(i_arr) >= binbord(pivot)) then
                            k0 = pivot
                            pivot = (km + k0) / 2
                        else
                            km = pivot
                            pivot = (km + k0) / 2
                        end if
                    end do

                    intpint(i_arr) = k0
                end if
            end do
        end subroutine find_interpolate_indices


        subroutine interpolate_line_opacities(pressures, temperatures, temperature_pressure_grid, custom_grid, &
                                          n_temperatures, n_pressures, opacity_grid, &
                                          n_layers, n_tp_grid_columns, n_tp_grid_rows, n_frequencies, n_g, &
                                          interpolated_opacities)
            ! """
            ! Subroutine to interpolate an opacity grid at a given temperature profile.
            ! """
            implicit none

            double precision, parameter :: temperature_index_factor1 = 81.14113604736988d0
            double precision, parameter :: temperature_index_factor2 = 12d0 / log10(2995d0 / temperature_index_factor1)

            integer, intent(in) :: n_g, n_layers, n_tp_grid_rows, n_tp_grid_columns ! n_tp_grid_columns = 2
            integer, intent(in) :: n_frequencies
            double precision, intent(in) :: pressures(n_layers), temperatures(n_layers)
            double precision, intent(in) :: temperature_pressure_grid(n_tp_grid_rows, n_tp_grid_columns)
            double precision, intent(in) :: opacity_grid(n_g, n_frequencies, n_tp_grid_rows)
            logical, intent(in) :: custom_grid
            integer, intent(in) :: n_temperatures, n_pressures
            double precision, intent(out) :: interpolated_opacities(n_g, n_frequencies, n_layers)

            integer :: i, i_temperature, i_pressure
            integer :: j_t0_p0, j_t0_p1, j_t1_p0, j_t1_p1
            integer :: j_t0_p0_pre, j_t0_p1_pre, j_t1_p0_pre, j_t1_p1_pre
            integer :: buffer_scalar_array(1)
            double precision :: dx, dy_0(n_g, n_frequencies), dy_1(n_g, n_frequencies), &
                y_p_low(n_g, n_frequencies), y_p_high(n_g, n_frequencies), &
                y_t_low(n_g, n_frequencies), y_t_high(n_g, n_frequencies)
            double precision :: pressure_grid(n_pressures), temperature_grid(n_temperatures)

            ! Initialize "custom" (non-default) pressure and temperature grid
            if (custom_grid) then
                pressure_grid = temperature_pressure_grid(1:n_pressures, 2)

                do i = 1, n_temperatures
                    temperature_grid(i) = temperature_pressure_grid((i - 1) * n_pressures + 1, 1)
                end do
            end if

            ! Initialize memory indices and arrays
            dy_0 = 0d0
            dy_1 = 0d0

            j_t0_p0_pre = -1
            j_t0_p1_pre = -1
            j_t1_p0_pre = -1
            j_t1_p1_pre = -1

            ! Loop over temperature profile's layers
            do i = 1, n_layers
                ! Get indices in temperature-pressure grid around the current layer's pressure and temperature
                if (custom_grid) then
                    ! Search the index in the grids that is the closest to the current layer's pressure and temperature
                    ! TODO is there truly an interest in searching for indices using a dichotomy instead of looping?
                    call find_interpolate_indices(temperature_grid, n_temperatures, temperatures(i), 1, buffer_scalar_array)

                    i_temperature = buffer_scalar_array(1)

                    call find_interpolate_indices(pressure_grid, n_pressures, pressures(i), 1, buffer_scalar_array)

                    i_pressure = buffer_scalar_array(1)

                    ! Get indices in temperature-pressure grid around the current layer's pressure and temperature
                    j_t0_p0 = (i_temperature - 1) * n_pressures + i_pressure
                    j_t0_p1 = (i_temperature - 1) * n_pressures + i_pressure + 1
                    j_t1_p0 = i_temperature * n_pressures + i_pressure
                    j_t1_p1 = i_temperature * n_pressures + i_pressure + 1
                else
                    ! Get the index in the grids that is the closest to the current layer's pressure and temperature
                    i_temperature = max(min(int(log10(temperatures(i) / temperature_index_factor1) &
                        * temperature_index_factor2) + 1, 12), 1)
                    i_pressure = max(min(int(log10(pressures(i) * 1d-6) + 6d0) + 1, 9), 1)

                    ! Get indices in temperature-pressure grid around the current layer's pressure and temperature
                    j_t0_p0 = (i_temperature - 1) * 10 + i_pressure
                    j_t0_p1 = (i_temperature - 1) * 10 + i_pressure + 1
                    j_t1_p0 = i_temperature * 10 + i_pressure
                    j_t1_p1 = i_temperature * 10 + i_pressure + 1
                end if

                ! Interpolation of opacity grid at lower temperature to the layer's pressure
                y_p_low = opacity_grid(:, :, j_t0_p0)
                y_p_high = opacity_grid(:, :, j_t0_p1)

                if (pressures(i) >= temperature_pressure_grid(j_t0_p1, 2)) then
                    y_t_low = y_p_high
                else if (pressures(i) <= temperature_pressure_grid(j_t0_p0, 2)) then
                    y_t_low = y_p_low
                else
                    dx = (pressures(i) - temperature_pressure_grid(j_t0_p0, 2)) &
                        / (temperature_pressure_grid(j_t0_p1, 2) - temperature_pressure_grid(j_t0_p0, 2))

                    if (j_t0_p0 /= j_t0_p0_pre .or. j_t0_p1 /= j_t0_p1_pre) then
                        ! No need to calculate dy_0 again if the indices have not updated
                        dy_0 = y_p_high - y_p_low

                        j_t0_p0_pre = j_t0_p0
                        j_t0_p1_pre = j_t0_p1
                    end if

                    y_t_low = y_p_low + dx * dy_0
                end if

                ! Interpolation of opacity grid at higher temperature to the layer's pressure
                y_p_low = opacity_grid(:, :, j_t1_p0)
                y_p_high = opacity_grid(:, :, j_t1_p1)

                if (pressures(i) >= temperature_pressure_grid(j_t1_p1, 2)) then
                    y_t_high = y_p_high
                else if (pressures(i) <= temperature_pressure_grid(j_t1_p0, 2)) then
                    y_t_high = y_p_low
                else
                    dx = (pressures(i) - temperature_pressure_grid(j_t1_p0, 2)) &
                        / (temperature_pressure_grid(j_t1_p1, 2) - temperature_pressure_grid(j_t1_p0, 2))

                    if (j_t1_p0 /= j_t1_p0_pre .or. j_t1_p1 /= j_t1_p1_pre) then
                        ! No need to calculate dy_1 again if the indices have not updated
                        dy_1 = y_p_high - y_p_low

                        j_t1_p0_pre = j_t1_p0
                        j_t1_p1_pre = j_t1_p1
                    end if

                    y_t_high = y_p_low + dx * dy_1
                end if

                ! Interpolation of opacity grid at the layer's pressure to the layer's temperature
                if (temperatures(i) >= maxval(temperature_pressure_grid(:, 1))) then
                    interpolated_opacities(:, :, i) = y_t_high
                else if (temperatures(i) <= minval(temperature_pressure_grid(:, 1))) then
                    interpolated_opacities(:, :, i) = y_t_low
                else
                    dx = (temperatures(i) - temperature_pressure_grid(j_t0_p0, 1)) &
                        / (temperature_pressure_grid(j_t1_p0, 1) - temperature_pressure_grid(j_t0_p0, 1))

                    interpolated_opacities(:, :, i) = y_t_low + dx * (y_t_high - y_t_low)
                end if
            end do
        end subroutine interpolate_line_opacities


        subroutine load_all_line_by_line_opacities(file_path, arr_len, kappa)
            ! """
            ! Subroutine to read all the kappa array in the high-res case.
            ! """
            implicit none

            character(len=*), intent(in)     :: file_path
            integer, intent(in) :: arr_len
            double precision, intent(out) :: kappa(arr_len)

            integer          :: i

            open(unit=49, file=trim(adjustl(file_path)), form = 'unformatted', ACCESS='stream')

            do i = 1, arr_len
                read(49) kappa(i)

                if (mod(i, 500000) == 0) then
                    write(*, '(I0, " / ", I0)') i, arr_len  ! check that we are reading the file and not stalling
                end if
            end do

            close(49)
        end subroutine load_all_line_by_line_opacities


        subroutine load_cia_opacities(cpair,opacity_path_str,CIA_cpair_lambda, &
                                      CIA_cpair_temp,CIA_cpair_alpha_grid,temp, wlen)
            ! """
            ! Subroutine to read the CIA opacities.
            ! """
            implicit none

            character(len=*), intent(in)              :: cpair
            double precision, intent(out)         :: CIA_cpair_alpha_grid(10000,50)
            double precision, intent(out)         :: CIA_cpair_lambda(10000)
            double precision, intent(out)         :: CIA_cpair_temp(50)
            character(len=*), intent(in)             :: opacity_path_str
            integer,intent(out)                   :: temp,wlen

            integer                               :: i,j,stat

            ! ELALEI allowing the tables to have whatever shape
            ! BUT the PYTHON CODE NEEDS TO TRIM THE OUTPUTS ACCORDING TO temp AND WLEN SO I CREATE NEW OUTPUTS
            open(unit=10,file=trim(adjustl(opacity_path_str)) &
               //'/opacities/continuum/CIA/'//trim(adjustl(cpair))//'/temps.dat')

            i=0

            do
                i = i + 1

                if (i>50) then
                    exit
                end if

                read(10,*,iostat=stat) CIA_cpair_temp(i)

                if (stat /= 0) then
                    exit
                end if
            end do

            close(10)
            temp=i-1

            open(unit=11,file=trim(adjustl(opacity_path_str)) &
                //'/opacities/continuum/CIA/'//trim(adjustl(cpair)) &
                //'/CIA_'//trim(adjustl(cpair))//'_final.dat')
            read(11,*)
            read(11,*)

            i = 0

            do
                i = i + 1

                if (i>10000) then
                    exit
                end if

                read(11,'(G22.12)',iostat=stat) CIA_cpair_lambda(i)

                if (stat /= 0) then
                    exit
                end if
            end do

            close(11)

            open(unit=11,file=trim(adjustl(opacity_path_str)) &
                //'/opacities/continuum/CIA/'//trim(adjustl(cpair)) &
                //'/CIA_'//trim(adjustl(cpair))//'_final.dat')
            read(11,*)
            read(11,*)

            wlen=i-1

            do i=1,wlen
                read(11,'(G22.12)',advance='no') CIA_cpair_lambda(i)

                do j = 1, temp-1
                    read(11,'(G22.12)',advance='no') CIA_cpair_alpha_grid(i,j)
                end do

                read(11,'(G22.12)') CIA_cpair_alpha_grid(i,temp)
            end do

            close(11)
        end subroutine load_cia_opacities


        subroutine load_cloud_opacities(path,species_names_tot,species_modes_tot,N_cloud_spec, &
                                        N_cloud_lambda_bins,rho_cloud_particles,cloud_specs_abs_opa,&
                                        cloud_specs_scat_opa, cloud_aniso,cloud_lambdas,cloud_rad_bins,cloud_radii)
            ! """
            ! Subroutine to read in the molecular opacities (c-k or line-by-line)
            ! """
            implicit none

            integer, parameter :: N_cloud_rad_bins = 130

            character(len=*), intent(in) :: path
            character(len=*), intent(in) :: species_names_tot,species_modes_tot
            integer, intent(in) :: N_cloud_spec,N_cloud_lambda_bins
            double precision, intent(out) :: rho_cloud_particles(N_cloud_spec)
            double precision, intent(out) :: cloud_specs_abs_opa(N_cloud_rad_bins,N_cloud_lambda_bins,N_cloud_spec), &
               cloud_specs_scat_opa(N_cloud_rad_bins,N_cloud_lambda_bins,N_cloud_spec), &
               cloud_aniso(N_cloud_rad_bins,N_cloud_lambda_bins,N_cloud_spec), cloud_lambdas(N_cloud_lambda_bins), &
               cloud_rad_bins(N_cloud_rad_bins+1), cloud_radii(N_cloud_rad_bins)

            integer :: i_str, curr_spec_ind, i_cloud, i_cloud_read, i_cloud_lamb, i_size, i_lamb
            integer :: species_name_inds(2,N_cloud_spec), species_mode_inds(2,N_cloud_spec)
            character(len=species_string_max_length)  :: cloud_opa_names(N_cloud_spec), cloud_name_buff, buff_line, path_add
            double precision :: cloud_dens_buff, buffer
            character(len=species_string_max_length) :: cloud_opa_mode(N_cloud_spec)

            ! Get single cloud species names
            curr_spec_ind = 1
            species_name_inds(1,curr_spec_ind) = 1
            do i_str = 1, len(species_names_tot)
                if (curr_spec_ind > N_cloud_spec) then
                    exit
                end if

                if (species_names_tot(i_str:i_str) == ':') then
                    species_name_inds(2,curr_spec_ind) = i_str-1
                    curr_spec_ind = curr_spec_ind+1

                    if (curr_spec_ind <= N_cloud_spec) then
                       species_name_inds(1,curr_spec_ind) = i_str+1
                    end if
                end if
            end do

            ! Get single cloud species modes
            curr_spec_ind = 1
            species_mode_inds(1,curr_spec_ind) = 1

            do i_str = 1, len(species_names_tot)
                if (curr_spec_ind > N_cloud_spec) then
                    exit
                end if

                if (species_modes_tot(i_str:i_str) == ':') then
                    species_mode_inds(2,curr_spec_ind) = i_str-1
                    curr_spec_ind = curr_spec_ind+1

                    if (curr_spec_ind <= N_cloud_spec) then
                        species_mode_inds(1,curr_spec_ind) = i_str+1
                    end if
                end if
            end do

            ! Read in cloud densities
            rho_cloud_particles = -1d0
            do i_cloud = 1, N_cloud_spec
                cloud_opa_names(i_cloud) = species_names_tot(&
                    species_name_inds(1,i_cloud):species_name_inds(2,i_cloud)&
                )

                open(unit=10,file=trim(adjustl(path))//'/opa_input_files/cloud_names.dat')
                open(unit=11,file=trim(adjustl(path))//'/opa_input_files/cloud_densities.dat')

                do i_cloud_read = 1, 1000000
                    read(10,*,end=199) cloud_name_buff
                    read(11,*) cloud_dens_buff

                    if (trim(adjustl(cloud_name_buff)) == trim(adjustl(cloud_opa_names(i_cloud)))) then
                        rho_cloud_particles(i_cloud) = cloud_dens_buff
                    end if
                end do

                199 close(10)
                close(11)

                if (rho_cloud_particles(i_cloud) < 0d0) then
                    write(*,*) 'ERROR! DENSITY FOR CLOUD SPECIES '//trim( &
                         adjustl(cloud_opa_names(i_cloud))) &
                         //'NOT FOUND!'
                    stop  ! TODO remove fortran stops and replace with error output
                end if
            end do

            ! Read in cloud opacities
            cloud_specs_abs_opa = 0d0
            cloud_specs_scat_opa = 0d0
            cloud_aniso = 0d0

            open(unit=10,file=trim(adjustl(path))// &
               '/opacities/continuum//clouds/MgSiO3_c/amorphous/mie/bin_borders.dat')
            read(10,*)

            do i_cloud_lamb = 1, N_cloud_rad_bins
                read(10,*) cloud_rad_bins(i_cloud_lamb)
            end do

            read(10,*) cloud_rad_bins(N_cloud_rad_bins+1)
            close(10)

            open(unit=11,file=trim(adjustl(path))// &
               '/opacities/continuum//clouds/MgSiO3_c/amorphous/mie/particle_sizes.dat')
            read(11,*)

            do i_cloud_lamb = 1, N_cloud_rad_bins
                read(11,'(A80)') buff_line
                read(buff_line(17:len(buff_line)),*) cloud_radii(i_cloud_lamb)
            end do

            close(11)

            open(unit=10,file=trim(adjustl(path))// &
               '/opacities/continuum//clouds/MgSiO3_c/amorphous/mie/opa_0001.dat')
            do i_cloud_lamb = 1,11
                read(10,*)
            end do

            do i_cloud_lamb = 1, N_cloud_lambda_bins
                read(10,*) cloud_lambdas(i_cloud_lamb)
                cloud_lambdas(i_cloud_lamb) = cloud_lambdas(i_cloud_lamb) / 1d4
            end do

            close(10)

            do i_cloud = 1, N_cloud_spec

                cloud_opa_mode(i_cloud) = &
                    species_modes_tot(species_mode_inds(1,i_cloud):species_mode_inds(2,i_cloud))

                path_add = trim(adjustl(&
                    cloud_opa_names(i_cloud)(1:len(trim(adjustl(cloud_opa_names(i_cloud))))-3)&
                ))

                if (trim(adjustl( &
                        cloud_opa_names(i_cloud)(len(trim(adjustl( &
                        cloud_opa_names(i_cloud))))-2: &
                        len(trim(adjustl( &
                        cloud_opa_names(i_cloud))))))) == '(c)') then
                    path_add = trim(adjustl(path_add))//'_c'
                else if (trim(adjustl( &
                      cloud_opa_names(i_cloud)(len(trim(adjustl( &
                      cloud_opa_names(i_cloud))))-2: &
                      len(trim(adjustl( &
                      cloud_opa_names(i_cloud))))))) == '(L)') then
                    path_add = trim(adjustl(path_add))//'_L'
                end if

                write(*,*) ' Read in opacity of cloud species ' &
                    //trim(adjustl(path_add(1:len(trim(adjustl(path_add)))-2)))//' ...'

                if (cloud_opa_mode(i_cloud)(1:1) == 'a') then
                    path_add = trim(adjustl(path_add))//'/amorphous'
                else if (cloud_opa_mode(i_cloud)(1:1) == 'c') then
                    path_add = trim(adjustl(path_add))//'/crystalline'
                end if

                if (cloud_opa_mode(i_cloud)(2:2) == 'm') then
                    path_add = trim(adjustl(path_add))//'/mie'
                else if (cloud_opa_mode(i_cloud)(2:2) == 'd') then
                    path_add = trim(adjustl(path_add))//'/DHS'
                    ! Decrease cloud particle density due to porosity
                    rho_cloud_particles(i_cloud) = rho_cloud_particles(i_cloud)*0.75d0
                end if

                open(unit=11,file=trim(adjustl(path))// &
                  '/opacities/continuum//clouds/'//trim(adjustl(path_add))// &
                  '/particle_sizes.dat')
                read(11,*)

                do i_size = 1, N_cloud_rad_bins
                    read(11,'(A80)') buff_line
                    open(unit=10,file=trim(adjustl(path))// &
                         '/opacities/continuum//clouds/'//trim(adjustl(path_add))// &
                         '/'//trim(adjustl(buff_line(1:17))))

                    do i_lamb = 1,11
                        read(10,*)
                    end do

                    do i_lamb = 1, N_cloud_lambda_bins
                        read(10,*) buffer, cloud_specs_abs_opa(i_size,i_lamb,i_cloud), &
                            cloud_specs_scat_opa(i_size,i_lamb,i_cloud), &
                            cloud_aniso(i_size,i_lamb,i_cloud)
                    end do

                    close(10)
                end do

                close(11)
            end do

            write(*,*) 'Done.'
            write(*,*)
        end subroutine load_cloud_opacities


        subroutine load_frequencies(path,spec_name,freq_len,freq,freq_use_ck)
            ! """
            ! Subroutine to read in frequency grid
            ! """
            implicit none
            ! I/O
            character(len=*), intent(in) :: path
            character(len=*), intent(in)  :: spec_name
            integer, intent(in) :: freq_len
            double precision, intent(out) :: freq(freq_len), &
               freq_use_ck(freq_len+1)
            ! internal
            integer :: i_freq, file_unit, freq_len_use_ck
            double precision :: buffer

            ! Because freqs for c-k are stored as borders!
            freq_len_use_ck = freq_len + 1

            open(newunit=file_unit, file=trim(adjustl(path)) // '/opacities/lines/corr_k/' // &
               trim(adjustl(spec_name))//'/kappa_g_info.dat')

            read(file_unit,*)

            do i_freq = 1, freq_len_use_ck - 2
                read(file_unit,*) buffer, freq_use_ck(i_freq)
            end do

            read(file_unit,*) freq_use_ck(freq_len_use_ck), freq_use_ck(freq_len_use_ck-1)
            close(file_unit)

            ! Correct, smallest wlen is slightly offset (not following log-spacing)
            freq_use_ck(1) = freq_use_ck(2) * exp(-log(freq_use_ck(4) / freq_use_ck(3)))
            freq = (freq_use_ck(1:freq_len_use_ck-1) + freq_use_ck(2:freq_len_use_ck)) / 2d0
        end subroutine load_frequencies


        subroutine load_frequencies_g_sizes(path, spec_name, freq_len, g_len)
            ! """
            ! Subroutine to get length of frequency grid in correlated-k mode.
            ! """
            implicit none

            character(len=*), intent(in) :: path
            character(len=*), intent(in)  :: spec_name
            integer, intent(out) :: freq_len, g_len

            integer :: file_unit

            g_len = 1

            open(newunit=file_unit,file=trim(adjustl(path)) // '/opacities/lines/corr_k/' // &
                 trim(adjustl(spec_name)) // '/kappa_g_info.dat')

            read(file_unit, *) freq_len, g_len

            close(file_unit)

            freq_len = freq_len - 1
        end subroutine load_frequencies_g_sizes
        

        subroutine find_line_by_line_frequency_loading_boundaries(wlen_min_read, wlen_max_read, &
                                                                  file_path, arr_len, arr_min)
            ! """
            ! Subroutine to get the length of the opacity arrays in the high-res case.
            ! """
            implicit none

            double precision, intent(in) :: wlen_min_read, wlen_max_read
            character(len=*), intent(in)    :: file_path
            integer, intent(out)         :: arr_len, arr_min

            double precision :: curr_wlen, last_wlen
            integer          :: curr_int, arr_max

            ! open wavelength file
            open(file=trim(adjustl(file_path)), unit=10, form = 'unformatted', &
                ACCESS='stream')

            ! to contain the current wavelength index
            curr_int = 1

            ! to contain the the minimum and the maximum wavelength index
            ! to be used for reading in the opacities and wavelengths later.
            arr_min = -1
            arr_max = -1

            ! to contain the wavelength of the previous line reading
            last_wlen = 0d0

            do while (1>0)
                read(10,end=123) curr_wlen

                if ((curr_int == 1) .AND. (curr_wlen > wlen_min_read)) then
                    write(*,*) 'ERROR! Desired minimum wavelength is too small!'
                    stop  ! TODO remove fortran stops and replace with error output
                end if

                ! look for minimum index, bracketing the desired range
                if (arr_min == -1) then
                    if ((curr_wlen > wlen_min_read) .AND. &
                            (last_wlen < wlen_min_read)) then
                        arr_min = curr_int - 1
                    end if
                end if

                ! look for maximum index, bracketing the desired range
                if (arr_min /= -1) then
                    if ((curr_wlen > wlen_max_read) .AND. &
                            (last_wlen < wlen_max_read)) then
                        arr_max = curr_int
                        exit
                    end if
                end if

                last_wlen = curr_wlen
                curr_int = curr_int + 1
            end do

            123 close(10)

            if ((arr_min == -1) .OR. (arr_max == -1)) then
                write(*,*) 'ERROR! Desired wavelength range is too large,'
                write(*,*) 'or not contained within the tabulated opacity' &
                // ' wavelength range.'
                write(*,*) wlen_min_read, wlen_max_read, curr_wlen
                stop  ! TODO remove fortran stops and replace with error output
            end if

            arr_len = arr_max - arr_min + 1
        end subroutine find_line_by_line_frequency_loading_boundaries


        subroutine load_line_by_line_opacity_grid(arr_min, arr_len, file_path, kappa)
            ! """
            ! Subroutine to read the kappa array in the high-res case.
            ! """
            implicit none

            integer, intent(in)           :: arr_min
            integer, intent(in)           :: arr_len
            character(len=*), intent(in)     :: file_path
            double precision, intent(out) :: kappa(arr_len)

            integer          :: i_lamb

            open(unit=49, file=trim(adjustl(file_path)), &
            form = 'unformatted', ACCESS='stream')

            read(49, pos = (arr_min-1)*8+1) kappa(1)

            do i_lamb = 2, arr_len
                read(49) kappa(i_lamb)
            end do

            close(49)
        end subroutine load_line_by_line_opacity_grid


        subroutine load_line_opacity_grid(path,species_names_tot,freq_len,g_len,species_len,opa_TP_grid_len, &
                                          opa_grid_kappas, mode, arr_min, custom_grid, custom_file_names)
            ! """
            ! Subroutine to read in the molecular opacities (c-k or line-by-line)
            ! """
            implicit none

            character(len=*), intent(in) :: path
            character(len=*), intent(in) :: species_names_tot  !# TODO character lenghts seems unreasonable
            character(len=*), intent(in) :: custom_file_names
            integer, intent(in) :: freq_len,g_len,species_len, opa_TP_grid_len, arr_min
            character(len=*), intent(in) :: mode
            double precision, intent(out) :: opa_grid_kappas(g_len,freq_len,species_len,opa_TP_grid_len)
            logical :: custom_grid

            character(len=2) :: species_id
            character(len=path_string_max_length) :: path_names(opa_TP_grid_len), filename
            character(len=path_string_max_length) :: path_read_stream
            logical :: file_exists
            integer :: species_name_inds(2,species_len)
            integer :: opa_file_names_inds(2,opa_TP_grid_len)
            double precision :: molparam
            integer :: i_spec, i_file, i_str, curr_spec_ind, &
               i_kg, curr_N_g_int, curr_cb_int, curr_file_ind

            ! Get single species names
            curr_spec_ind = 1
            species_name_inds(1,curr_spec_ind) = 1
            do i_str = 1, len(species_names_tot)
                if (curr_spec_ind > species_len) then
                    exit
                end if

                if (species_names_tot(i_str:i_str) == ':') then
                    species_name_inds(2,curr_spec_ind) = i_str-1
                    curr_spec_ind = curr_spec_ind+1

                    if (curr_spec_ind <= species_len) then
                        species_name_inds(1,curr_spec_ind) = i_str+1
                    end if
                end if
            end do

            ! Get opacity file names if defined by user
            if (custom_grid) then
                curr_file_ind = 1 !
                opa_file_names_inds(1,curr_file_ind) = 1
                do i_str = 1, len(custom_file_names)
                    if (curr_file_ind > opa_TP_grid_len) then
                        exit
                    end if

                    if (custom_file_names(i_str:i_str) == ':') then
                        opa_file_names_inds(2,curr_file_ind) = i_str-1
                        curr_file_ind = curr_file_ind+1

                        if (curr_file_ind <= opa_TP_grid_len) then
                            opa_file_names_inds(1,curr_file_ind) = i_str+1
                        end if
                    end if
                end do
            end if

            ! Get paths of opacity files
            if (custom_grid) then
                do i_file = 1, opa_TP_grid_len
                    path_names(i_file) = custom_file_names(opa_file_names_inds(1,i_file): &
                        opa_file_names_inds(2,i_file))
                end do
            else
                open(unit=20,file=trim(adjustl(path))//'/opa_input_files/opa_filenames.txt')

                do i_file = 1, opa_TP_grid_len
                    read(20,*) path_names(i_file)
                end do

                close(20)
            end if

            ! Read opas for every species...
            do i_spec = 1, species_len
                ! Get species file ID and molparam
                if (mode == 'c-k') then
                    filename = trim(adjustl(path))//'/opacities/lines/corr_k/' &
                         //trim(adjustl(species_names_tot(species_name_inds(1,i_spec): &
                         species_name_inds(2,i_spec))))//'/molparam_id.txt'
                else if (mode == 'lbl') then
                    filename = trim(adjustl(path))//'/opacities/lines/line_by_line/' &
                         //trim(adjustl(species_names_tot(species_name_inds(1,i_spec): &
                         species_name_inds(2,i_spec))))//'/molparam_id.txt'
                end if

                inquire(file=trim(filename), exist=file_exists)

                if (.not. file_exists) then  ! put all opacities values to -1 and abort read
                    write(*, '("Cannot open file ''", A, "'': No such file or directory")') trim(filename)

                    opa_grid_kappas = -1d0

                    return
                end if

                open(unit=20,file=filename)

                write(*, '(A)') ' Reading line opacities of species '''//trim(adjustl(&
                    species_names_tot(species_name_inds(1, i_spec):species_name_inds(2,i_spec))))//'''...'
                read(20,*)
                read(20,'(A2)') species_id
                read(20,*)
                read(20,*) molparam
                close(20)

                ! ...for every P-T grid point...
                do i_file = 1, opa_TP_grid_len
                    ! Open opacity file
                    if (mode == 'c-k') then
                        if (custom_grid) then
                            open(unit=20,file=trim(adjustl(path))//'/opacities/lines/corr_k/' &
                                //trim(adjustl(species_names_tot(species_name_inds(1,i_spec): &
                                species_name_inds(2,i_spec))))//'/'// &
                                adjustl(trim(path_names(i_file))), form='unformatted')
                        else
                            open(unit=20,file=trim(adjustl(path))//'/opacities/lines/corr_k/' &
                                //trim(adjustl(species_names_tot(species_name_inds(1,i_spec): &
                                species_name_inds(2,i_spec))))//'/sigma_'//species_id// &
                                adjustl(trim(path_names(i_file))), form='unformatted')
                        end if
                    else if (mode == 'lbl') then
                        if (custom_grid) then
                            path_read_stream =  trim(adjustl(path))//'/opacities/lines/line_by_line/' &
                            //trim(adjustl(species_names_tot(species_name_inds(1,i_spec): &
                            species_name_inds(2,i_spec))))//'/'// &
                            adjustl(trim(path_names(i_file)))
                        else
                            path_read_stream =  trim(adjustl(path))//'/opacities/lines/line_by_line/' &
                               //trim(adjustl(species_names_tot(species_name_inds(1,i_spec): &
                               species_name_inds(2,i_spec))))//'/sigma_'//species_id// &
                               adjustl(trim(path_names(i_file)))
                        end if

                        call load_line_by_line_opacity_grid(arr_min, freq_len, path_read_stream, opa_grid_kappas(1,:,i_spec,i_file))
                    end if

                    ! ...for every frequency point.
                    if (mode == 'c-k') then
                        do i_kg = 1, g_len*freq_len
                            curr_cb_int = (i_kg-1)/g_len+1
                            curr_N_g_int = i_kg - (curr_cb_int-1)*g_len
                            read(20) opa_grid_kappas(curr_N_g_int,curr_cb_int,i_spec,i_file)
                        end do

                        close(20)
                    end if

                end do

                opa_grid_kappas(:,:,i_spec,:) = opa_grid_kappas(:,:,i_spec,:) / molparam
            end do

            write(*, *) 'Done.'
        end subroutine load_line_opacity_grid


        subroutine load_line_by_line_wavelengths(arr_min, arr_len, file_path, wlen)
            ! """
            ! Subroutine to read the wavelength array in the high-res case.
            ! """
            implicit none

            integer, intent(in)           :: arr_min
            integer, intent(in)           :: arr_len
            character(len=*), intent(in)     :: file_path
            double precision, intent(out) :: wlen(arr_len)

            integer          :: i_lamb

            open(unit=49, file=trim(adjustl(file_path)), &
            form = 'unformatted', ACCESS='stream')

            read(49, pos = (arr_min-1)*8+1) wlen(1)

            do i_lamb = 2, arr_len
                read(49) wlen(i_lamb)
            end do

            close(49)
        end subroutine load_line_by_line_wavelengths
end module fortran_inputs