! """
! Utility functions to read, interpolate and mix opacities for the petitRADTRANS radiative transfer package
! """


module fortran_inputs
    implicit none

    integer, parameter :: path_string_max_length = 1024  ! Windows path length soft limit is 256
    integer, parameter :: species_string_max_length = 64  ! longest molelcule formula ("tintin") is 30 characters long

    contains
        subroutine count_file_line_number(file, n_lines)
            ! """
            ! Subroutine to get a opacities array size in the high-res case.
            ! """
            implicit none

            character(len=*), intent(in) :: file
            integer, intent(out) :: n_lines

            integer          :: io
            doubleprecision :: dump

            open(unit=49, file=trim(adjustl(file)), form='unformatted', ACCESS='stream', status='old')

            n_lines = 0

            do
                read(49, iostat=io) dump
                n_lines = n_lines + 1

                if (mod(n_lines, 500000) == 0) then
                    write(*, *) n_lines, dump, io  ! check that we are reading the file and not stalling
                end if

                if(io < 0) then
                    exit
                end if
            end do

            close(49)

            n_lines = n_lines - 1
        end subroutine count_file_line_number


        subroutine find_line_by_line_frequency_loading_boundaries(wavelength_min, wavelength_max, file, &
                                                                  n_frequencies, start_index)
            ! """
            ! Subroutine to get the length of the opacity arrays in the high-res case.
            ! """
            implicit none

            double precision, intent(in) :: wavelength_min, wavelength_max
            character(len=*), intent(in)    :: file
            integer, intent(out)         :: n_frequencies, start_index

            double precision :: curr_wlen, last_wlen
            integer          :: curr_int, arr_max

            ! open wavelength file
            open(file=trim(adjustl(file)), unit=10, form = 'unformatted', &
                ACCESS='stream', status='old')

            ! to contain the current wavelength index
            curr_int = 1

            ! to contain the the minimum and the maximum wavelength index
            ! to be used for reading in the opacities and wavelengths later.
            start_index = -1
            arr_max = -1

            ! to contain the wavelength of the previous line reading
            last_wlen = 0d0

            do while (1>0)
                read(10,end=123) curr_wlen

                if ((curr_int == 1) .AND. (curr_wlen > wavelength_min)) then
                    write(*,*) 'ERROR! Desired minimum wavelength is too small!'
                    stop  ! TODO remove fortran stops and replace with error output
                end if

                ! look for minimum index, bracketing the desired range
                if (start_index == -1) then
                    if ((curr_wlen > wavelength_min) .AND. &
                            (last_wlen < wavelength_min)) then
                        start_index = curr_int - 1
                    end if
                end if

                ! look for maximum index, bracketing the desired range
                if (start_index /= -1) then
                    if ((curr_wlen > wavelength_max) .AND. &
                            (last_wlen < wavelength_max)) then
                        arr_max = curr_int
                        exit
                    end if
                end if

                last_wlen = curr_wlen
                curr_int = curr_int + 1
            end do

            123 close(10)

            if ((start_index == -1) .OR. (arr_max == -1)) then
                write(*,*) 'ERROR! Desired wavelength range is too large,'
                write(*,*) 'or not contained within the tabulated opacity' &
                // ' wavelength range.'
                write(*,*) wavelength_min, wavelength_max, curr_wlen
                stop  ! TODO remove fortran stops and replace with error output
            end if

            n_frequencies = arr_max - start_index + 1
        end subroutine find_line_by_line_frequency_loading_boundaries


        subroutine interpolate_line_opacities(pressures, temperatures, temperature_pressure_grid, &
                                              has_custom_line_opacities_temperature_profile_grid, &
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
            logical, intent(in) :: has_custom_line_opacities_temperature_profile_grid
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
            if (has_custom_line_opacities_temperature_profile_grid) then
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
                if (has_custom_line_opacities_temperature_profile_grid) then
                    ! Search the index in the grids that is the closest to the current layer's pressure and temperature
                    ! TODO is there truly an interest in searching for indices using a dichotomy instead of looping?
                    call find_interpolate_indices(&
                        temperature_grid, n_temperatures, temperatures(i), 1, buffer_scalar_array&
                    )

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

            contains
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
        end subroutine interpolate_line_opacities


        subroutine load_all_line_by_line_opacities(file, n_frequencies, opacities)
            ! """
            ! Subroutine to read all the opacities array in the high-res case.
            ! """
            implicit none

            character(len=*), intent(in)     :: file
            integer, intent(in) :: n_frequencies
            double precision, intent(out) :: opacities(n_frequencies)

            integer          :: i

            open(unit=49, file=trim(adjustl(file)), form = 'unformatted', ACCESS='stream', status='old')

            do i = 1, n_frequencies
                read(49) opacities(i)

                if (mod(i, 500000) == 0) then
                    write(*, '(I0, " / ", I0)') i, n_frequencies  ! check that we are reading the file and not stalling
                end if
            end do

            close(49)
        end subroutine load_all_line_by_line_opacities


        subroutine load_cia_opacities(collision, directory, &
                                      cia_wavelength_grid, cia_temperature_grid, cia_alpha_grid, &
                                      n_temperatures_grid, n_wavelengths_grid)
            ! """
            ! Subroutine to read the CIA opacities.
            ! """
            implicit none

            character(len=*), intent(in)              :: collision
            double precision, intent(out)         :: cia_alpha_grid(10000,50)
            double precision, intent(out)         :: cia_wavelength_grid(10000)
            double precision, intent(out)         :: cia_temperature_grid(50)
            character(len=*), intent(in)             :: directory
            integer,intent(out)                   :: n_temperatures_grid, n_wavelengths_grid

            integer                               :: i,j,stat

            ! ELALEI allowing the tables to have whatever shape
            ! The Python code needs trimmed outputs (n_temperatures_grid, n_wavelengths_grid)
            open(unit=10,file=trim(adjustl(directory))//'/temps.dat', status='old')

            i=0

            do
                i = i + 1

                if (i>50) then
                    exit
                end if

                read(10,*,iostat=stat) cia_temperature_grid(i)

                if (stat /= 0) then
                    exit
                end if
            end do

            close(10)
            n_temperatures_grid=i-1

            open(unit=11,file=trim(adjustl(directory))//'/CIA_'//trim(adjustl(collision))//'_final.dat', status='old')
            read(11,*)
            read(11,*)

            i = 0

            do
                i = i + 1

                if (i>10000) then
                    exit
                end if

                read(11,'(G22.12)',iostat=stat) cia_wavelength_grid(i)

                if (stat /= 0) then
                    exit
                end if
            end do

            close(11)

            open(unit=11,file=trim(adjustl(directory))//'/CIA_'//trim(adjustl(collision))//'_final.dat', status='old')
            read(11,*)
            read(11,*)

            n_wavelengths_grid=i-1

            do i=1,n_wavelengths_grid
                read(11,'(G22.12)',advance='no') cia_wavelength_grid(i)

                do j = 1, n_temperatures_grid-1
                    read(11,'(G22.12)',advance='no') cia_alpha_grid(i,j)
                end do

                read(11,'(G22.12)') cia_alpha_grid(i,n_temperatures_grid)
            end do

            close(11)
        end subroutine load_cia_opacities


        subroutine load_cloud_opacities(path_input_data, path_input_files, path_reference_files, &
                                        all_species, all_iso, all_modes, &
                                        n_clouds, n_cloud_wavelength_bins, &
                                        clouds_particles_densities, clouds_absorption_opacities, &
                                        clouds_scattering_opacities, clouds_asymmetry_parameters, cloud_wavelengths, &
                                        clouds_particles_radii_bins, clouds_particles_radii)
            ! """
            ! Subroutine to read in the molecular opacities (c-k or line-by-line)
            ! """
            implicit none

            integer, parameter :: N_cloud_rad_bins = 130

            character(len=*), intent(in) :: path_input_data
            character(len=*), intent(in) :: path_input_files
            character(len=*), intent(in) :: path_reference_files
            character(len=*), intent(in) :: all_species, all_iso, all_modes
            integer, intent(in) :: n_clouds, n_cloud_wavelength_bins
            double precision, intent(out) :: clouds_particles_densities(n_clouds)
            double precision, intent(out) :: &
                clouds_absorption_opacities(N_cloud_rad_bins,n_cloud_wavelength_bins,n_clouds), &
                clouds_scattering_opacities(N_cloud_rad_bins,n_cloud_wavelength_bins,n_clouds), &
                clouds_asymmetry_parameters(N_cloud_rad_bins,n_cloud_wavelength_bins,n_clouds), &
                cloud_wavelengths(n_cloud_wavelength_bins), &
                clouds_particles_radii_bins(N_cloud_rad_bins+1), clouds_particles_radii(N_cloud_rad_bins)

            integer :: i_cloud, i_cloud_lamb, i_size, i_lamb
            integer :: io
            integer :: species_name_inds(2,n_clouds), species_mode_inds(2,n_clouds), species_iso_inds(2,n_clouds)
            character(len=species_string_max_length)  :: cloud_opa_names(n_clouds), cloud_opa_mode(n_clouds), &
                cloud_name_buff, buff_line
            double precision :: cloud_dens_buff, buffer
            character(len=species_string_max_length * 2) :: cloud_opa_isos(n_clouds), path_add

            ! Get single cloud species names
            call split_string(all_species, n_clouds, species_name_inds)

            ! Get single cloud species modes
            call split_string(all_modes, n_clouds, species_mode_inds)

            ! Get single isotopologue names
            call split_string(all_iso, n_clouds, species_iso_inds)

            ! Load cloud densities
            clouds_particles_densities = -1d0

            do i_cloud = 1, n_clouds
                cloud_opa_names(i_cloud) = all_species(&
                    species_name_inds(1,i_cloud):species_name_inds(2,i_cloud)&
                )
                cloud_opa_isos(i_cloud) = all_iso(&
                    species_iso_inds(1, i_cloud):species_iso_inds(2, i_cloud)&
                )

                open(unit=10, file=trim(adjustl(path_input_files))//'/cloud_names.dat', status='old')
                open(unit=11, file=trim(adjustl(path_input_files))//'/cloud_densities.dat', status='old')

                do
                    read(10, *, iostat=io) cloud_name_buff
                    read(11, *, iostat=io) cloud_dens_buff

                    if (trim(adjustl(cloud_name_buff)) == trim(adjustl(cloud_opa_names(i_cloud)))) then
                        clouds_particles_densities(i_cloud) = cloud_dens_buff
                    end if

                    if(io < 0) then
                        exit
                    end if
                end do

                close(10)
                close(11)

                if (clouds_particles_densities(i_cloud) < 0d0) then
                    write(*, *) 'ERROR! DENSITY FOR CLOUD SPECIES '//trim( &
                         adjustl(cloud_opa_names(i_cloud))) &
                         //'NOT FOUND!'
                    stop  ! TODO remove fortran stops and replace with error output
                end if
            end do

            ! Read in cloud opacities
            clouds_absorption_opacities = 0d0
            clouds_scattering_opacities = 0d0
            clouds_asymmetry_parameters = 0d0

            open(unit=10, file=trim(adjustl(path_reference_files))// &
               '/bin_borders.dat', status='old')
            read(10, *)

            do i_cloud_lamb = 1, N_cloud_rad_bins
                read(10, *) clouds_particles_radii_bins(i_cloud_lamb)
            end do

            read(10, *) clouds_particles_radii_bins(N_cloud_rad_bins+1)
            close(10)

            open(unit=11, file=trim(adjustl(path_reference_files))// &
               '/particle_sizes.dat', status='old')
            read(11, *)

            do i_cloud_lamb = 1, N_cloud_rad_bins
                read(11, '(A80)') buff_line
                read(buff_line(17:len(buff_line)),*) clouds_particles_radii(i_cloud_lamb)
            end do

            close(11)

            open(unit=10, file=trim(adjustl(path_reference_files))// &
               '/opa_0001.dat', status='old')
            do i_cloud_lamb = 1,11
                read(10, *)
            end do

            do i_cloud_lamb = 1, n_cloud_wavelength_bins
                read(10, *) cloud_wavelengths(i_cloud_lamb)
                cloud_wavelengths(i_cloud_lamb) = cloud_wavelengths(i_cloud_lamb) / 1d4
            end do

            close(10)

            do i_cloud = 1, n_clouds
                cloud_opa_mode(i_cloud) = &
                    all_modes(species_mode_inds(1,i_cloud):species_mode_inds(2,i_cloud))

                path_add = trim(adjustl(cloud_opa_isos(i_cloud)))

                if (cloud_opa_mode(i_cloud)(2:2) == 'm') then
                    path_add = trim(adjustl(path_add))//'/mie'
                else if (cloud_opa_mode(i_cloud)(2:2) == 'd') then
                    path_add = trim(adjustl(path_add))//'/DHS'
                    ! Decrease cloud particle density due to porosity
                    clouds_particles_densities(i_cloud) = clouds_particles_densities(i_cloud)*0.75d0
                end if

                write(*, *) ' Loading opacity of cloud species '//trim(adjustl(path_add))//'...'

                open(unit=11,file=trim(adjustl(path_input_data))// &
                  '/'//trim(adjustl(path_add))// &
                  '/particle_sizes.dat', status='old')
                read(11, *)

                do i_size = 1, N_cloud_rad_bins
                    read(11, '(A80)') buff_line
                    open(unit=10, file=trim(adjustl(path_input_data))// &
                         '/'//trim(adjustl(path_add))// &
                         '/'//trim(adjustl(buff_line(1:17))), status='old')

                    do i_lamb = 1,11
                        read(10, *)
                    end do

                    do i_lamb = 1, n_cloud_wavelength_bins
                        read(10, *) buffer, clouds_absorption_opacities(i_size,i_lamb,i_cloud), &
                            clouds_scattering_opacities(i_size,i_lamb,i_cloud), &
                            clouds_asymmetry_parameters(i_size,i_lamb,i_cloud)
                    end do

                    close(10)
                end do

                close(11)
            end do

            write(*, *) 'Done.'
            write(*, *)

            contains
                subroutine split_string(input_string, n_clouds, split_indices)
                    implicit none

                    character(len=*), intent(in) :: input_string
                    integer, intent(in) :: n_clouds
                    integer, intent(out) :: split_indices(2, n_clouds)

                    integer :: i, i_obj

                    i_obj = 1
                    split_indices(1, i_obj) = 1

                    do i = 1, len(input_string)
                        if (i_obj > n_clouds) then
                            exit
                        end if

                        if (input_string(i:i) == ',') then
                            split_indices(2, i_obj) = i - 1
                            i_obj = i_obj + 1

                            if (i_obj <= n_clouds) then
                               split_indices(1, i_obj) = i + 1
                            end if
                        end if
                    end do
                end subroutine split_string
        end subroutine load_cloud_opacities


        subroutine load_cloud_opacities_external(path_to_species_opacity_folder, &
                                        n_clouds, n_cloud_wavelength_bins, &
                                        clouds_absorption_opacities, &
                                        clouds_scattering_opacities, clouds_asymmetry_parameters, cloud_wavelengths, &
                                        clouds_particles_radii_bins, clouds_particles_radii)
            ! """
            ! Subroutine to read in the molecular opacities (c-k or line-by-line)
            ! """
            implicit none

            integer, parameter :: N_cloud_rad_bins = 130

            character(len=*), intent(in) :: path_to_species_opacity_folder
            integer, intent(in) :: n_clouds, n_cloud_wavelength_bins
            double precision, intent(out) :: &
                clouds_absorption_opacities(N_cloud_rad_bins,n_cloud_wavelength_bins,n_clouds), &
                clouds_scattering_opacities(N_cloud_rad_bins,n_cloud_wavelength_bins,n_clouds), &
                clouds_asymmetry_parameters(N_cloud_rad_bins,n_cloud_wavelength_bins,n_clouds), &
                cloud_wavelengths(n_cloud_wavelength_bins), &
                clouds_particles_radii_bins(N_cloud_rad_bins+1), clouds_particles_radii(N_cloud_rad_bins)

            integer :: i_cloud, i_cloud_lamb, i_size, i_lamb

            character(len=species_string_max_length)  :: buff_line
            double precision :: buffer

            ! Read in cloud opacities
            clouds_absorption_opacities = 0d0
            clouds_scattering_opacities = 0d0
            clouds_asymmetry_parameters = 0d0

            open(unit=10, file=trim(adjustl(path_to_species_opacity_folder))// &
               '/bin_borders.dat', status='old')
            read(10, *)

            do i_cloud_lamb = 1, N_cloud_rad_bins
                read(10, *) clouds_particles_radii_bins(i_cloud_lamb)
            end do

            read(10, *) clouds_particles_radii_bins(N_cloud_rad_bins+1)
            close(10)

            open(unit=11, file=trim(adjustl(path_to_species_opacity_folder))// &
               '/particle_sizes.dat', status='old')
            read(11, *)

            do i_cloud_lamb = 1, N_cloud_rad_bins
                read(11, '(A80)') buff_line
                read(buff_line(17:len(buff_line)),*) clouds_particles_radii(i_cloud_lamb)
            end do

            close(11)

            open(unit=10, file=trim(adjustl(path_to_species_opacity_folder))// &
               '/opa_0001.dat', status='old')
            do i_cloud_lamb = 1,11
                read(10, *)
            end do

            do i_cloud_lamb = 1, n_cloud_wavelength_bins
                read(10, *) cloud_wavelengths(i_cloud_lamb)
                cloud_wavelengths(i_cloud_lamb) = cloud_wavelengths(i_cloud_lamb) / 1d4
            end do

            close(10)

            do i_cloud = 1, n_clouds

                write(*, *) ' Loading opacity of cloud species...'

                open(unit=11,file=trim(adjustl(path_to_species_opacity_folder))// &
                  '/particle_sizes.dat', status='old')
                read(11, *)

                do i_size = 1, N_cloud_rad_bins
                    read(11, '(A80)') buff_line
                    open(unit=10, file=trim(adjustl(path_to_species_opacity_folder))// &
                         '/'//trim(adjustl(buff_line(1:17))), status='old')

                    do i_lamb = 1,11
                        read(10, *)
                    end do

                    do i_lamb = 1, n_cloud_wavelength_bins
                        read(10, *) buffer, clouds_absorption_opacities(i_size,i_lamb,i_cloud), &
                            clouds_scattering_opacities(i_size,i_lamb,i_cloud), &
                            clouds_asymmetry_parameters(i_size,i_lamb,i_cloud)
                    end do

                    close(10)
                end do

                close(11)
            end do

            write(*, *) 'Done.'
            write(*, *)

        end subroutine load_cloud_opacities_external


        subroutine load_frequencies(path_input_data, n_frequencies, frequencies, frequency_bins_edges)
            ! """
            ! Subroutine to read in frequency grid
            ! """
            implicit none
            ! I/O
            character(len=*), intent(in) :: path_input_data
            integer, intent(in) :: n_frequencies
            double precision, intent(out) :: frequencies(n_frequencies), &
               frequency_bins_edges(n_frequencies+1)
            ! internal
            integer :: i_freq, file_unit, n_frequencies_use_ck
            double precision :: buffer

            ! Because freqs for c-k are stored as borders!
            n_frequencies_use_ck = n_frequencies + 1

            open(newunit=file_unit, file=trim(adjustl(path_input_data)) // &
                 '/kappa_g_info.dat', status='old')

            read(file_unit,*)

            do i_freq = 1, n_frequencies_use_ck - 2
                read(file_unit,*) buffer, frequency_bins_edges(i_freq)
            end do

            read(file_unit,*) frequency_bins_edges(n_frequencies_use_ck), frequency_bins_edges(n_frequencies_use_ck-1)
            close(file_unit)

            ! Correct, smallest n_wavelengths_grid is slightly offset (not following log-spacing)
            frequency_bins_edges(1) = &
                frequency_bins_edges(2) * exp(-log(frequency_bins_edges(4) / frequency_bins_edges(3)))
            frequencies = &
                (frequency_bins_edges(1:n_frequencies_use_ck-1) + frequency_bins_edges(2:n_frequencies_use_ck)) / 2d0
        end subroutine load_frequencies


        subroutine load_frequencies_g_sizes(path_input_data, n_frequencies, n_g)
            ! """
            ! Subroutine to get length of frequency grid in correlated-k line_opacity_mode.
            ! """
            implicit none

            character(len=*), intent(in) :: path_input_data
            integer, intent(out) :: n_frequencies, n_g

            integer :: file_unit

            n_g = 1

            open(newunit=file_unit,file=trim(adjustl(path_input_data)) // &
                 '/kappa_g_info.dat', status='old')

            read(file_unit, *) n_frequencies, n_g

            close(file_unit)

            n_frequencies = n_frequencies - 1
        end subroutine load_frequencies_g_sizes


        subroutine load_line_opacity_grid(path_input_data, species_directory, &
                                          all_species, n_frequencies, n_g, n_species, &
                                          size_temperature_profile_grid, line_opacity_mode, start_index, &
                                          has_custom_line_opacities_temperature_profile_grid, &
                                          custom_file_names, line_opacities_grid)
            ! """
            ! Subroutine to read in the molecular opacities (c-k or line-by-line)
            ! """
            implicit none

            character(len=*), intent(in) :: path_input_data
            character(len=*), intent(in) :: species_directory
            character(len=*), intent(in) :: all_species
            character(len=*), intent(in) :: custom_file_names
            integer, intent(in) :: n_frequencies, n_g, n_species, size_temperature_profile_grid, start_index
            character(len=*), intent(in) :: line_opacity_mode
            double precision, intent(out) :: &
                line_opacities_grid(n_g, n_frequencies, n_species, size_temperature_profile_grid)
            logical :: has_custom_line_opacities_temperature_profile_grid

            character(len=2) :: species_id
            character(len=path_string_max_length) :: path_names(size_temperature_profile_grid), filename
            character(len=path_string_max_length) :: path_read_stream
            logical :: file_exists
            integer :: species_name_inds(2,n_species)
            integer :: opa_file_names_inds(2,size_temperature_profile_grid)
            double precision :: molparam
            integer :: i_spec, i_file, i_str, curr_spec_ind, &
               i_kg, curr_N_g_int, curr_cb_int, curr_file_ind

            ! Get single species names
            curr_spec_ind = 1
            species_name_inds(1,curr_spec_ind) = 1
            do i_str = 1, len(all_species)
                if (curr_spec_ind > n_species) then
                    exit
                end if

                if (all_species(i_str:i_str) == ':') then
                    species_name_inds(2,curr_spec_ind) = i_str-1
                    curr_spec_ind = curr_spec_ind+1

                    if (curr_spec_ind <= n_species) then
                        species_name_inds(1,curr_spec_ind) = i_str+1
                    end if
                end if
            end do

            ! Get opacity file names if defined by user
            if (has_custom_line_opacities_temperature_profile_grid) then
                curr_file_ind = 1 !
                opa_file_names_inds(1,curr_file_ind) = 1
                do i_str = 1, len(custom_file_names)
                    if (curr_file_ind > size_temperature_profile_grid) then
                        exit
                    end if

                    if (custom_file_names(i_str:i_str) == ':') then
                        opa_file_names_inds(2,curr_file_ind) = i_str-1
                        curr_file_ind = curr_file_ind+1

                        if (curr_file_ind <= size_temperature_profile_grid) then
                            opa_file_names_inds(1,curr_file_ind) = i_str+1
                        end if
                    end if
                end do
            end if

            ! Get paths of opacity files
            if (has_custom_line_opacities_temperature_profile_grid) then
                do i_file = 1, size_temperature_profile_grid
                    path_names(i_file) = custom_file_names(opa_file_names_inds(1,i_file): &
                        opa_file_names_inds(2,i_file))
                end do
            else
                open(unit=20,file=trim(adjustl(path_input_data))//'/opa_filenames.txt', status='old')

                do i_file = 1, size_temperature_profile_grid
                    read(20,*) path_names(i_file)
                end do

                close(20)
            end if

            ! Read opas for every species...
            do i_spec = 1, n_species
                ! Get species file ID and molparam
                if (line_opacity_mode == 'c-k') then
                    filename = trim(adjustl(species_directory))//'/molparam_id.txt'
                else if (line_opacity_mode == 'lbl') then
                    filename = trim(adjustl(species_directory))//'/molparam_id.txt'
                end if

                inquire(file=trim(filename), exist=file_exists)

                if (.not. file_exists) then  ! put all opacities values to -1 and abort read
                    write(*, '("Error: cannot open file ''", A, "'': no such file or directory")') trim(filename)

                    line_opacities_grid = -1d0

                    return
                end if

                open(unit=20,file=filename, status='old')

                write(*, '(A)') ' Reading line opacities of species '''//trim(adjustl(&
                    all_species(species_name_inds(1, i_spec):species_name_inds(2,i_spec))))//'''...'
                read(20,*)
                read(20,'(A2)') species_id
                read(20,*)
                read(20,*) molparam
                close(20)

                ! ...for every P-T grid point...
                do i_file = 1, size_temperature_profile_grid
                    ! Open opacity file
                    if (line_opacity_mode == 'c-k') then
                        if (has_custom_line_opacities_temperature_profile_grid) then
                            open(unit=20,file=trim(adjustl(species_directory))// &
                                '/'//adjustl(trim(path_names(i_file))), form='unformatted', status='old')
                        else
                            open(unit=20,file=trim(adjustl(species_directory))// &
                                '/sigma_'//species_id//adjustl(trim(path_names(i_file))), form='unformatted')
                        end if
                    else if (line_opacity_mode == 'lbl') then
                        if (has_custom_line_opacities_temperature_profile_grid) then
                            path_read_stream = trim(adjustl(species_directory))//'/opacities/lines/line_by_line/'// &
                            '/'//adjustl(trim(path_names(i_file)))
                        else
                            path_read_stream =  trim(adjustl(species_directory))// &
                               '/sigma_'//species_id// adjustl(trim(path_names(i_file)))
                        end if

                        call load_line_by_line_opacity_grid(&
                            start_index, n_frequencies, path_read_stream, line_opacities_grid(1,:,i_spec,i_file)&
                        )
                    end if

                    ! ...for every frequency point.
                    if (line_opacity_mode == 'c-k') then
                        do i_kg = 1, n_g*n_frequencies
                            curr_cb_int = (i_kg-1)/n_g+1
                            curr_N_g_int = i_kg - (curr_cb_int-1)*n_g
                            read(20) line_opacities_grid(curr_N_g_int,curr_cb_int,i_spec,i_file)
                        end do

                        close(20)
                    end if

                end do

                line_opacities_grid(:,:,i_spec,:) = line_opacities_grid(:,:,i_spec,:) / molparam
            end do

            write(*, *) 'Done.'

            contains
                subroutine load_line_by_line_opacity_grid(start_index, n_frequencies, file, opacities)
                    ! """
                    ! Subroutine to read the opacities array in the high-res case.
                    ! """
                    implicit none

                    integer, intent(in)           :: start_index
                    integer, intent(in)           :: n_frequencies
                    character(len=*), intent(in)     :: file
                    double precision, intent(out) :: opacities(n_frequencies)

                    integer          :: i_lamb

                    open(unit=49, file=trim(adjustl(file)), &
                    form = 'unformatted', ACCESS='stream', status='old')

                    read(49, pos = (start_index-1)*8+1) opacities(1)

                    do i_lamb = 2, n_frequencies
                        read(49) opacities(i_lamb)
                    end do

                    close(49)
                end subroutine load_line_by_line_opacity_grid
        end subroutine load_line_opacity_grid


        subroutine load_line_by_line_wavelengths(start_index, n_frequencies, file, n_wavelengths_grid)
            ! """
            ! Subroutine to read the wavelength array in the high-res case.
            ! """
            implicit none

            integer, intent(in)           :: start_index
            integer, intent(in)           :: n_frequencies
            character(len=*), intent(in)     :: file
            double precision, intent(out) :: n_wavelengths_grid(n_frequencies)

            integer          :: i_lamb

            open(unit=49, file=trim(adjustl(file)), &
            form = 'unformatted', ACCESS='stream')

            read(49, pos = (start_index-1)*8+1) n_wavelengths_grid(1)

            do i_lamb = 2, n_frequencies
                read(49) n_wavelengths_grid(i_lamb)
            end do

            close(49)
        end subroutine load_line_by_line_wavelengths
end module fortran_inputs