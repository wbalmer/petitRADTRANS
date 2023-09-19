module fortran_chemistry
    implicit none

    contains
        subroutine read_dat_chemical_table(n_metallicities, n_co_ratios, n_pressures, n_temperatures, n_species, &
                                           path_input_data, mass_fractions_table)
            implicit none
            ! I/O
            integer, intent(in) :: n_metallicities, n_co_ratios, n_pressures, n_temperatures, n_species
            character(len=500), intent(in) :: path_input_data
            double precision, intent(out) :: &
                mass_fractions_table(n_species, n_temperatures, n_pressures, n_co_ratios, n_metallicities)
            ! Internal
            integer :: i_feh, i_co, i_p, i_t

            mass_fractions_table = 0d0

            open(unit=13,file=trim(adjustl(path_input_data)) // &
                'abundance_files/abunds_python.dat',form='unformatted')

            do i_feh = 1, n_metallicities
                do i_co = 1, n_co_ratios
                    do i_p = 1, n_pressures
                        do i_t = 1, n_temperatures
                            read(13) mass_fractions_table(1:n_species,i_t,i_p,i_co,i_feh)
                        end do
                    end do
                end do
            end do

            ! Do not powerlaw interpolate MMW and nabla_ad, only the abundances!
            ! Use linear interpolation for the former instead.
            mass_fractions_table(1:n_species-2,:,:,:,:) = log10(mass_fractions_table(1:n_species-2,:,:,:,:))

            close(13)
        end subroutine read_dat_chemical_table


        subroutine interpolate_mass_fractions_table(co_ratios, log10_metallicities, temperatures, pressures, &
                                                    largest_co_ratios_indices, largest_metallicities_indices, &
                                                    largest_temperatures_indices, largest_pressures_indices, &
                                                    table_metallicities, table_co_ratios, table_pressures, &
                                                    table_temperatures, mass_fractions_table, &
                                                    n_layers, n_metallicities, n_co_ratios, n_pressures, &
                                                    n_temperatures, n_species, mass_fractions)

            implicit none
            ! I/O
            integer, intent(in) :: n_metallicities, n_co_ratios, n_pressures, n_temperatures, n_species, n_layers
            double precision, intent(in) :: co_ratios(n_layers), log10_metallicities(n_layers), &
               temperatures(n_layers), pressures(n_layers)
            integer, intent(in) :: largest_co_ratios_indices(n_layers), largest_metallicities_indices(n_layers), &
               largest_temperatures_indices(n_layers), largest_pressures_indices(n_layers)
            double precision, intent(in) :: table_co_ratios(n_co_ratios), table_metallicities(n_metallicities), &
               table_temperatures(n_temperatures), table_pressures(n_pressures)
            double precision, intent(in) :: &
                mass_fractions_table(n_species, n_temperatures, n_pressures, n_co_ratios, n_metallicities)
            double precision, intent(out) :: mass_fractions(n_species, n_layers)
            ! internal
            integer :: i_goal, i_p, i_t, i_feh, i_co, i_p_take, i_t_take, i_feh_take, &
               i_co_take, i_spec
            double precision :: intp_bound(n_species, 2, 2, 2, 2), &
               intp_coords(4,2), intp_bound_m1(n_species, 2, 2, 2), &
               intp_bound_m2(n_species, 2, 2), intp_bound_m3(n_species, 2)

            do i_goal = 1, n_layers
                do i_feh = 1,2
                    i_feh_take = largest_metallicities_indices(i_goal)-mod(i_feh,2)

                    do i_co = 1,2
                        i_co_take = largest_co_ratios_indices(i_goal)-mod(i_co,2)

                       do i_p = 1,2
                           i_p_take = largest_pressures_indices(i_goal)-mod(i_p,2)

                           do i_t = 1,2
                               i_t_take = largest_temperatures_indices(i_goal)-mod(i_t,2)

                               intp_bound(1:n_species, i_t, i_p, i_co, i_feh) = &
                                   mass_fractions_table(1:n_species, i_t_take, i_p_take, i_co_take, i_feh_take)

                               intp_coords(1, i_t) = table_temperatures(i_t_take)
                           end do
                           intp_coords(2, i_p) = table_pressures(i_p_take)
                       end do

                       intp_coords(3, i_co) = table_co_ratios(i_co_take)
                    end do

                    intp_coords(4, i_feh) = table_metallicities(i_feh_take)
                end do

                ! Interpolate to correct FEH
                intp_bound_m1(1:n_species, 1:2, 1:2, 1:2) = intp_bound(1:n_species, 1:2, 1:2, 1:2, 1) + &
                    (intp_bound(1:n_species, 1:2, 1:2, 1:2, 2) - intp_bound(1:n_species, 1:2, 1:2, 1:2, 1)) / &
                    (intp_coords(4,2) - intp_coords(4,1)) * (log10_metallicities(i_goal) - intp_coords(4,1))

                ! Interpolate to correct C/O
                intp_bound_m2(1:n_species, 1:2, 1:2) = intp_bound_m1(1:n_species, 1:2, 1:2, 1) + &
                    (intp_bound_m1(1:n_species, 1:2, 1:2, 2) - intp_bound_m1(1:n_species, 1:2, 1:2, 1)) / &
                    (intp_coords(3,2) - intp_coords(3,1)) * (co_ratios(i_goal) - intp_coords(3,1))

                ! Interpolate to correct pressure
                intp_bound_m3(1:n_species, 1:2) = intp_bound_m2(1:n_species, 1:2, 1) + &
                    (intp_bound_m2(1:n_species, 1:2, 2) - intp_bound_m2(1:n_species, 1:2, 1)) / &
                    (intp_coords(2,2) - intp_coords(2,1)) * (pressures(i_goal) - intp_coords(2,1))

                ! Interpolate to correct temperature
                mass_fractions(1:n_species,i_goal) = intp_bound_m3(1:n_species, 1) + &
                    (intp_bound_m3(1:n_species, 2) - intp_bound_m3(1:n_species, 1)) / &
                    (intp_coords(1,2) - intp_coords(1,1)) * (temperatures(i_goal) - intp_coords(1,1))

                do i_spec = 1, n_species
                    if (ISNAN(mass_fractions(i_spec,i_goal))) then
                        mass_fractions(i_spec,i_goal) = -50d0
                    end if
                end do
            end do

            mass_fractions(1:n_species-2,:) = 1d1**mass_fractions(1:n_species-2,:)
        end subroutine interpolate_mass_fractions_table
end module fortran_chemistry
