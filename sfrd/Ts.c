
int ComputeTsBox(float redshift, float prev_redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                  struct AstroParams *astro_params, struct FlagOptions *flag_options,
                  float perturbed_field_redshift, short cleanup,
                  struct PerturbedField *perturbed_field, struct TsBox *previous_spin_temp,
                  struct InputHeating *input_heating, struct InputIonization *input_ionization, struct InputJAlpha *input_jalpha,
                  struct InitialConditions *ini_boxes, struct TsBox *this_spin_temp) {

    first_radii = true;
    first_zero = true;
    n_pts_radii = 1000;
    float M_MIN_WDM =  M_J_WDM();
    min_curr_dens = max_curr_dens = 0.;
    fcoll_int_min = fcoll_int_max = 0;
    double LOG10_MTURN_INT = (double) ((LOG10_MTURN_MAX - LOG10_MTURN_MIN)) / ((double) (NMTURN - 1.));
    int table_int_boundexceeded = 0;
    int fcoll_int_boundexceeded = 0;
    for(i=0;i<user_params->N_THREADS;i++) {
        fcoll_int_boundexceeded_threaded[i] = 0;
        table_int_boundexceeded_threaded[i] = 0;
    }
    int NO_LIGHT = 0;
    ION_EFF_FACTOR = global_params.Pop2_ion * astro_params->F_STAR10 * astro_params->F_ESC10;

    ///////////////////////////////  BEGIN INITIALIZATION   //////////////////////////////
    growth_factor_z = dicke(perturbed_field_redshift);
    inverse_growth_factor_z = 1./growth_factor_z;

    //set the minimum ionizing source mass
    // In v1.4 the miinimum ionizing source mass does not depend on redshift.
    // For the constant ionizing efficiency parameter, M_MIN is set to be M_TURN which is a sharp cut-off.
    // For the new parametrization, the number of halos hosting active galaxies (i.e. the duty cycle) is assumed to
    // exponentially decrease below M_TURNOVER Msun, : fduty \propto e^(- M_TURNOVER / M)
    // In this case, we define M_MIN = M_TURN/50, i.e. the M_MIN is integration limit to compute follapse fraction.
    M_MIN = (astro_params->M_TURN)/50.;
    Mlim_Fstar = Mass_limit_bisection(M_MIN, global_params.M_MAX_INTEGRAL, astro_params->ALPHA_STAR, astro_params->F_STAR10);
    Mlim_Fesc = Mass_limit_bisection(M_MIN, global_params.M_MAX_INTEGRAL, astro_params->ALPHA_ESC, astro_params->F_ESC10);
    init_ps();
    init_heat();

    // Flag is set for previous spin temperature box as a previous spin temperature box must be passed, which makes it the initial condition
    x_e_ave = ...
    Tk_ave = ...

    /////////////// Create the z=0 non-linear density fields smoothed on scale R to be used in computing fcoll //////////////
    R = L_FACTOR*user_params->BOX_LEN/(float)user_params->HII_DIM;
    R_factor = pow(global_params.R_XLy_MAX/R, 1/((float)global_params.NUM_FILTER_STEPS_FOR_Ts));

    zp = perturbed_field_redshift*1.0001; //higher for rounding
    while (zp < global_params.Z_HEAT_MAX)
        zp = ((1+zp)*global_params.ZPRIME_STEP_FACTOR - 1);
    prev_zp = global_params.Z_HEAT_MAX;
    zp = ((1+zp)/ global_params.ZPRIME_STEP_FACTOR - 1);

    // This sets the delta_z step for determining the heating/ionisation integrals. If first box is true,
    // it is only the difference between Z_HEAT_MAX and the redshift. Otherwise it is the difference between
    // the current and previous redshift (this behaviour is chosen to mimic 21cmFAST)
    dzp = redshift - prev_redshift;

    determine_zpp_min = perturbed_field_redshift*0.999;

    for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
        if (R_ct==0){
            prev_zpp = zp;
            prev_R = 0;
        }
        else{
            prev_zpp = zpp_edge[R_ct-1];
            prev_R = R_values[R_ct-1];
        }
        zpp_edge[R_ct] = prev_zpp - (R_values[R_ct] - prev_R)*CMperMPC / drdz(prev_zpp); // cell size
        zpp = (zpp_edge[R_ct]+prev_zpp)*0.5; // average redshift value of shell: z'' + 0.5 * dz''
    }

    determine_zpp_max = zpp*1.001;

    if(!flag_options->M_MIN_in_Mass) {
        M_MIN = (float)TtoM(determine_zpp_max, astro_params->X_RAY_Tvir_MIN, mu_for_Ts);
        if(user_params->USE_INTERPOLATION_TABLES) {
            if(user_params->FAST_FCOLL_TABLES){
                initialiseSigmaMInterpTable(fmin(MMIN_FAST,M_MIN),1e20);
            }
            else{
                initialiseSigmaMInterpTable(M_MIN,1e20);
            }
        }
    }

    LOG_SUPER_DEBUG("Initialised sigma interp table");

    zpp_bin_width = (determine_zpp_max - determine_zpp_min)/((float)zpp_interp_points_SFR-1.0);
    dens_width = 1./((double)dens_Ninterp - 1.);

    LOG_SUPER_DEBUG("got density gridpoints");

    /* generate a table for interpolation of the collapse fraction with respect to the X-ray heating, as functions of
        filtering scale, redshift and overdensity.
        Note that at a given zp, zpp values depends on the filtering scale R, i.e. f_coll(z(R),delta).
        Compute the conditional mass function, but assume f_{esc10} = 1 and \alpha_{esc} = 0. */

    for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
        if (R_ct==0){
            prev_zpp = redshift;
            prev_R = 0;
        }
        else{
            prev_zpp = zpp_edge[R_ct-1];
            prev_R = R_values[R_ct-1];
        }
        zpp_edge[R_ct] = prev_zpp - (R_values[R_ct] - prev_R)*CMperMPC / drdz(prev_zpp); // cell size
        zpp = (zpp_edge[R_ct]+prev_zpp)*0.5; // average redshift value of shell: z'' + 0.5 * dz''
        zpp_growth[R_ct] = dicke(zpp);
    }

    initialise_SFRD_Conditional_table(global_params.NUM_FILTER_STEPS_FOR_Ts,min_densities,
                                                max_densities,zpp_growth,R_values, astro_params->M_TURN,
                                                astro_params->ALPHA_STAR, astro_params->F_STAR10, user_params->FAST_FCOLL_TABLES);

    LOG_SUPER_DEBUG("Initialised SFRD table");

    zp = redshift;
    prev_zp = prev_redshift;
    redshift_int_Nion_z = (int)floor( ( zp - determine_zpp_min )/zpp_bin_width );

    if(redshift_int_Nion_z < 0 || (redshift_int_Nion_z + 1) > (zpp_interp_points_SFR - 1)) {
        LOG_ERROR("I have overstepped my allocated memory for the interpolation table Nion_z_val");
        Throw(TableEvaluationError);
    }

    redshift_table_Nion_z = determine_zpp_min + zpp_bin_width*(float)redshift_int_Nion_z;

    Splined_Fcollzp_mean = Nion_z_val[redshift_int_Nion_z] + \
            ( zp - redshift_table_Nion_z )*( Nion_z_val[redshift_int_Nion_z+1] - Nion_z_val[redshift_int_Nion_z] )/(zpp_bin_width);


    if ( ( Splined_Fcollzp_mean < 1e-15 ) && (Splined_Fcollzp_mean_MINI < 1e-15))
        NO_LIGHT = 1;
    else
        NO_LIGHT = 0;
    filling_factor_of_HI_zp = 1 - ( ION_EFF_FACTOR * Splined_Fcollzp_mean + ION_EFF_FACTOR_MINI * Splined_Fcollzp_mean_MINI )/ (1.0 - x_e_ave);

    if (filling_factor_of_HI_zp > 1) filling_factor_of_HI_zp=1;
    // let's initialize an array of redshifts (z'') corresponding to the
    // far edge of the dz'' filtering shells
    // and the corresponding minimum halo scale, sigma_Tmin,
    // as well as an array of the frequency integrals
    LOG_SUPER_DEBUG("beginning loop over R_ct");

    for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
        if (R_ct==0){
            prev_zpp = zp;
            prev_R = 0;
        }
        else{
            prev_zpp = zpp_edge[R_ct-1];
            prev_R = R_values[R_ct-1];
        }

        zpp_edge[R_ct] = prev_zpp - (R_values[R_ct] - prev_R)*CMperMPC / drdz(prev_zpp); // cell size
        zpp = (zpp_edge[R_ct]+prev_zpp)*0.5; // average redshift value of shell: z'' + 0.5 * dz''

        zpp_for_evolve_list[R_ct] = zpp;
        if (R_ct==0){
            dzpp_for_evolve = zp - zpp_edge[0];
        }
        else{
            dzpp_for_evolve = zpp_edge[R_ct-1] - zpp_edge[R_ct];
        }
        zpp_growth[R_ct] = dicke(zpp);

        fcoll_R_array[R_ct] = 0.0;

        // let's now normalize the total collapse fraction so that the mean is the
        // Sheth-Torman collapse fraction
        // Using the interpolated values to update arrays of relevant quanties for the IGM spin temperature calculation

        redshift_int_SFRD = (int)floor( ( zpp - determine_zpp_min )/zpp_bin_width );

        if(redshift_int_SFRD < 0 || (redshift_int_SFRD + 1) > (zpp_interp_points_SFR - 1)) {
            LOG_ERROR("I have overstepped my allocated memory for the interpolation table SFRD_val");
            Throw(TableEvaluationError);
        }

        redshift_table_SFRD = determine_zpp_min + zpp_bin_width*(float)redshift_int_SFRD;

        Splined_SFRD_zpp = SFRD_val[redshift_int_SFRD] + \
                        ( zpp - redshift_table_SFRD )*( SFRD_val[redshift_int_SFRD+1] - SFRD_val[redshift_int_SFRD] )/(zpp_bin_width);

        ST_over_PS[R_ct] = pow(1+zpp, -astro_params->X_RAY_SPEC_INDEX)*fabs(dzpp_for_evolve);
        ST_over_PS[R_ct] *= Splined_SFRD_zpp;
        SFR_timescale_factor[R_ct] = hubble(zpp)*fabs(dtdz(zpp));
        lower_int_limit = fmax(nu_tau_one(zp, zpp, x_e_ave, filling_factor_of_HI_zp), (astro_params->NU_X_THRESH)*NU_over_EV);

        if (filling_factor_of_HI_zp < 0) filling_factor_of_HI_zp = 0; // for global evol; nu_tau_one above treats negative (post_reionization) inferred filling factors properly

        // set up frequency integral table for later interpolation for the cell's x_e value
        for (x_e_ct = 0; x_e_ct < x_int_NXHII; x_e_ct++){
            freq_int_heat_tbl[x_e_ct][R_ct] = integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 0);
            freq_int_ion_tbl[x_e_ct][R_ct] = integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 1);
            freq_int_lya_tbl[x_e_ct][R_ct] = integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 2);

            if(isfinite(freq_int_heat_tbl[x_e_ct][R_ct])==0 || isfinite(freq_int_ion_tbl[x_e_ct][R_ct])==0 || isfinite(freq_int_lya_tbl[x_e_ct][R_ct])==0) {
                LOG_ERROR("One of the frequency interpolation tables has an infinity or a NaN");
                Throw(TableGenerationError);
            }
        }

        // and create the sum over Lya transitions from direct Lyn flux
        sum_lyn[R_ct] = 0;
        
        for (n_ct=NSPEC_MAX; n_ct>=2; n_ct--){
            if (zpp > zmax(zp, n_ct))
                continue;
            nuprime = nu_n(n_ct)*(1+zpp)/(1.0+zp);
            sum_lyn[R_ct] += frecycle(n_ct) * spectral_emissivity(nuprime, 0, global_params.Pop);
        }

        // Find if we need to add a partial contribution to a radii to avoid kinks in the Lyman-alpha flux
        // As we look at discrete radii (light-cone redshift, zpp) we can have two radii where one has a
        // contribution and the next (larger) radii has no contribution. However, if the number of filtering
        // steps were infinitely large, we would have contributions between these two discrete radii
        // Thus, this aims to add a weighted contribution to the first radii where this occurs to smooth out
        // kinks in the average Lyman-alpha flux.

        // Note: We do not apply this correction to the LW background as it is unaffected by this. It is only
        // the Lyn contribution that experiences the kink. Applying this correction to LW introduces kinks
        // into the otherwise smooth quantity
        if(R_ct > 1 && sum_lyn[R_ct]==0.0 && sum_lyn[R_ct-1]>0. && first_radii) {

            // The current zpp for which we are getting zero contribution
            trial_zpp_max = (prev_zpp - (R_values[R_ct] - prev_R)*CMperMPC / drdz(prev_zpp)+prev_zpp)*0.5;
            // The zpp for the previous radius for which we had a non-zero contribution
            trial_zpp_min = (zpp_edge[R_ct-2] - (R_values[R_ct-1] - R_values[R_ct-2])*CMperMPC / drdz(zpp_edge[R_ct-2])+zpp_edge[R_ct-2])*0.5;

            // Split the previous radii and current radii into n_pts_radii smaller radii (redshift) to have fine control of where
            // it transitions from zero to non-zero
            // This is a coarse approximation as it assumes that the linear sampling is a good representation of the different
            // volumes of the shells (from different radii).
            for(ii=0;ii<n_pts_radii;ii++) {
                trial_zpp = trial_zpp_min + (trial_zpp_max - trial_zpp_min)*(float)ii/((float)n_pts_radii-1.);

                counter = 0;
                for (n_ct=NSPEC_MAX; n_ct>=2; n_ct--){
                    if (trial_zpp > zmax(zp, n_ct))
                        continue;
                    counter += 1;
                }
                if(counter==0&&first_zero) {
                    first_zero = false;
                    weight = (float)ii/(float)n_pts_radii;
                }
            }

            // Now add a non-zero contribution to the previously zero contribution
            // The amount is the weight, multplied by the contribution from the previous radii
            sum_lyn[R_ct] = weight * sum_lyn[R_ct-1];
            first_radii = false;
        }

    } // end loop over R_ct filter steps

    freq_int_heat_tbl[x_e_ct][R_ct] = ...
    freq_int_ion_tbl[x_e_ct][R_ct] = ...
    freq_int_lya_tbl[x_e_ct][R_ct] = ...


    fcoll_interp_high_min = global_params.CRIT_DENS_TRANSITION;
    fcoll_interp_high_bin_width = 1./((float)NSFR_high-1.)*(Deltac - fcoll_interp_high_min);
    fcoll_interp_high_bin_width_inv = 1./fcoll_interp_high_bin_width;

    growth_factor_zp = dicke(zp);
    dgrowth_factor_dzp = ddicke_dz(zp);
    dt_dzp = dtdz(zp);

    Luminosity_converstion_factor = ...
    const_zp_prefactor = ...
    //////////////////////////////  LOOP THROUGH BOX //////////////////////////////

    // Extra pre-factors etc. are defined here, as they are independent of the density field,
    // and only have to be computed once per z' or R_ct, rather than each box_ct
    for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
        zpp_integrand = ( pow(1+zp,2)*(1+zpp_for_evolve_list[R_ct]) )/( pow(1+zpp_for_evolve_list[R_ct], -(astro_params->X_RAY_SPEC_INDEX)) );
        dstarlya_dt_prefactor[R_ct]  = zpp_integrand * sum_lyn[R_ct];
    }

    // Can precompute these quantities, independent of the density field (i.e. box_ct)
    freq_int_heat_tbl_diff[i][R_ct] = ...
    freq_int_ion_tbl_diff[i][R_ct] = ...
    freq_int_lya_tbl_diff[i][R_ct] = ...

    LOG_SUPER_DEBUG("looping over box...");

    // Main loop over the entire box for the IGM spin temperature and relevant quantities.
    for (box_ct=0; box_ct<HII_TOT_NUM_PIXELS; box_ct++){
        xHII_call = previous_spin_temp->x_e_box[box_ct];
        //interpolate to correct nu integral value based on the cell's ionization state
        m_xHII_low_box[box_ct] = locate_xHII_index(xHII_call);
        inverse_val_box[box_ct] = (xHII_call - x_int_XHII[m_xHII_low_box[box_ct]])*inverse_diff[m_xHII_low_box[box_ct]];
    }

    for (R_ct=global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct--;){

        Mmax = RtoM(R_values[R_ct]);
        sigmaMmax = sigma_z0(Mmax);

        // fcoll_interp_min
        if( min_densities[R_ct]*zpp_growth[R_ct] <= -1.) {
            fcoll_interp_min = log10(global_params.MIN_DENSITY_LOW_LIMIT);
        }
        else {
            fcoll_interp_min = log10(1+min_densities[R_ct]*zpp_growth[R_ct]);
        }

        // fcoll_interp_bin_width and inv
        if( max_densities[R_ct]*zpp_growth[R_ct] > global_params.CRIT_DENS_TRANSITION ) {
            fcoll_interp_bin_width = 1./((float)NSFR_low-1.)*(log10(1.+global_params.CRIT_DENS_TRANSITION)-fcoll_interp_min);
        }
        else {
            fcoll_interp_bin_width = 1./((float)NSFR_low-1.)*(log10(1.+max_densities[R_ct]*zpp_growth[R_ct])-fcoll_interp_min);
        }
        fcoll_interp_bin_width_inv = 1./fcoll_interp_bin_width;

        for (box_ct=0; box_ct<HII_TOT_NUM_PIXELS; box_ct++) {
            curr_dens = delNL0[R_ct][box_ct]*zpp_growth[R_ct];
            // Now determine all the differentials for the heating/ionisation rate equations
            if (curr_dens < global_params.CRIT_DENS_TRANSITION){
                if (curr_dens <= -1.) {
                    fcoll = 0;
                    fcoll_MINI = 0;
                }
                else {
                    dens_val = (log10f(curr_dens+1.) - fcoll_interp_min)*fcoll_interp_bin_width_inv;
                    fcoll_int = (int)floorf( dens_val );
                    if(fcoll_int < 0 || (fcoll_int + 1) > (NSFR_low - 1)) {
                        if(fcoll_int==(NSFR_low - 1)) {
                            if(fabs(curr_dens - global_params.CRIT_DENS_TRANSITION) < 1e-4) {
                                // There can be instances where the numerical rounding causes it to go in here,
                                // rather than the curr_dens > global_params.CRIT_DENS_TRANSITION case
                                // This checks for this, and calculates f_coll in this instance, rather than causing it to error
                                dens_val = (curr_dens - fcoll_interp_high_min)*fcoll_interp_high_bin_width_inv;
                                fcoll_int = (int)floorf( dens_val );
                                fcoll = SFRD_z_high_table[R_ct][fcoll_int]*( 1. + (float)fcoll_int - dens_val ) + \
                                        SFRD_z_high_table[R_ct][fcoll_int+1]*( dens_val - (float)fcoll_int );
                            }
                            else {
                                fcoll = log10_SFRD_z_low_table[R_ct][fcoll_int];
                                fcoll = expf(fcoll);
                            }
                        }
                        else {
                            fcoll_int_boundexceeded_threaded[omp_get_thread_num()] = 1;
                        }
                    }
                    else {
                        fcoll = log10_SFRD_z_low_table[R_ct][fcoll_int]*( 1 + (float)fcoll_int - dens_val ) + \
                                log10_SFRD_z_low_table[R_ct][fcoll_int+1]*( dens_val - (float)fcoll_int );
                        fcoll = expf(fcoll);
                    }
                }
            }
            else {
                if (curr_dens < 0.99*Deltac) {
                    dens_val = (curr_dens - fcoll_interp_high_min)*fcoll_interp_high_bin_width_inv;
                    fcoll_int = (int)floorf( dens_val );
                    if(fcoll_int < 0 || (fcoll_int + 1) > (NSFR_high - 1)) {
                        fcoll_int_boundexceeded_threaded[omp_get_thread_num()] = 1;
                    }
                    fcoll = SFRD_z_high_table[R_ct][fcoll_int]*( 1. + (float)fcoll_int - dens_val ) + \
                            SFRD_z_high_table[R_ct][fcoll_int+1]*( dens_val - (float)fcoll_int );
                }
                else {
                    fcoll = pow(10.,10.);
                }
            }
            ave_fcoll += fcoll;
            del_fcoll_Rct[box_ct] = (1.+curr_dens)*fcoll;
        }

        Throw(fcoll_int_boundexceeded_threaded)

        ave_fcoll /= (pow(10.,10.)*(double)HII_TOT_NUM_PIXELS);
        ave_fcoll_inv = 1./ave_fcoll;

        dfcoll_dz_val = (ave_fcoll_inv/pow(10.,10.))*ST_over_PS[R_ct]*SFR_timescale_factor[R_ct]/astro_params->t_STAR;
        for (box_ct=0; box_ct<HII_TOT_NUM_PIXELS; box_ct++){

            dxheat_dt_box[box_ct] += ...
            dxion_source_dt_box[box_ct] += (dfcoll_dz_val*(double)del_fcoll_Rct[box_ct]*(
                (freq_int_ion_tbl_diff[m_xHII_low_box[box_ct]][R_ct]) * inverse_val_box[box_ct] + \
                freq_int_ion_tbl[m_xHII_low_box[box_ct]][R_ct]
            ));
            dxlya_dt_box[box_ct] += ...
            dstarlya_dt_box[box_ct] += ...

            // If R_ct == 0, as this is the final smoothing scale (i.e. it is reversed)
            if(R_ct==0) {
                curr_delNL0 = delNL0[0][box_ct];
                // YS DEBUG
                this_spin_temp->x_e_box[box_ct] = ...;
                this_spin_temp->Tk_box[box_ct] = ...;
                TS = ...
            }
        }
    }
}