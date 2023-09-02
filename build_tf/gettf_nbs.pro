PRO gettf_nbs, check=check, fixed_cfdt=fixed_cfdt, part_i=part_i, debug=debug, showtimeinfo=showtimeinfo
; fixed_cfdt : flag for fixed conformal delta t
    
    ;---------- Abscissas ----------
    ; photeng
    nphoteng     = 500
    dlnphoteng   = ALOG(5565952217145.328d/1d-4)/nphoteng
    photbins     = 1d-4*EXP(DINDGEN(nphoteng+1)*dlnphoteng)
    photenglow   = photbins[0:nphoteng-1]
    photenghigh  = photbins[1:nphoteng]
    photbinwidth = photenghigh - photenglow
    photeng      = SQRT(photenglow*photenghigh)
    
    ; eleceng
    neleceng     = 500
    dlneleceng   = ALOG(5565952217145.328d)/neleceng
    melec        = 510998.903
    elecbins     = melec + EXP(DINDGEN(neleceng+1)*dlneleceng)
    elecenglow   = elecbins[0:neleceng-1]
    elecenghigh  = elecbins[1:neleceng]
    elecbinwidth = elecenghigh - elecenglow ; bin width [eV]
    eleceng      = melec + SQRT((elecenglow-melec)*(elecenghigh-melec)) ; electron total energies [eV]
    
    ;---------- Config ----------
    ; dlnz and abscissa for nBs, xH=xHe, z(actually 1+z)
    
    IF KEYWORD_SET(debug) THEN BEGIN
        dlnz = 0.001d
        z_s_global = [38.71318413405d]
        x_s_global = [0.0010000000d]
        nBs_s_global = [1.0000000000d]
        injection_mode = 'phot'
        outfolder = '$DM21CM_DIR/build_tf/ionhist_outputs/debug'
    ENDIF ELSE BEGIN
        !EXCEPT = 0 ; turn off underflow error
        dlnz = 0.009950330853168092d
        z_s_global = [5.0000000000d, 5.9183673469d, 6.8367346939d, 7.7551020408d, 8.6734693878d, 9.5918367347d, 10.5102040816d, 11.4285714286d, 12.3469387755d, 13.2653061224d, 14.1836734694d, 15.1020408163d, 16.0204081633d, 16.9387755102d, 17.8571428571d, 18.7755102041d, 19.6938775510d, 20.6122448980d, 21.5306122449d, 22.4489795918d, 23.3673469388d, 24.2857142857d, 25.2040816327d, 26.1224489796d, 27.0408163265d, 27.9591836735d, 28.8775510204d, 29.7959183673d, 30.7142857143d, 31.6326530612d, 32.5510204082d, 33.4693877551d, 34.3877551020d, 35.3061224490d, 36.2244897959d, 37.1428571429d, 38.0612244898d, 38.9795918367d, 39.8979591837d, 40.8163265306d, 41.7346938776d, 42.6530612245d, 43.5714285714d, 44.4897959184d, 45.4081632653d, 46.3265306122d, 47.2448979592d, 48.1632653061d, 49.0816326531d, 50.0000000000d]
        x_s_global = [0.0070000000d, 0.0110000000d]
        nBs_s_global = [1.0000000000d]
        run_name = '230721'
        injection_mode = 'phot'
        outfolder = '$DM21CM_DATA_DIR/tf/'+run_name+'/'+injection_mode+'/ionhist_outputs'
    ENDELSE
    
    
    ; paralleling & tqdms
    ;z_s    = [z_s_global[part_i*5:part_i*5+4]]
    z_s    = z_s_global
    xH_s   = x_s_global
    nBs_s  = nBs_s_global
    part_total = N_ELEMENTS(z_s) * N_ELEMENTS(xH_s) * N_ELEMENTS(nBs_s) * N_ELEMENTS(injE_s)
    prog_every_n = 10
    prog   = 0
    
    ;---------- Initialize ----------
    IF injection_mode EQ 'phot' THEN BEGIN
        injElow_i = VALUE_LOCATE(photeng, 125) ; lowest photon input energy for highengphot is > 125eV
        injE_s    = photeng[injElow_i:*]
        n_in_eng  = nphoteng
        channel = 'delta'
    ENDIF ELSE IF injection_mode EQ 'elec' THEN BEGIN
        injE_s    = eleceng
        n_in_eng  = neleceng
        channel = 'elecd'
    ENDIF ELSE BEGIN
        MESSAGE, 'Invalid injection_mode setting.'
    ENDELSE

    PRINT, '--------------------'
    PRINT, 'Injection mode : ', injection_mode
    PRINT, STRING('xH   start=', xH_s[0]  , ' end=', xH_s[-1]  , ' n_step=', N_ELEMENTS(xH_s)  , format='(A,E0.3,A,E0.3,A,I0)')
    PRINT, STRING('nBs  start=', nBs_s[0] , ' end=', nBs_s[-1] , ' n_step=', N_ELEMENTS(nBs_s) , format='(A,E0.3,A,E0.3,A,I0)')
    PRINT, STRING('z    start=', z_s[0]   , ' end=', z_s[-1]   , ' n_step=', N_ELEMENTS(z_s)   , format='(A,E0.3,A,E0.3,A,I0)')
    PRINT, STRING('injE start=', injE_s[0], ' end=', injE_s[-1], ' n_step=', N_ELEMENTS(injE_s), format='(A,E0.3,A,E0.3,A,I0)')
    PRINT, outfolder
    PRINT, '--------------------'
    
    IF KEYWORD_SET(check) THEN RETURN
    
    PRINT, ''
    PRINT, '' ; preventing printing tqdms line together with idl outputs
    PRINT, STRING('tqdms init ', part_i, ' ', part_total, format='(A,I0,A,I0)')
    
    FOR xH_i  = 0, N_ELEMENTS(xH_s)-1   DO BEGIN
    FOR nBs_i = 0, N_ELEMENTS(nBs_s)-1  DO BEGIN
    FOR z_i   = 0, N_ELEMENTS(z_s)-1    DO BEGIN
    
        ;---------- Initialize tfs ----------
        zinit = z_s[z_i] ; actually 1+z
        xH  = xH_s[xH_i]
        xHe = xH_s[xH_i]
        nBs = nBs_s[nBs_i]

        IF KEYWORD_SET(fixed_cfdt) THEN BEGIN
            Mpc = 3.08568d24 ; [cm]
            c0 = 29979245800d ; [cm/s]
            cfdt = 0.6742 * 1d * Mpc / c0 ; [s]
            H0 = 1d/4.5979401d17 ; [s^-1]
            Omega_M = 0.3175d
            Omega_Lam = 0.6825d
            Omega_rad = 8d-5
            hubblerate = H0 * sqrt(Omega_M*zinit^3 + Omega_rad*zinit^4 + Omega_Lam)
            phys_dt = cfdt / zinit
            dlnz = phys_dt * hubblerate
            PRINT, '\nUsing fixed conformal delta t!\n'
        ENDIF
        
        epsilon = 1d-50
        numsteps = 2
        hep_tf = DBLARR(nphoteng, n_in_eng, numsteps) + epsilon ; IDL row column convention: (out, in, step)
        lep_tf = DBLARR(nphoteng, n_in_eng, numsteps) + epsilon
        lee_tf = DBLARR(neleceng, n_in_eng, numsteps) + epsilon
        hed_tf = DBLARR(4,        n_in_eng, numsteps) + epsilon
        cmbloss = DBLARR(n_in_eng, numsteps)
        lowerbound = DBLARR(numsteps)
        dt = DBLARR(numsteps)
        hubblerate = DBLARR(numsteps)
        
        ;---------- Initialize variables for each tf ----------
        UNDEFINE, tot_time
        UNDEFINE, reuse_electronprocesses
        UNDEFINE, reuse_photon_input_electronprocesses
        
        FOR injE_i = 0, N_ELEMENTS(injE_s)-1 DO BEGIN ; higher injection take a longer time

            injE = injE_s[injE_i]

            ;---------- Call ih_transferfunction ----------
            ih_transferfunction, $
            dlnz=dlnz, zinit=zinit, zfinal=zfinal, $
            numsteps=numsteps, mwimp=injE, channel=channel, $
            customionization=xH, xHe=xHe, $
            nBscale=nBs, $
            ; outfolder=outfolder, $
            output=output, $
            reuse_electronprocesses=reuse_electronprocesses, $
            reuse_photoninput_electronprocesses=reuse_photoninput_electronprocesses, $
            timeinfo=timeinfo, $
            /singleinjection, /altpp, /ionizationdetailed, /comptonsmooth, $
            /modifiedheat, /modifiedion, /depositiondetailed, depositionpartition=3d3, $
            /planckparams,/fixedbinning, nphoteng=nphoteng, /heliumseparated, $
            /dontredshiftphotons, /silent

            prog += 1
            IF prog MOD prog_every_n EQ 0 THEN BEGIN
                str  = STRING('tqdms ', part_i, ' ', prog, format='(A,I0,A,I0)')
                str += STRING(' xH='  , xH    , ': ', xH_i+1  , '/', N_ELEMENTS(xH_s)  , format='(A,E0.3,A,I0,A,I0)')
                str += STRING(' nBs=' , nBs   , ': ', nBs_i+1 , '/', N_ELEMENTS(nBs_s) , format='(A,E0.3,A,I0,A,I0)')
                str += STRING(' zini=', zinit , ': ', z_i+1   , '/', N_ELEMENTS(z_s)   , format='(A,E0.3,A,I0,A,I0)')
                str += STRING(' dlnz=', dlnz  , format='(A,E0.3)')
                str += STRING(' injE=', injE  , ': ', injE_i+1, '/', N_ELEMENTS(injE_s), format='(A,E0.3,A,I0,A,I0)')
                PRINT, str
            ENDIF
            
            ;---------- Save output ----------
            IF injection_mode EQ 'phot' THEN BEGIN
                E_i = injElow_i + injE_i
            ENDIF ELSE IF injection_mode EQ 'elec' THEN BEGIN
                E_i = injE_i
            ENDIF
            
            FOR i_step = 0, numsteps-1 DO BEGIN
                hep_tf[*, E_i, i_step] = output.photonspectrum[*, i_step] / 2d ; dNdE | (out, in, step) <- (out, step) for this 'in'
                lep_tf[*, E_i, i_step] = output.lowengphot[*, i_step] / 2d     ; dNdE | (out, in, step) <- (out, step) for this 'in'
                lee_tf[*, E_i, i_step] = output.lowengelec[*, i_step] / 2d     ; dNdE | (out, in, step) <- (out, step) for this 'in'
                hed_tf[*, E_i, i_step] = output.highdeposited_grid[i_step, *] / 2d ;    (out, in, step) <- (step, out) for this 'in'
                cmbloss[E_i, i_step] = output.cmblosstable[i_step] / 2d ; (in, step) <- (step,) for this 'in'
                lowerbound[i_step] = output.lowerbound[i_step]          ; (step,) <- (step,) same for all 'in'
                dt[i_step] = output.dts[i_step]                         ; (step,) <- (step,) same for all 'in'
                hubblerate[i_step] = output.hubblerate[i_step]          ; (step,) <- (step,) same for all 'in'
            ENDFOR
            
            ;---------- timeinfo ----------
            IF injE_i GE 1 THEN BEGIN
                IF KEYWORD_SET(tot_time) THEN BEGIN
                    tot_time += REFORM(timeinfo.time.TOARRAY())
                ENDIF ELSE BEGIN
                    tot_time = REFORM(timeinfo.time.TOARRAY())
                ENDELSE
            ENDIF

        ENDFOR
        
        ;---------- Save to file ----------
        save_struct = { $
            hep_tf : hep_tf, $
            lep_tf : lep_tf, $
            lee_tf : lee_tf, $
            hed_tf : hed_tf, $
            cmbloss : cmbloss, $
            lowerbound : lowerbound, $
            dt : dt, $
            hubblerate : hubblerate $
        }
        outname = STRING(injection_mode, '_tf_z_', zinit, '_x_', xH, '_nBs_', nBs, $
                         format='(A,A,E0.3,A,E0.3,A,E0.3)')
        outname = outfolder + '/' + outname + '.fits'
        mwrfits, save_struct, outname, /create, /silent ; when saving to fits, (a, b, c, ...) will load to (..., c, b, a) in numpy
        
        IF KEYWORD_SET(showtimeinfo) THEN BEGIN
            PRINT, 'timeinfo:'
            PRINT, REFORM(timeinfo.title.TOARRAY())
            PRINT, tot_time / FLOAT(N_ELEMENTS(injE_s)-1)
        ENDIF
        
    ENDFOR
    ENDFOR
    ENDFOR
    
    RETURN
END