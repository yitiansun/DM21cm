PRO gettf_nbs, check=check, fixed_cfdt=fixed_cfdt, part_i=part_i, debug=debug
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
        dlnz = 0.04879016d
        z_s_global = [5.0000000000d]
        x_s_global = [0.0000100000d]
        nBs_s_global = [0.0010000000d]
    ENDIF ELSE BEGIN
        !EXCEPT = 0 ; turn off underflow error
        dlnz = 0.04879016d
        z_s_global = [5.0000000000d, 6.4577483251d, 8.3405026860d, 10.7721734502d, 13.9127970110d, 17.9690683190d, 23.2079441681d, 29.9742125159d, 38.7131841341d, 50.0000000000d]
        x_s_global = [0.0000100000d, 0.0001000000d, 0.0010000000d, 0.0100000000d, 0.1000000000d, 0.3000000000d, 0.7000000000d, 0.9000000000d, 0.9990000000d, 0.9999900000d]
        nBs_s_global = [0.0010000000d, 0.2620000000d, 0.7080000000d, 0.8180000000d, 0.8880000000d, 0.9440000000d, 1.0060000000d, 1.1020000000d, 4.8550000000d, 10.0000000000d]
    ENDELSE
    
    run_name = '230408'
    injection_mode = 'elec'
    outfolder = '/zfs/yitians/dm21cm/DM21cm/data/tf/'+run_name+'/'+injection_mode+'/ionhist_output'
    
    ; paralleling & tqdms
    z_s    = [z_s_global[part_i]]
    xH_s   = x_s_global
    nBs_s  = nBs_s_global
    part_total = N_ELEMENTS(z_s) * N_ELEMENTS(xH_s) * N_ELEMENTS(nBs_s) * N_ELEMENTS(injE_s)
    prog_every_n = 11
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
    
    for xH_i   = 0, N_ELEMENTS(xH_s)-1   DO BEGIN
    for nBs_i  = 0, N_ELEMENTS(nBs_s)-1  DO BEGIN
    for z_i    = 0, N_ELEMENTS(z_s)-1    DO BEGIN
    
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
        hep_tf = DBLARR(nphoteng, n_in_eng) + epsilon ; IDL row column convention
        lep_tf = DBLARR(nphoteng, n_in_eng) + epsilon
        lee_tf = DBLARR(neleceng, n_in_eng) + epsilon
        hed_tf  = DBLARR(4, n_in_eng)
        cmbloss = DBLARR(n_in_eng)
        lowerbound = 0d
        
        ;---------- Initialize variables for each tf ----------
        UNDEFINE, tot_time
        UNDEFINE, reuse_electronprocesses
        UNDEFINE, reuse_photon_input_electronprocesses
        
        FOR injE_i = 0, N_ELEMENTS(injE_s)-1 DO BEGIN ; higher injection take a longer time
        
            injE = injE_s[injE_i]

            ;---------- Call ih_transferfunction ----------
            ih_transferfunction, $
            dlnz=dlnz, zinit=zinit, zfinal=zfinal, $
            numsteps=2, mwimp=injE, channel=channel, $
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
            
            hep_tf[*, E_i] = output.photonspectrum[*, 1] / 2d ; dNdE
            lep_tf[*, E_i] = output.lowengphot[*, 1] / 2d ; dNdE
            lee_tf[*, E_i] = output.lowengelec[*, 1] / 2d ; dNdE
            hed_tf[*, E_i] = output.highdeposited_grid[1, *] / 2d
            cmbloss[E_i] = output.cmblosstable[1] / 2d
            lowerbound = output.lowerbound[1]
            
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
            lowerbound : lowerbound $
        }
        outname = STRING('tf_z_', zinit, '_x_', xH, '_nBs_', nBs, $
                         format='(A,E0.3,A,E0.3,A,E0.3)')
        outname = outfolder + '/' + outname + '.fits'
        mwrfits, save_struct, outname, /create, /silent
        
        PRINT, 'timeinfo:'
        PRINT, REFORM(timeinfo.title.TOARRAY())
        PRINT, tot_time / FLOAT(N_ELEMENTS(injE_s)-1)
        
    ENDFOR
    ENDFOR
    ENDFOR
    
    RETURN
END