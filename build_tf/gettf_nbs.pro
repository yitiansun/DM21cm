PRO gettf_nbs, $
    check=check, $            ; Dry run to check the input parameters
    i_rs_st=i_rs_st, $        ; Start index for rs=1+z
    i_rs_ed=i_rs_ed, $        ; End index (exclusive)
    i_xx_st=i_xx_st, $        ; Start index for xH=xHe
    i_xx_ed=i_xx_ed, $        ; End index (exclusive)
    i_nB_st=i_nB_st, $        ; Start index for nBscale (1=mean density)
    i_nB_ed=i_nB_ed, $        ; End index (exclusive)
    fixed_cfdt=fixed_cfdt, $  ; Flag for using fixed conformal delta t
    debug=debug, $            ; Flag for debug mode
    showtimeinfo=showtimeinfo ; Flag for showing time info
    

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
    ; set dlnz and abscissa for nBs, xH=xHe, z(actually 1+z)
    
    IF KEYWORD_SET(debug) THEN BEGIN
        dlnz = 0.001d
        rs_s = [38.71318413405d]
        xx_s = [0.0010000000d]
        nB_s = [1.0000000000d]
        inj_mode = 'phot'
        outfolder = '$DM21CM_DIR/build_tf/ionhist_outputs/debug'
    ENDIF ELSE BEGIN
        !EXCEPT = 0 ; turn off underflow error
        dlnz = 0.009950330853168d
        rs_s = [5.0000000000d, 6.4577483251d, 8.3405026860d, 10.7721734502d, 13.9127970110d, 17.9690683190d, 23.2079441681d, 29.9742125159d, 38.7131841341d, 50.0000000000d]
        xx_s = [0.0000100000d, 0.0001000000d, 0.0010000000d, 0.0100000000d, 0.1000000000d, 0.3000000000d, 0.7000000000d, 0.9000000000d, 0.9990000000d, 0.9999900000d]
        nB_s = [0.0010000000d, 0.2620000000d, 0.7080000000d, 0.8180000000d, 0.8880000000d, 0.9440000000d, 1.0060000000d, 1.1020000000d, 4.8550000000d, 10.0000000000d]
        run_name = '230629'
        inj_mode = 'elec'
        outfolder = '$DM21CM_DATA_DIR/tf/'+run_name+'/'+inj_mode+'/ionhist_outputs'
    ENDELSE


    ;---------- Initialize ----------
    IF N_ELEMENTS(i_rs_st) EQ 0 THEN i_rs_st = 0
    IF N_ELEMENTS(i_rs_ed) EQ 0 THEN i_rs_ed = N_ELEMENTS(rs_s)
    IF N_ELEMENTS(i_xx_st) EQ 0 THEN i_xx_st = 0
    IF N_ELEMENTS(i_xx_ed) EQ 0 THEN i_xx_ed = N_ELEMENTS(xx_s)
    IF N_ELEMENTS(i_nB_st) EQ 0 THEN i_nB_st = 0
    IF N_ELEMENTS(i_nB_ed) EQ 0 THEN i_nB_ed = N_ELEMENTS(nB_s)

    IF inj_mode EQ 'phot' THEN BEGIN
        channel = 'delta'
        iE_s    = photeng
        n_iE    = N_ELEMENTS(photeng)
        i_iE_st = VALUE_LOCATE(photeng, 125) ; lowest photon input energy for highengphot is > 125eV
        i_iE_ed = N_ELEMENTS(photeng)
    ENDIF ELSE IF inj_mode EQ 'elec' THEN BEGIN
        channel = 'elecd'
        iE_s    = eleceng
        n_iE    = N_ELEMENTS(eleceng)
        i_iE_st = 0
        i_iE_ed = N_ELEMENTS(eleceng)
    ENDIF ELSE BEGIN
        MESSAGE, 'Invalid inj_mode setting.'
    ENDELSE

    prog_total = (i_rs_ed - i_rs_st) * (i_xx_ed - i_xx_st) * (i_nB_ed - i_nB_st) * (i_iE_ed - i_iE_st)
    prog_show_every_n = 10
    prog = 0

    PRINT, '--------------------'
    PRINT, 'Injection mode : ', inj_mode
    PRINT, STRING('rs : start=', rs_s[i_rs_st], ' end=', rs_s[i_rs_ed-1], ' n_step=', i_rs_ed-i_rs_st, format='(A,E0.3,A,E0.3,A,I0)')
    PRINT, STRING('xx : start=', xx_s[i_xx_st], ' end=', xx_s[i_xx_ed-1], ' n_step=', i_xx_ed-i_xx_st, format='(A,E0.3,A,E0.3,A,I0)')
    PRINT, STRING('nB : start=', nB_s[i_nB_st], ' end=', nB_s[i_nB_ed-1], ' n_step=', i_nB_ed-i_nB_st, format='(A,E0.3,A,E0.3,A,I0)')
    PRINT, STRING('iE : start=', iE_s[i_iE_st], ' end=', iE_s[i_iE_ed-1], ' n_step=', i_iE_ed-i_iE_st, format='(A,E0.3,A,E0.3,A,I0)')
    PRINT, 'output folder : ' + outfolder
    PRINT, '--------------------'
    
    IF KEYWORD_SET(check) THEN RETURN

    ;---------- Loop ----------
    FOR i_xx = i_xx_st, i_xx_ed-1 DO BEGIN
    FOR i_nB = i_nB_st, i_nB_ed-1 DO BEGIN
    FOR i_rs = i_rs_st, i_rs_ed-1 DO BEGIN

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
        numsteps = 2 ; must be 2 for tf generation
        hep_tf = DBLARR(nphoteng, n_iE, numsteps) + epsilon ; IDL row column convention: (out, in, step)
        lep_tf = DBLARR(nphoteng, n_iE, numsteps) + epsilon
        lee_tf = DBLARR(neleceng, n_iE, numsteps) + epsilon
        hed_tf = DBLARR(4,        n_iE, numsteps) + epsilon
        cmbloss = DBLARR(n_iE, numsteps)
        lowerbound = DBLARR(numsteps)
        dt = DBLARR(numsteps)
        hubblerate = DBLARR(numsteps)
        
        ;---------- Initialize variables for each tf ----------
        UNDEFINE, tot_time
        UNDEFINE, reuse_electronprocesses
        UNDEFINE, reuse_photon_input_electronprocesses
        
        FOR i_iE = i_iE_st, i_iE_ed-1 DO BEGIN ; higher injection energy take a longer time

            ;---------- Call ih_transferfunction ----------
            rs = rs_s[i_rs]
            xx = xx_s[i_xx]
            nB = nB_s[i_nB]
            iE = iE_s[i_iE]

            ih_transferfunction, $
                dlnz     = dlnz, $
                zinit    = rs, $ ; zinit is actually 1+z
                numsteps = numsteps, $
                mwimp    = iE, $
                channel  = channel, $
                customionization = xx, $
                xHe      = xx, $
                nBscale  = nB, $
                output   = output, $
                reuse_electronprocesses = reuse_electronprocesses, $
                reuse_photoninput_electronprocesses = reuse_photoninput_electronprocesses, $
                timeinfo = timeinfo, $
                nphoteng = nphoteng, $
                /singleinjection, /altpp, /ionizationdetailed, /comptonsmooth, $
                /modifiedheat, /modifiedion, /depositiondetailed, depositionpartition=3d3, $
                /planckparams,/fixedbinning, /heliumseparated, $
                /dontredshiftphotons, /silent

            prog += 1
            IF prog MOD prog_show_every_n EQ 0 THEN BEGIN
                str =  STRING(FLOAT(prog)/FLOAT(prog_total)*100d, '%  |', format='(F0.2,A)')
                str += STRING('  xx[', i_xx, ']=', xx, format='(A,I0,A,E0.3)')
                str += STRING('  nB[', i_nB, ']=', nB, format='(A,I0,A,E0.3)')
                str += STRING('  rs[', i_rs, ']=', rs, format='(A,I0,A,E0.3)')
                str += STRING('  iE[', i_iE, ']=', iE, format='(A,I0,A,E0.3)')
                PRINT, str
            ENDIF
            
            ;---------- Save output ----------
            FOR i_step = 0, numsteps-1 DO BEGIN
                hep_tf[*, i_iE, i_step] = output.photonspectrum[*, i_step] / 2d ; dNdE | (out, in, step) <- (out, step) for this 'in'
                lep_tf[*, i_iE, i_step] = output.lowengphot[*, i_step] / 2d     ; dNdE | (out, in, step) <- (out, step) for this 'in'
                lee_tf[*, i_iE, i_step] = output.lowengelec[*, i_step] / 2d     ; dNdE | (out, in, step) <- (out, step) for this 'in'
                hed_tf[*, i_iE, i_step] = output.highdeposited_grid[i_step, *] / 2d ;    (out, in, step) <- (step, out) for this 'in'
                cmbloss[i_iE, i_step] = output.cmblosstable[i_step] / 2d ; (in, step) <- (step,) for this 'in'
                lowerbound[i_step] = output.lowerbound[i_step]          ; (step,) <- (step,) same for all 'in'
                dt[i_step] = output.dts[i_step]                         ; (step,) <- (step,) same for all 'in'
                hubblerate[i_step] = output.hubblerate[i_step]          ; (step,) <- (step,) same for all 'in'
            ENDFOR
            
            ;---------- timeinfo ----------
            IF i_iE GE 1 THEN BEGIN
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
        IF KEYWORD_SET(debug) THEN BEGIN
            outname = STRING(inj_mode, 'tf_rs', rs, '_xx', xx, '_nB', nB, format='(A,A,E0.4,A,E0.4,A,E0.4)')
        ENDIF ELSE BEGIN
            outname = STRING(inj_mode, 'tf_rs', i_rs, '_xx', i_xx, '_nB', i_nB, format='(A,A,I0,A,I0,A,I0,A,I0)')
        ENDELSE
        outpath = outfolder + '/' + outname + '.fits'
        mwrfits, save_struct, outpath, /create, /silent ; when saving to fits, (a, b, c, ...) will load to (..., c, b, a) in numpy
        
        IF KEYWORD_SET(showtimeinfo) THEN BEGIN
            PRINT, 'timeinfo:'
            PRINT, REFORM(timeinfo.title.TOARRAY())
            PRINT, tot_time / FLOAT(N_ELEMENTS(iE_s)-1)
        ENDIF
        
    ENDFOR
    ENDFOR
    ENDFOR
    
    RETURN
END