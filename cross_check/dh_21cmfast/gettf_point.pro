PRO gettf_point, check=check
    
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
    
    dlnz = 0.001d
    z_s_global = [6.793051330746054d]
    x_s_global = [0.0000400000d, 0.0101402020d, 0.0202404040d, 0.0303406061d, 0.0404408081d, 0.0505410101d, 0.0606412121d, 0.0707414141d, 0.0808416162d, 0.0909418182d, 0.1010420202d, 0.1111422222d, 0.1212424242d, 0.1313426263d, 0.1414428283d, 0.1515430303d, 0.1616432323d, 0.1717434343d, 0.1818436364d, 0.1919438384d, 0.2020440404d, 0.2121442424d, 0.2222444444d, 0.2323446465d, 0.2424448485d, 0.2525450505d, 0.2626452525d, 0.2727454545d, 0.2828456566d, 0.2929458586d, 0.3030460606d, 0.3131462626d, 0.3232464646d, 0.3333466667d, 0.3434468687d, 0.3535470707d, 0.3636472727d, 0.3737474747d, 0.3838476768d, 0.3939478788d, 0.4040480808d, 0.4141482828d, 0.4242484848d, 0.4343486869d, 0.4444488889d, 0.4545490909d, 0.4646492929d, 0.4747494949d, 0.4848496970d, 0.4949498990d, 0.5050501010d, 0.5151503030d, 0.5252505051d, 0.5353507071d, 0.5454509091d, 0.5555511111d, 0.5656513131d, 0.5757515152d, 0.5858517172d, 0.5959519192d, 0.6060521212d, 0.6161523232d, 0.6262525253d, 0.6363527273d, 0.6464529293d, 0.6565531313d, 0.6666533333d, 0.6767535354d, 0.6868537374d, 0.6969539394d, 0.7070541414d, 0.7171543434d, 0.7272545455d, 0.7373547475d, 0.7474549495d, 0.7575551515d, 0.7676553535d, 0.7777555556d, 0.7878557576d, 0.7979559596d, 0.8080561616d, 0.8181563636d, 0.8282565657d, 0.8383567677d, 0.8484569697d, 0.8585571717d, 0.8686573737d, 0.8787575758d, 0.8888577778d, 0.8989579798d, 0.9090581818d, 0.9191583838d, 0.9292585859d, 0.9393587879d, 0.9494589899d, 0.9595591919d, 0.9696593939d, 0.9797595960d, 0.9898597980d, 0.9999600000d]
    nBs_s_global = [1.0d]
    injection_mode = 'phot'
    channel = 'delta'
    outfolder = '/zfs/yitians/dm21cm/DM21cm/cross_check/ionhist_outputs/'
    
    ; override energy injection
    injE_s = photeng[300:301]
    n_in_eng = 2
    injElow_i = 0
    
    ; paralleling & tqdms
    z_s    = z_s_global
    xH_s   = x_s_global
    nBs_s  = nBs_s_global
    part_total = N_ELEMENTS(z_s) * N_ELEMENTS(xH_s) * N_ELEMENTS(nBs_s) * N_ELEMENTS(injE_s)
    prog_every_n = 1
    prog   = 0
    
    ;---------- Initialize ----------
    ;IF injection_mode EQ 'phot' THEN BEGIN
    ;    injElow_i = VALUE_LOCATE(photeng, 125) ; lowest photon input energy for highengphot is > 125eV
    ;    injE_s    = photeng[injElow_i:*]
    ;    n_in_eng  = nphoteng
    ;    channel = 'delta'
    ;ENDIF ELSE IF injection_mode EQ 'elec' THEN BEGIN
    ;    injE_s    = eleceng
    ;    n_in_eng  = neleceng
    ;    channel = 'elecd'
    ;ENDIF ELSE BEGIN
    ;    MESSAGE, 'Invalid injection_mode setting.'
    ;ENDELSE
    

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
    
    for xH_i   = 0, N_ELEMENTS(xH_s)-1   DO BEGIN
    for nBs_i  = 0, N_ELEMENTS(nBs_s)-1  DO BEGIN
    for z_i    = 0, N_ELEMENTS(z_s)-1    DO BEGIN
    
        ;---------- Initialize tfs ----------
        zinit = z_s[z_i] ; actually 1+z
        xH  = xH_s[xH_i]
        xHe = xH_s[xH_i]
        nBs = nBs_s[nBs_i]
        
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
            /planckparams, /fixedbinning, nphoteng=nphoteng, /heliumseparated, $
            /dontredshiftphotons, /silent

            prog += 1
            IF prog MOD prog_every_n EQ 0 THEN BEGIN
                str  = STRING(prog, format='(I0)')
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