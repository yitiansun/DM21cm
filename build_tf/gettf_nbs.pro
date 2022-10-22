pro gettf_nbs, check=check, fixed_cfdt=fixed_cfdt, part_nbs_i=part_nbs_i, part_x_i=part_x_i
; fixed_cfdt : flag for fixed conformal delta t

    ; test abscissa for nBs, xH=xHe, z(actually 1+z)
    nBs_s_global = [0.0000000000d, 0.6750000000d, 1.3500000000d, 2.0250000000d, 2.7000000000d]
    x_s_global = [0.0000100000d, 0.2500050000d, 0.5000000000d, 0.7499950000d, 0.9999900000d]
    z_s = [5.0000000000d, 8.8913970502d, 15.8113883008d, 28.1170662595d, 50.0000000000d]
    
    ; photeng
    nphoteng    = 500
    dlnphoteng  = alog(5565952217145.328d/1d-4)/nphoteng
    photbins    = 1d-4*exp(dindgen(nphoteng+1)*dlnphoteng)
    photenglow  = photbins[0:nphoteng-1]
    photenghigh = photbins[1:nphoteng]
    injE        = sqrt(photenglow*photenghigh)
    injElow_i   = value_locate(injE, 125) ; lowest photon input energy for highengphot is > 125eV
    injE_s      = injE[injElow_i:*]
    
    ; part & tqdms
    part_i = part_nbs_i * 5 + part_x_i
    nBs_s = [nBs_s_global[part_nbs_i]]
    xH_s = [x_s_global[part_x_i]]
    
    ; config
    if n_elements(fixed_cfdt) NE 0 then begin
        Mpc = 3.08568d24 ; cm
        c0 = 29979245800d ; cm/s
        cfdt = 0.6742 * 1d * Mpc / c0 ; s
    endif
    channel = 'delta'
    outfolder = '/zfs/yitians/dm21cm/data/idl_output/test_nBs_tf/'
    
    ; Planck parameters
    H0 = 1d/4.5979401d17 ; s^-1
    Omega_M = 0.3175d
    Omega_Lam = 0.6825d
    Omega_rad = 8d-5

    print, '--------------------'
    print, string('xH   start=', xH_s[0]  , ' end=', xH_s[-1]  , ' n_step=', n_elements(xH_s)  , format='(A,E0.3,A,E0.3,A,I0)')
    print, string('nBs  start=', nBs_s[0] , ' end=', nBs_s[-1] , ' n_step=', n_elements(nBs_s) , format='(A,E0.3,A,E0.3,A,I0)')
    print, string('z    start=', z_s[0]   , ' end=', z_s[-1]   , ' n_step=', n_elements(z_s)   , format='(A,E0.3,A,E0.3,A,I0)')
    print, string('injE start=', injE_s[0], ' end=', injE_s[-1], ' n_step=', n_elements(injE_s), format='(A,E0.3,A,E0.3,A,I0)')
    print, outfolder
    print, '--------------------'
    
    if keyword_set(check) then return
    
    part_total = n_elements(z_s) * n_elements(injE_s)
    print, string('tqdms init ', part_i, ' ', part_total, format='(A,I0,A,I0)')
    
    ; block should apply for redshift or xe, since the larger masses take a longer time.
    for xH_i   = 0, n_elements(xH_s)-1   do begin
    for nBs_i  = 0, n_elements(nBs_s)-1  do begin
    for z_i    = 0, n_elements(z_s)-1    do begin
    for injE_i = 0, n_elements(injE_s)-1 do begin
    
        ; calculate variables
        zinit = z_s[z_i] ; actually 1+z
        injE = injE_s[injE_i]
        xH  = xH_s[xH_i]
        xHe = xH_s[xH_i]
        nBs = nBs_s[nBs_i]
        
        if n_elements(fixed_cfdt) EQ 0 then begin
            dlnz = 1d-3
        endif else begin
            hubblerate = H0 * sqrt(Omega_M*zinit^3 + Omega_rad*zinit^4 + Omega_Lam) ;inverse timescale for redshifting
            phys_dt = cfdt / zinit
            dlnz = phys_dt * hubblerate
        endelse
    
        ih_transferfunction, $
        dlnz=dlnz, zinit=zinit, zfinal=zfinal, $
        numsteps=2, mwimp=injE, channel=channel, $
        customionization=xH, xHe=xHe, $
        nBscale=nBs, $
        outfolder=outfolder, $
        /singleinjection, /altpp, /ionizationdetailed, /comptonsmooth, /modifiedheat, /modifiedion, /depositiondetailed, depositionpartition=3d3, /planckparams,/fixedbinning, nphoteng=nphoteng, /heliumseparated, /dontredshiftphotons, /silent
        
        prog = z_i*n_elements(injE_s) + injE_i + 1
        str  = string('tqdms ', part_i, ' ', prog, format='(A,I0,A,I0)')
        
        str += string(' xH='  , xH    , ': ', xH_i+1  , '/', n_elements(xH_s)  , format='(A,E0.3,A,I0,A,I0)')
        str += string(' nBs=' , nBs   , ': ', nBs_i+1 , '/', n_elements(nBs_s) , format='(A,E0.3,A,I0,A,I0)')
        str += string(' zini=', zinit , ': ', z_i+1   , '/', n_elements(z_s)   , format='(A,E0.3,A,I0,A,I0)')
        str += string(' dlnz=', dlnz  , format='(A,E0.3)')
        str += string(' injE=', injE  , ': ', injE_i+1, '/', n_elements(injE_s), format='(A,E0.3,A,I0,A,I0)')
        print, str
        
        

    endfor
    endfor
    endfor
    endfor
    
    return
end