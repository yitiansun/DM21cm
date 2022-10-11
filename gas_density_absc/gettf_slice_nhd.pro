PRO gettf_slice_nhd, check=check

    ; one (xH, xHe, z, Ein) point
    xH   = [0.5000000000d]
    ; xHe  = [0.5d]
    z    = [50.0000000000d] ; z is actually 1+z (rs)
    ; injE
    single_injE = 0
    if single_injE EQ 1 then begin
        injE = [1d5]
    endif else begin
        nphoteng    = 500
        dlnphoteng  = alog(5565952217145.328d/1d-4)/nphoteng
        photbins    = 1d-4*exp(dindgen(nphoteng+1)*dlnphoteng)
        photenglow  = photbins[0:nphoteng-1]
        photenghigh = photbins[1:nphoteng]
        injE        = sqrt(photenglow*photenghigh)
        injElow_i   = value_locate(injE, 125) ; lowest photon input energy for highengphot is > 125eV
        injE        = injE[injElow_i:*]
    endelse
    ; nhd
    ;nhd = DINDGEN(31)/((31-1)/2.7) - 1 ; nh =equiv= nB is baryon number
    nhd = [-1.0000000000d]
    ;nhd = [-1.0000000000d, -0.9990000000d, -0.9975941827d, -0.9942120432d, -0.9860752335d, -0.9664995561d, -0.9194040531d, -0.8061008779d, -0.5335141409d, 0.1222797421d, 1.7000000000d]
    
    dlnz = 1d-3
    nphoteng = 500
    channel = 'delta'
    outfolder = '/zfs/yitians/21cm_inhomogeneity/data/idl_output/'

    print, '--------------------'
    print, string('xH   start=', xH[0]  , ' end=', xH[-1]  , ' n_step=', n_elements(xH)  , format='(A,E0.3,A,E0.3,A,I0)')
    ; print, string('xHe  start=', xHe[0] , ' end=', xHe[-1] , ' n_step=', n_elements(xHe) , format='(A,E0.3,A,E0.3,A,I0)')
    print, string('nhd  start=', nhd[0] , ' end=', nhd[-1] , ' n_step=', n_elements(nhd) , format='(A,E0.3,A,E0.3,A,I0)')
    print, string('z    start=', z[0]   , ' end=', z[-1]   , ' n_step=', n_elements(z)   , format='(A,E0.3,A,E0.3,A,I0)')
    print, string('injE start=', injE[0], ' end=', injE[-1], ' n_step=', n_elements(injE), format='(A,E0.3,A,E0.3,A,I0)')
    print, outfolder
    print, '--------------------'
    
    if keyword_set(check) then return
    
    ; block should apply for redshift or xe, since the larger masses take a longer time.
    for xH_i   = 0, n_elements(xH)-1   do begin
    ; for xHe_i  = 0, n_elements(xHe)-1  do begin
    for nhd_i  = 0, n_elements(nhd)-1  do begin
    for z_i    = 0, n_elements(z)-1    do begin
    for injE_i = 0, n_elements(injE)-1 do begin
    
        xHe = xH[xH_i]
    
        ih_transferfunction, $
        dlnz=dlnz, zinit=z[z_i], zfinal=zfinal, $
        numsteps=4, mwimp=injE[injE_i], channel=channel, $
        customionization=xH[xH_i], xHe=xHe, $
        nhdelta=nhd[nhd_i], $
        outfolder=outfolder, $
        /singleinjection, /altpp, /ionizationdetailed, /comptonsmooth, /modifiedheat, /modifiedion, /depositiondetailed, depositionpartition=3d3, /planckparams,/fixedbinning, nphoteng=nphoteng, /heliumseparated, /dontredshiftphotons, /silent
        
        str  = string(' xH='  , xH[xH_i]    , ': ', xH_i+1  , '/', n_elements(xH)  , format='(A,E0.3,A,I0,A,I0)')
        ;str += string(' xHe=' , xHe[xHe_i]  , ': ', xHe_i+1 , '/', n_elements(xHe) , format='(A,E0.3,A,I0,A,I0)')
        str += string(' nhd=' , nhd[nhd_i]  , ': ', nhd_i+1 , '/', n_elements(nhd) , format='(A,E0.3,A,I0,A,I0)')
        str += string(' zini=', z[z_i]      , ': ', z_i+1   , '/', n_elements(z)   , format='(A,E0.3,A,I0,A,I0)')
        str += string(' injE=', injE[injE_i], ': ', injE_i+1, '/', n_elements(injE), format='(A,E0.3,A,I0,A,I0)')
        print, str

    ; endfor
    endfor
    endfor
    endfor
    endfor
    
    return
end