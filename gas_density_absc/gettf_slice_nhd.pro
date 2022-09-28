pro gettf_slice_nhd, check=check

    ; one (xH, xHe, z, Ein) point
    xH   = [0.5000000000d]
    ; xHe  = [0.5d]
    z    = [50.0000000000d] ; z is actually 1+z (rs)
    ;injE = [5011872336272.715d]
    injE = [1d12]
    ; nhd  = DINDGEN(31)/((31-1)/2.7) - 1 ; nh =equiv= nB is baryon number
    nhd = [-1.0000000000d, -0.9990000000d, -0.9986868241d, -0.9982755692d, -0.9977355191d, -0.9970263383d, -0.9960950593d, -0.9948721262d, -0.9932661999d, -0.9911573363d, -0.9883880276d, -0.9847514382d, -0.9799759568d, -0.9737049100d, -0.9654699228d, -0.9546559364d, -0.9404552706d, -0.9218072991d, -0.8973192332d, -0.8651620964d, -0.8229341209d, -0.7674813631d, -0.6946621405d, -0.5990376957d, -0.4734659837d, -0.3085682437d, -0.0920285132d, 0.1923262322d, 0.5657340177d, 1.0560841052d, 1.7000000000d]
    
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