pro electron_rates, check=check

    rs_s = []
    xH_s = []
    xHe_s = xH_s
    

    ; redshift ; free
    ; photengbins=photenergybins
    nphoteng = 500
    dlnphoteng = alog(5565952217145.328d/1d-4)/nphoteng
    photbins     = 1d-4*exp(dindgen(nphoteng+1)*dlnphoteng)
    photenglow   = photbins[0:nphoteng-1]
    photenghigh  = photbins[1:nphoteng]
    photeng      = sqrt(photenglow*photenghigh) ; photon energies [eV]
    photbinwidth = photeng*dlnphoteng    
    photenergybins = {nbins:nphoteng,
                      bins:photbins,
                      photoneng:photeng,
                      dlnphoteng:dlnphoteng}

    ; elecengbins=elecengbins
    melec = 510998.903d
    neng = 500
    dlneng = alog(5565952217145.328d)/neng
    elecbins = melec + exp(dindgen(neng+1)*dlneng)
    englow =  elecbins[0:neng-1]
    enghigh = elecbins[1:neng]
    eng = melec + sqrt((englow-melec)*(enghigh-melec)) ; electron energies [eV]
    elecbinwidth = enghigh - englow                    ; bin width [eV]
    elecengbins = {nbins:neng,
                   bins:elecbins,
                   eleceng:eng,
                   dlneng:dlneng}
    ; mwimp=0 ; not used!
    ; nh0
    kmperMpc = 3.08568025d19              ; [km in one Mpc]
    amuperg = 6.0221415d23                ; [amu/g]
    G_newt  = 6.6730d-8                   ; [cm^3/g/s^2]
    nh0 = 0.022068*(1d2/kmperMpc)^2*amuperg*3d/(8d*!dpi*G_newt)
    ; YHe
    YHe = 0.24d
    ; electronprocesses is the sole output
    ; depositionpartition set in ih_transferfunction call
    depositionpartition=3d3
    ; icscutoff NOT SET!

    ih_electronprocesses_detailed, redshift=rs[...], photengbins=photenergybins, elecengbins=elecengbins, mwimp=0, Hionfrac=xH[...], Heionfrac=xH[...], nh0=nh0, YHe=YHe, electronprocesses=electronprocesses, lowecutoff=depositionpartition, lowphotcutoff=???, /ionizationdetailed, /depositiondetailed, /modifiedheat, /modifiedion

    electronprocesses =  {
        photonnumfractable : photonnumfractable,
        deposittable : deposittable,
        lowengelecnumfractable : lowengelecnumfractable,
        highdeposittable : highdeposittable,
        lowecutoffval : lowecutoff,
        continuumloss : continuumlosstable,
        highdepgridtable : highdepgridtable
    }