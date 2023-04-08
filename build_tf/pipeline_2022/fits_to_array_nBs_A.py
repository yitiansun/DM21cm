import numpy as np
import pickle
import sys, os
from astropy.io import fits

sys.path.append(os.environ['DH_DIR'])
from darkhistory.spec.spectrum import Spectrum
import darkhistory.physics as phys

sys.path.append('..')
from dm21cm.common import abscs_nBs_test_2, fitsfn

## config

abscs = abscs_nBs_test_2

nBs    = abscs['nBs']
x      = abscs['x']
rs     = abscs['rs']
nrs    = len(rs)
photE  = abscs['photE']
nphotE = len(photE)
elecE  = abscs['elecEk']
nelecE = len(elecE)

injElow_i = np.searchsorted(abscs['photE'], 125) + 1
nBs_i = int(sys.argv[1])
x_i = int(sys.argv[2])
part_i = nBs_i*len(x)+x_i

FITS_DIR = '../data/idl_output/test_nBs_2_tf'
ARRAY_DIR = '../data/tfdata/array/nBs_test_2/tmp'
os.makedirs(ARRAY_DIR, exist_ok=True)

EPSILON = 1e-100

## initialize

hep_tf  = np.full((nrs, nphotE, nphotE), EPSILON)
lep_tf  = np.full((nrs, nphotE, nphotE), EPSILON)
lee_tf  = np.full((nrs, nphotE, nelecE), EPSILON)
cmbloss = np.full((nrs, nphotE, ), 0.0)
hed_tf  = np.full((nrs, nphotE, 4), 0.0)
lowerbound = np.full((nrs,), 0.0)

## extract

print()
print(f'tqdms init {part_i} {int(nrs*(nphotE-injElow_i))}', flush=True)

for rs_i in range(nrs):
    for Ein_i in range(injElow_i, nphotE):
        
        with fits.open(fitsfn(rs[rs_i], np.log10(photE[Ein_i]), x[x_i], nBs[nBs_i], base=FITS_DIR+'/')) as fitsf:
            hep_tf[rs_i][Ein_i] = Spectrum(photE, fitsf[1].data['photonspectrum'][0,1]/2.0, spec_type='dNdE').N
            lep_tf[rs_i][Ein_i] = Spectrum(photE, fitsf[1].data['lowengphot'][0,1]/2.0, spec_type='dNdE').N
            lee_tf[rs_i][Ein_i] = Spectrum(elecE, fitsf[1].data['lowengelec'][0,1]/2.0, spec_type='dNdE').N
            cmbloss[rs_i][Ein_i] = fitsf[1].data['cmblosstable'][0,1]/2.0
            hed_tf[rs_i][Ein_i] = fitsf[1].data['highdeposited_grid'][0,:,1]/2.0
            if Ein_i == nphotE - 1:
                lowerbound[rs_i] = fitsf[1].data['lowerbound'][0,1]
                
        prog = rs_i*(nphotE-injElow_i) + (Ein_i-injElow_i) + 1
        print(f'tqdms {part_i} {int(prog)}', flush=True)

data = {'hep_tf' : hep_tf,
        'lep_tf' : lep_tf,
        'lee_tf' : lee_tf,
        'cmbloss': cmbloss,
        'hed_tf' : hed_tf,
        'lowerbound': lowerbound}

pickle.dump(data, open(f'{ARRAY_DIR}/nBs{nBs_i}_x{x_i}.tfpart', 'wb'))