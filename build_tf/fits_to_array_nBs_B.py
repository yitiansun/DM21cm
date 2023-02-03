import numpy as np
import sys
import pickle
from tqdm import tqdm

sys.path.append('..')
from dm21cm.common import abscs_nBs_test_2

abscs = abscs_nBs_test_2

interp_shape = (len(abscs['nBs']), len(abscs['x']), len(abscs['rs']))

EPSILON = 1e-100

print('initializing hep_tf...')
hep_tf  = np.full(interp_shape + (len(abscs['photE']), len(abscs['photE'])), EPSILON)
print('initializing lep_tf...')
lep_tf  = np.full(interp_shape + (len(abscs['photE']), len(abscs['photE'])), EPSILON)
print('initializing lee_tf...')
lee_tf  = np.full(interp_shape + (len(abscs['photE']), len(abscs['elecEk'])), EPSILON)
print('initializing cmbloss...')
cmbloss = np.full(interp_shape + (len(abscs['photE']),), 0.0)
print('initializing hed_tf...')
hed_tf  = np.full(interp_shape + (len(abscs['photE']), len(abscs['dep_c'])-1), 0.0)
print('initializing lowerbound...')
lowerbound = np.full(interp_shape, 0.0)

DIR = '../data/tfdata/array/nBs_test_2'

pbar = tqdm(total=interp_shape[0]*interp_shape[1])

for nBs_i in range(interp_shape[0]):
    for x_i in range(interp_shape[1]):
        data = pickle.load( open(f'{DIR}/tmp/nBs{nBs_i}_x{x_i}.tfpart', 'rb') )
        hep_tf[nBs_i][x_i] = data['hep_tf']
        lep_tf[nBs_i][x_i] = data['lep_tf']
        lee_tf[nBs_i][x_i] = data['lee_tf']
        cmbloss[nBs_i][x_i] = data['cmbloss']
        hed_tf[nBs_i][x_i] = data['hed_tf']
        lowerbound[nBs_i][x_i] = data['lowerbound']
        pbar.update()
        
np.save(DIR+'/hep_tf.npy', hep_tf)
np.save(DIR+'/lep_tf.npy', lep_tf)
np.save(DIR+'/lee_tf.npy', lee_tf)
np.save(DIR+'/cmbloss.npy', cmbloss)
np.save(DIR+'/hed_tf.npy', hed_tf)
np.save(DIR+'/lowerbound.npy', lowerbound)