import numpy as np
import pickle
from tqdm import tqdm

EPSILON = 1e-100

hep_tf  = np.full((5,5,5,500,500), EPSILON)
lep_tf  = np.full((5,5,5,500,500), EPSILON)
lee_tf  = np.full((5,5,5,500,500), EPSILON)
cmbloss = np.full((5,5,5,500,), 0.0)
hed_tf  = np.full((5,5,5,500,4), 0.0)
lowerbound = np.full((5,5,5,), 0.0)

DIR = '../data/tfdata/array/nBs_test'

pbar = tqdm(total=25)

for nBs_i in range(5):
    for x_i in range(5):
        #print(f'nBs_i={nBs_i} x_i={x_i}')
        pbar.update()
        data = pickle.load( open(f'{DIR}/tmp/nBs{nBs_i}_x{x_i}.tfpart', 'rb') )
        hep_tf[nBs_i][x_i] = data['hep_tf']
        lep_tf[nBs_i][x_i] = data['lep_tf']
        lee_tf[nBs_i][x_i] = data['lee_tf']
        cmbloss[nBs_i][x_i] = data['cmbloss']
        hed_tf[nBs_i][x_i] = data['hed_tf']
        lowerbound[nBs_i][x_i] = data['lowerbound']
        
np.save(DIR+'/hep_tf.npy', hep_tf)
np.save(DIR+'/lep_tf.npy', lep_tf)
np.save(DIR+'/lee_tf.npy', lee_tf)
np.save(DIR+'/cmbloss.npy', cmbloss)
np.save(DIR+'/hed_tf.npy', hed_tf)
np.save(DIR+'/lowerbound.npy', lowerbound)