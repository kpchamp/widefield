from __future__ import division
from pca import ppca_model
import tables as tb
import numpy as np
import scipy.linalg as la
from optimal_svht_coef import optimal_svht_coef
import pickle

fname="/gscratch/riekesheabrown/kpchamp/data/m187201_150727_decitranspose_detrend.h5"
#fname="/Users/kpchamp/Dropbox (uwamath)/backup/research/python/data/20150727_detrend.h5"
f=tb.open_file(fname,'r')


# note: total frames are 347973
q, Tmax = f.root.data.shape
Twin = Tmax
#n_samples = 347972
n_samples = 30000
n_folds = 4
if (n_samples % n_folds) != 0:
    raise ValueError("number of samples n_samples=%d is not a multiple of n_folds=%d",n_samples,n_folds)
testSize = n_samples/n_folds
trainSize = (n_folds-1)*testSize

fout="p_twin%d.pkl" % Twin

perm=np.random.choice(np.arange(Twin),n_samples,replace=False)
folds=np.reshape(perm,(n_folds,testSize))

# compute p for full set using singular value thresholding
U,s,V = la.svd(f.root.data[:,perm].T,full_matrices=False)
tau = optimal_svht_coef(q/n_samples,False)*np.median(s)
p_threshold = np.where(s<tau)[0][0]-1

p_ll=np.zeros((n_folds,))
for i in np.arange(n_folds):
    testSet = folds[i,:]
    trainSet = folds[np.arange(n_folds)[~(np.arange(n_folds) == i)],:].flatten()

    ppca = ppca_model(f.root.data[:,trainSet].T)
    ps = np.arange(1,4000,50)
    LLs = np.zeros(ps.shape)
    for p_idx,p in enumerate(ps):
        LLs[p_idx] = ppca.logLikelihood(f.root.data[:,testSet].T,p)

    p_ll[i]=ps[np.argmax(LLs)]


pickle.dump({'p_threshold': p_threshold, 'p_ll': p_ll},open(fout,'w'))
