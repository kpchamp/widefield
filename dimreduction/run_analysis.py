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
X=f.root.data[:,:].T

# note: total frames are 347973
n_features, Tmax = f.root.data.shape
Twin = Tmax
#n_samples = 347972
samples = np.array([30000])
n_folds = 4
ps = np.concatenate(([1],np.arange(25,8200,25)))
ll = np.zeros((samples.size,ps.size))
bic = np.zeros((samples.size,ps.size))
p_threshold = np.zeros((samples.size,))

for i_samples,n_samples in enumerate(samples):
    if (n_samples % n_folds) != 0:
        raise ValueError("number of samples n_samples=%d is not a multiple of n_folds=%d",n_samples,n_folds)
    testSize = n_samples/n_folds
    trainSize = (n_folds-1)*testSize

    perm=np.random.choice(np.arange(Twin),n_samples,replace=False)
    folds=np.reshape(perm,(n_folds,testSize))

    # compute p for full set using singular value thresholding
    U,s,V = la.svd(X[perm,:],full_matrices=False)
    tau = optimal_svht_coef(n_features/n_samples,False)*np.median(s)
    p_threshold[i_samples] = np.where(s<tau)[0][0]-1

    LLs = np.zeros(n_folds,ps.shape)
    BICs = np.zeros(n_folds,ps.shape)
    for i_folds in np.arange(n_folds):
        testSet = folds[i_folds,:]
        trainSet = folds[np.arange(n_folds)[~(np.arange(n_folds) == i_folds)],:].flatten()

        Xtest = X[testSet,:]
        Xtrain = X[trainSet,:]

        ppca = ppca_model(Xtrain)

        for p_idx,p in enumerate(ps):
            LLs[i_folds,p_idx] = ppca.logLikelihood(Xtest,p)
            BICs[i_folds,p_idx] = -2*LLs[i_folds,p_idx]+(n_features*p+1.-0.5*p*(p-1.))*np.log(Xtest.shape[0])


    ll[i_samples,:]=np.mean(LLs,axis=0)
    bic[i_samples,:]=np.mean(BICs,axis=0)

    fout="p_twin%d_nsamples%d.pkl" % (Twin,n_samples)
    pickle.dump({'p_threshold': p_threshold[i_samples], 'll': ll[i_samples,:], 'bic': bic[i_samples,:]},open(fout,'w'))


fout="p_twin%d.pkl" % Twin
pickle.dump({'n_samples': samples, 'p_threshold': p_threshold, 'll': ll, 'bic': bic},open(fout,'w'))
f.close()
