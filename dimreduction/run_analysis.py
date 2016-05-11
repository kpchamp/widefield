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
# actually use only first 347904 = 128*2718 frames
Tmax = 347904
Twin = np.int(Tmax/16)
Tstart = 15*Twin
samples = np.arange(Tmax/32,Twin+1,Tmax/32,dtype=np.int)
n_folds = 4
ps = np.concatenate(([1],np.arange(25,8200,25)))
p_threshold = np.zeros((samples.size,))
p_xval = np.zeros((samples.size,))
p_bic = np.zeros((samples.size,))
p_aic = np.zeros((samples.size,))

for i_samples,n_samples in enumerate(samples):
    if (n_samples % n_folds) != 0:
        raise ValueError("number of samples n_samples=%d is not a multiple of n_folds=%d",n_samples,n_folds)
    testSize = n_samples/n_folds
    trainSize = (n_folds-1)*testSize

    perm=np.random.choice(np.arange(Twin),n_samples,replace=False)+Tstart
    folds=np.reshape(perm,(n_folds,testSize))
    print >>open('output.txt','a'), np.min(perm), np.max(perm)

    # compute p for full set using singular value thresholding
    ppca = ppca_model(X[perm,:])
    #U,s,V = la.svd(X[perm,:],full_matrices=False)
    s = np.sqrt(ppca.evals*n_samples)
    tau = optimal_svht_coef(n_features/n_samples,False)*np.median(s)
    p_threshold[i_samples] = np.where(s<tau)[0][0]-1

    lltrain = ppca.LLtrain
    ps_all=np.arange(min(n_features,n_samples))+1.
    m=n_features*ps_all+1.-0.5*ps_all*(ps_all-1.)
    aic = -2.*lltrain+m*2.
    bic = -2.*lltrain+m*np.log(n_samples)
    p_bic[i_samples]=np.argmin(bic)+1
    p_aic[i_samples]=np.argmin(aic)+1

    LLs_test = np.zeros((n_folds,ps.size))
    for i_folds in np.arange(n_folds):
        testSet = folds[i_folds,:]
        trainSet = folds[np.arange(n_folds)[~(np.arange(n_folds) == i_folds)],:].flatten()

        Xtest = X[testSet,:]
        Xtrain = X[trainSet,:]

        ppca = ppca_model(Xtrain)

        for p_idx,p in enumerate(ps):
            LLs_test[i_folds,p_idx] = ppca.logLikelihood(Xtest,p)

    lltest=np.mean(LLs_test,axis=0)
    p_xval[i_samples]=ps[np.argmax(lltest)]

    fout="p_twin%d_nsamples%d_16.pkl" % (Twin,n_samples)
    pickle.dump({'ps': ps, 'p_threshold': p_threshold[i_samples], 'lltrain': lltrain, 'svs': s,
                 'lltest': lltest, 'bic': bic, 'aic': aic, 'perm': perm}, open(fout,'w'))


fout="p_twin%d_16.pkl" % Twin
pickle.dump({'n_samples': samples, 'p_threshold': p_threshold, 'p_xval': p_xval, 'p_aic': p_aic, 'p_bic': p_bic},open(fout,'w'))
f.close()
