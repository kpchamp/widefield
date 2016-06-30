from __future__ import division

import numpy as np

from widefield.dimreduction.optimal_svht_coef import optimal_svht_coef
from widefield.dimreduction.pca import ppca_model


def pca_select(X, ps, n_folds=4):
    n_samples, n_features = X.shape
    testSize = n_samples/n_folds

    folds=np.reshape(np.arange(n_samples),(n_folds,testSize))

    # compute p for full set using singular value thresholding
    ppca = ppca_model(X)
    s = np.sqrt(ppca.evals*n_samples)
    tau = optimal_svht_coef(n_features/n_samples,False)*np.median(s)
    p_threshold = np.where(s<tau)[0][0]-1

    ll_all = ppca.LLtrain
    ps_all = np.arange(min(n_features,n_samples))+1.
    m = n_features*ps_all+1.-0.5*ps_all*(ps_all-1.)
    aic = -2.*ll_all+m*2.
    bic = -2.*ll_all+m*np.log(n_samples)

    LLs_xval = np.zeros((n_folds,ps.size))
    err_xval = np.zeros((n_folds,))
    for i_folds in np.arange(n_folds):
        testSet = folds[i_folds,:]
        trainSet = folds[np.arange(n_folds)[~(np.arange(n_folds) == i_folds)],:].flatten()

        Xtest = X[testSet,:]
        Xtrain = X[trainSet,:]

        ppca_xval = ppca_model(Xtrain)

        for p_idx,p in enumerate(ps):
            LLs_xval[i_folds,p_idx] = ppca_xval.logLikelihood(Xtest,p)
        err_xval[i_folds] = ps[np.argmax(LLs_xval[i_folds,:])]

    ll_xval=np.mean(LLs_xval,axis=0)

    data = {'ps': ps, 'p_threshold': p_threshold, 'll_all': ll_all, 'svs': s,
            'll_xval': ll_xval, 'err_xval': err_xval, 'bic': bic, 'aic': aic}

    return data, ppca.evecs
