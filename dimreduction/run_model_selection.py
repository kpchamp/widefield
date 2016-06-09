from __future__ import division
import tables as tb
import numpy as np
import pickle
import pandas as pd
from widefield.dimreduction.model_selection import pca_select


mouseId = 'm1877931'
collectionDate = '150729'
basepath = "/gscratch/riekesheabrown/kpchamp/data/"
datapath = basepath + mouseId + "/" + collectionDate + "/data_detrend_mask.h5"
dfpath = basepath + "allData_df_new.pkl"
#df = pd.read_pickle(dfpath)
df = pd.DataFrame()
f=tb.open_file(datapath,'r')
X=f.root.data[:,:]

# note: total frames are 347973
n_features, Tmax = f.root.data.shape
# actually use only first 347904 = 128*2718 frames (for full would be 366000 = 128*2859)
Tmax = 347904
winDiv = 1
Twin = np.int(Tmax/winDiv)
print >>open('output.txt','a'), Twin

for idx in range(winDiv):
    Tstart = idx*Twin
    print >>open('output.txt','a'), Tstart
    samples = np.arange(Twin,Twin+1,Twin/2,dtype=np.int)
    n_folds = 4
    ps = np.concatenate(([1],np.arange(25,8200,25)))

    for i_samples,n_samples in enumerate(samples):
        dfrow = {'mouseId': mouseId, 'date': collectionDate, 'windowLength': Twin, 'startTime': Tstart,
                 'sampleSize': n_samples}
        print >>open('output.txt','a'), n_samples
        if (n_samples % n_folds) != 0:
            raise ValueError("number of samples n_samples=%d is not a multiple of n_folds=%d",n_samples,n_folds)

        perm=np.random.choice(np.arange(Twin),n_samples,replace=False)+Tstart

        fout="p_twin%d_nsamples%d_%d.pkl" % (Twin,n_samples,idx+1)
        dfrow['data'],evecs = pca_select(X[perm,:], ps, n_folds=n_folds)

        fout="evecs_twin%d_nsamples%d_tstart%d.pkl" % (Twin,n_samples,Tstart)
        pickle.dump(evecs, open(basepath + mouseId + "/" + collectionDate + "/evecs/" + fout,'w'))

        fout="p_twin%d_nsamples%d_%d.pkl" % (Twin,n_samples,idx+1)
        dfrow['data']['perm'] = perm
        pickle.dump(dfrow['data'], open(fout,'w'))
        df = df.append(dfrow, ignore_index=True)
        df.to_pickle(dfpath)

f.close()

# mouseId = 'm187201'
# collectionDate = '150805'
# basepath = "/gscratch/riekesheabrown/kpchamp/data/"
# datapath = basepath + mouseId + "/" + collectionDate + "/transpose_detrend.h5"
# dfpath = basepath + "allData_df.pkl"
# df = pd.read_pickle(dfpath)
# f=tb.open_file(datapath,'r')
# X=f.root.data[:,:].T
#
# # note: total frames are 347973
# n_features, Tmax = f.root.data.shape
# # actually use only first 347904 = 128*2718 frames
# Tmax = 347904
# winDiv = 2
# Twin = np.int(Tmax/winDiv)
#
# for idx in range(winDiv):
#     Tstart = idx*Twin
#     print >>open('output.txt','a'), Tstart
#     samples = np.arange(Twin/4,Twin+1,Twin/4,dtype=np.int)
#     n_folds = 4
#     ps = np.concatenate(([1],np.arange(25,8200,25)))
#     p_threshold = np.zeros((samples.size,))
#     p_xval = np.zeros((samples.size,))
#     p_bic = np.zeros((samples.size,))
#     p_aic = np.zeros((samples.size,))
#
#     for i_samples,n_samples in enumerate(samples):
#         dfrow = {'mouseId': mouseId, 'date': collectionDate, 'windowLength': Twin, 'startTime': Tstart,
#                  'sampleSize': n_samples}
#         print >>open('output.txt','a'), n_samples
#         if (n_samples % n_folds) != 0:
#             raise ValueError("number of samples n_samples=%d is not a multiple of n_folds=%d",n_samples,n_folds)
#         testSize = n_samples/n_folds
#         trainSize = (n_folds-1)*testSize
#
#         perm=np.random.choice(np.arange(Twin),n_samples,replace=False)+Tstart
#         folds=np.reshape(perm,(n_folds,testSize))
#         print >>open('output.txt','a'), np.min(perm), np.max(perm)
#
#         # compute p for full set using singular value thresholding
#         ppca = ppca_model(X[perm,:])
#         s = np.sqrt(ppca.evals*n_samples)
#         tau = optimal_svht_coef(n_features/n_samples,False)*np.median(s)
#         p_threshold[i_samples] = np.where(s<tau)[0][0]-1
#
#         ll_all = ppca.LLtrain
#         ps_all=np.arange(min(n_features,n_samples))+1.
#         m=n_features*ps_all+1.-0.5*ps_all*(ps_all-1.)
#         aic = -2.*ll_all+m*2.
#         bic = -2.*ll_all+m*np.log(n_samples)
#         p_bic[i_samples]=np.argmin(bic)+1
#         p_aic[i_samples]=np.argmin(aic)+1
#
#         fout="evecs_twin%d_nsamples%d_tstart%d.pkl" % (Twin,n_samples,Tstart)
#         pickle.dump(ppca.evecs, open(basepath + mouseId + "/" + collectionDate + "/evecs/" + fout,'w'))
#
#         LLs_xval = np.zeros((n_folds,ps.size))
#         err_xval = np.zeros((n_folds,))
#         for i_folds in np.arange(n_folds):
#             testSet = folds[i_folds,:]
#             trainSet = folds[np.arange(n_folds)[~(np.arange(n_folds) == i_folds)],:].flatten()
#
#             Xtest = X[testSet,:]
#             Xtrain = X[trainSet,:]
#
#             ppca = ppca_model(Xtrain)
#
#             for p_idx,p in enumerate(ps):
#                 LLs_xval[i_folds,p_idx] = ppca.logLikelihood(Xtest,p)
#             err_xval[i_folds] = ps[np.argmax(LLs_xval[i_folds,:])]
#
#         ll_xval=np.mean(LLs_xval,axis=0)
#         p_xval[i_samples]=ps[np.argmax(ll_xval)]
#
#         fout="p_twin%d_nsamples%d_%d.pkl" % (Twin,n_samples,idx+1)
#         dfrow['data'] = {'ps': ps, 'p_threshold': p_threshold[i_samples], 'll_all': ll_all, 'svs': s,
#                      'll_xval': ll_xval, 'err_xval': err_xval, 'bic': bic, 'aic': aic, 'perm': perm}
#         pickle.dump(dfrow['data'], open(fout,'w'))
#         df = df.append(dfrow, ignore_index=True)
#         df.to_pickle(dfpath)
#
#     f.close()
