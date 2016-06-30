from __future__ import division
import pickle
import numpy as np
import pandas as pd
import tables as tb
from widefield.dimreduction.model_selection import pca_select

mouseId = 'm187201'
collectionDate = '150810'
basepath = "/suppscr/riekesheabrown/kpchamp/data/"
datapath = basepath + mouseId + "/" + collectionDate + "/data_detrend_mask.h5"
dfpath = basepath + "allData_df_new.pkl"
df = pd.read_pickle(dfpath)
#df = pd.DataFrame()
f=tb.open_file(datapath,'r')
X=f.root.data[:,:]

# note: total frames are 347973
Tmax, n_features = f.root.data.shape
# actually use only first 347904 = 128*2718 frames (for full would be 366000 = 128*2859)
Tmax = 347904
winDiv = 1
Twin = np.int(Tmax/winDiv)
print >>open('output.txt','a'), Twin

for idx in range(winDiv):
    Tstart = idx*Twin
    print >>open('output.txt','a'), Tstart
    samples = np.arange(Twin/4,Twin+1,Twin/4,dtype=np.int)
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
