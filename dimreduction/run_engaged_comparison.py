from __future__ import division
import tables as tb
import numpy as np
import pickle
import pandas as pd
from widefield.dimreduction.model_selection import pca_select

mouseId = 'm187201'
collectionDate = '150727'
basepath = "/gscratch/riekesheabrown/kpchamp/data/"
datapath = basepath + mouseId + "/" + collectionDate + "/data_detrend_mask.h5"
dfpath = basepath + "df_engaged_comparison.pkl"
df = pd.read_pickle(dfpath)
f = tb.open_file(datapath, 'r')
X = f.root.data[:, :]

de_idxs = pickle.load(open(basepath + mouseId + "/" + collectionDate + "/idxs.pkl", 'r'))
num_disengaged = de_idxs[0].shape[0]
num_engaged = de_idxs[1].shape[0]

# take the number of samples to be number of disengaged frames, rounded to be divisible by 4 for cross-val
Tmax, n_features = f.root.data.shape
n_samples = np.int(np.floor(num_disengaged / 4) * 4)
n_folds = 4
ps = np.concatenate(([1], np.arange(25, n_features, 25)))  # values of p to evaluate in cross validation
if (n_samples % n_folds) != 0:
    raise ValueError("number of samples n_samples=%d is not a multiple of n_folds=%d", n_samples, n_folds)

# run disengaged analysis
print >>open('output.txt','a'), "starting disengaged analysis"
frames = de_idxs[0][0:n_samples]
dfrow = {'mouseId': mouseId, 'date': collectionDate, 'sampleSize': n_samples, 'chunkNumber': 1,
         'engagement': 'D', 'frames': frames}
dfrow['data'], evecs = pca_select(X[frames, :], ps, n_folds=n_folds)
pickle.dump(evecs, open(basepath + mouseId + "/" + collectionDate + "/evecs_de/" + "evecs_D.pkl", 'w'))
df = df.append(dfrow, ignore_index=True)
df.to_pickle(dfpath)

# run engaged analysis
startIdxs = np.arange(0, num_engaged, num_disengaged, dtype=np.int)
for idx in startIdxs.shape[0]:
    print >>open('output.txt','a'), "starting engaged analysis, chunk %d" % (idx+1)
    startIdx = startIdxs[idx]

    frames = de_idxs[1][startIdx:(startIdx + n_samples)]
    dfrow = {'mouseId': mouseId, 'date': collectionDate, 'sampleSize': n_samples, 'chunkNumber': idx + 1,
             'engagement': 'E', 'frames': frames}

    dfrow['data'], evecs = pca_select(X[frames, :], ps, n_folds=n_folds)

    fout = "evecs_E_%d.pkl" % (idx + 1)
    pickle.dump(evecs, open(basepath + mouseId + "/" + collectionDate + "/evecs_de/" + fout, 'w'))

    df = df.append(dfrow, ignore_index=True)
    df.to_pickle(dfpath)

f.close()
