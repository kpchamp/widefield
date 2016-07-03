from sklearn.decomposition import FactorAnalysis
import tables as tb
import numpy as np
import pickle
import pandas as pd

mouseId = 'm187201'
collectionDate = '150727'
basepath = "/suppscr/riekesheabrown/kpchamp/data/"
datapath = basepath + mouseId + "/" + collectionDate + "/data_detrend_mask.h5"
dfpath = basepath + "allData_df.pkl"
df = pd.read_pickle(dfpath)
f=tb.open_file(datapath,'r')
X=f.root.data[:,:]
f.close()

# note: total frames are 347973
Tmax, n_features = X.shape
# actually use only first 347904 = 128*2718 frames
Tmax = 347904
Twin = Tmax
n_samples = Tmax/2
Tstart = 0

perm = df.loc[(df['sampleSize']==n_samples) & (df['windowLength']==Twin)]['data']['perm']
#perm=np.random.choice(np.arange(Twin),n_samples,replace=False)+Tstart
fa = FactorAnalysis(n_components=1000)
fa.fit(X[perm,:])
pickle.dump(fa, open('factoranalysis_test.pkl','w'))
