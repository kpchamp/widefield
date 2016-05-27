from sklearn.decomposition import FactorAnalysis
import tables as tb
import numpy as np
import pickle

fname="/gscratch/riekesheabrown/kpchamp/data/m187201_150727_decitranspose_detrend.h5"
f=tb.open_file(fname,'r')
X=f.root.data[:,:].T
# note: total frames are 347973
n_features, Tmax = f.root.data.shape
# actually use only first 347904 = 128*2718 frames
Tmax = 347904
winDiv = 16
Twin = Tmax/winDiv

fa = FactorAnalysis(n_components=1000)
fa.fit(X[:,:])
pickle.dump(fa, open('factoranalysis_test.pkl','w'))
f.close()
