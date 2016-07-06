import pickle
import numpy as np
import tables as tb
from widefield.dynamics.lds import lds_model

mouseId = 'm187201'
collectionDate = '150727'
basepath = "/suppscr/riekesheabrown/kpchamp/data/"
datapath = basepath + mouseId + "/" + collectionDate + "/data_detrend_mask.h5"
f=tb.open_file(datapath,'r')
X=f.root.data[:,:]
f.close()

# note: total frames are 347973
Tmax, n_features = X.shape
# actually use only first 347904 = 128*2718 frames (for full would be 366000 = 128*2859)
Tmax = 347904
Twin = Tmax/8

lds = lds_model(X[0:100,:].T, 25)    # do only one iteration of EM for timing purposes
#lds.fit_em(X[0:100,:].T, max_iters=1)
lds.fit_constrained(X[0:100,:])
pickle.dump(lds, open('/suppscr/riekesheabrown/kpchamp/lds_model_test_constrained.pkl','w'))