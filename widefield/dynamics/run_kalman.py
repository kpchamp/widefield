import pickle
import numpy as np
import scipy.linalg as la
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

#lds = lds_model(n_dim_obs=n_features, n_dim_state=300)
#U, _, _ = la.svd(X[0:10000,:].T, full_matrices=False)
#lds.C = U[:,0:300]
#print >>open('progress.txt','a'), "SVD dims %d,%d" % lds.C.shape
lds2 = pickle.load(open('/suppscr/riekesheabrown/kpchamp/lds_model_startSVD_12iter.pkl','r'))
lds = lds_model(A=lds2.A, C=lds2.C, Q=lds2.Q, R=lds2.R, mu0=lds2.mu0, V0=lds2.V0)
lds.fit_em(X[0:10000,:].T, max_iters=8)   # do only one iteration of EM for timing purposes
#lds.fit_constrained(X[0:100,:].T)
pickle.dump(lds, open('/suppscr/riekesheabrown/kpchamp/lds_model_20iter.pkl','w'))
#result = {}
#result['mu'], result['V'], _ = lds.kalman_smoothing(X[10000:11000,:].T)
#pickle.dump(result, open('kalman_smooth_result.pkl','w'))
