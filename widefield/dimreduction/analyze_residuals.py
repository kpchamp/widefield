import numpy as np
from scipy.stats import skew, kurtosis

from widefield.dimreduction.pca import ppca_model


def get_residual_and_moments(n_components, X, bias=True):
    # mouseId = 'm187201'
    # collectionDate = '150727'
    # basepath = "/gscratch/riekesheabrown/kpchamp/data/"
    # datapath = basepath + mouseId + "/" + collectionDate + "/data_detrend_mask.h5"
    # f=tb.open_file(datapath,'r')
    # X=np.log(f.root.data[:,:])
    # f.close()

    ppca = ppca_model(X, n_components=n_components)
    print "number of components is %d" % ppca.components.shape[1]
    Xnew = ppca.reconstruct(X)
    R = X - Xnew
    del X
    del Xnew

    print "getting moments"
    moments = np.zeros((R.shape[1],4))
    for i in range(R.shape[1]):
        moments[i,0] = np.mean(R[:,i])
        moments[i,1] = np.std(R[:,i])**2
        moments[i,2] = skew(R[:,i], bias=bias)
        moments[i,3] = kurtosis(R[:,i], fisher=True, bias=bias)

    return R,moments


def get_moments(R, bias=True):
    moments = np.zeros((R.shape[1],4))
    for i in range(R.shape[1]):
        moments[i,0] = np.mean(R[:,i])
        moments[i,1] = np.std(R[:,i])**2
        moments[i,2] = skew(R[:,i], bias=bias)
        moments[i,3] = kurtosis(R[:,i], fisher=True, bias=bias)
    return moments