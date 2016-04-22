from __future__ import division
from dimreduction.pca import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from dimreduction.optimal_svht_coef import optimal_svht_coef

#X=np.load("/Users/kpchamp/Dropbox (uwamath)/backup/research/python/data/testdata_small.npy")
X=np.load("/Users/kpchamp/Dropbox (uwamath)/backup/research/python/data/circledata2d.npy")
Xtrain=X[0:1000,:]
Xtest=X[1000:2000,:]
#pca=pca_model(X,n_components=25)
#pcaEM=pca_model(X,n_components=25,fitWith='EM',max_iters=1000)
ppca=ppca_model(Xtrain)
LL=ppca.logLikelihood(Xtest,100)

#pca_sklearn=PCA(n_components='mle')
#pca_sklearn.fit(X)

tau = optimal_svht_coef(400/1000,False)
print tau
