from dimreduction.pca import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#X=np.load("/Users/kpchamp/Dropbox (uwamath)/backup/research/python/data/testdata_small.npy")
X=np.load("/Users/kpchamp/Dropbox (uwamath)/backup/research/python/data/circledata2d.npy")
Xtrain=X[0:200,:]
Xtest=X[200:400,:]
#pca=pca_model(X,n_components=25)
#pcaEM=pca_model(X,n_components=25,fitWith='EM',max_iters=1000)
ppca=ppca_model(Xtrain,n_components=5)
LL=ppca.logLikelihood(100,Xtest)

#pca_sklearn=PCA(n_components='mle')
#pca_sklearn.fit(X)
