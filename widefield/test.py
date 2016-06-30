from __future__ import division

import pickle

from sklearn.decomposition import FactorAnalysis

from widefield.widefield.dimreduction.pca import *

#X=np.load("/Users/kpchamp/Dropbox (uwamath)/backup/research/python/data/testdata_small.npy")
#X=np.load("/Users/kpchamp/Dropbox (uwamath)/backup/research/python/data/circledata2d.npy")
X=np.load("/Users/kpchamp/Dropbox (uwamath)/backup/research/results/circleexample_2d/circledata2d.npy")
Xtrain=X[0:1000,:]
Xtest=X[1000:2000,:]

fa1 = FactorAnalysis(n_components=100)
fa2 = FactorAnalysis()
fa1.fit(Xtrain)
fa2.fit(Xtrain)

#pca=pca_model(X,n_components=25)
#pcaEM=pca_model(X,n_components=25,fitWith='EM',max_iters=1000)
#ppca=ppca_model(Xtrain)
#LL=ppca.logLikelihood(Xtest,100)
#LL=ppca.logLikelihood(Xtrain,100)

#dict = {'perm': [0,1,2], 'svs': np.array([3.,1.,0.5]), 'p_max': 5.}
#test.add_samples(dict)
#pickle.dump(test,open("test_pickle.pkl",'w'))
test2=pickle.load(open("test_pickle.pkl",'r'))
print test2
