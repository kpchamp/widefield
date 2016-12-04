from widefield.dynamics.ssm import LinearGaussianSSM, BilinearGaussianSSM
from widefield.regression.linregress import LinearRegression, DynamicRegression, BilinearRegression
import pickle
import tables as tb
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import time

# Import
basepath = "/suppscr/riekesheabrown/kpchamp/data/m187474/150804/"

train = pickle.load(open(basepath + "ml_project/train.pkl",'r'))
test = pickle.load(open(basepath + "ml_project/test.pkl",'r'))

# Create SSM with the test parameters
test_params = pickle.load(open(basepath + "ml_project/test_params.pkl",'r'))
test_model = LinearGaussianSSM(A=test_params['W'], Q=np.diag(test_params['Q']), C=np.eye(21), R=np.diag(test_params['R']),
                               B=test_params['B'], mu0=test_params['mu0'], V0=test_params['V0'])
pickle.dump(test_model, open(basepath + "ml_project/big_test/model.pkl",'w'))
#test_model = pickle.load(open(basepath + "ml_project/big_test/model.pkl",'r'))

# Sample the SSM
T = 50000
Z,Y = test_model.sample(T, U=train['U'][0:T].T)

# See if we can learn the SSM, starting with the fit LR model
lr_test = DynamicRegression(fit_offset=False)
lr_test.fit(train['Y'], train['U'])
leftover_var = np.mean((train['Y'] - lr_test.reconstruct(train['Y'], train['U']))**2, axis=1)
test_model_learn = LinearGaussianSSM(A=np.copy(lr_test.coefficients[4:].T), B=np.copy(lr_test.coefficients[0:4].T), C=np.eye(21),
                                     Q=np.diag(leftover_var/2.), R=np.diag(leftover_var/2.), V0=np.diag(leftover_var/10.))
test_model_learn.fit_em(Y, train['U'][0:T].T, max_iters=500, tol=1., exclude_list=['C'], diagonal_covariance=True)
pickle.dump(test_model_learn, open(basepath + "ml_project/big_test/model_learn_50000.pkl",'w'))
