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
test_model = LinearGaussianSSM(A=test_params['W'], Q=test_params['Q'], C=np.eye(21), R=test_params['R'],
                               B=test_params['B'], mu0=test_params['mu0'], V0=test_params['V0'])
pickle.dump(test_model, open(basepath + "ml_project/test_model.pkl",'w'))

# Sample the SSM
T = 5000
Z,Y = test_model.sample(T, U=train['U'].T[0:T])

# See if we can learn the SSM, starting with the fit LR model
lr_test = DynamicRegression(fit_offset=False)
lr_test.fit(train['Y'], train['U'])
test_model_learn = LinearGaussianSSM(A=np.copy(lr_test.coefficients.T), B=np.copy(lr_test.coefficients[0:4].T), C=np.eye(21))
test_model_learn.fit_em(Y, train['U'].T[0:T], max_iters=500, tol=1., exclude_list=['C'], diagonal_covariance=True)
pickle.dump(test_model_learn, open(basepath + "ml_project/test_model_learn.pkl",'w'))
