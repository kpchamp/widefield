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

train = pickle.load(open(basepath + "ml_project/small_test/train.pkl",'r'))

# Create SSM with the test parameters
test_params = pickle.load(open(basepath + "ml_project/small_test/params.pkl",'r'))
test_model = LinearGaussianSSM(A=test_params['W'], Q=test_params['Q'], C=np.eye(3), R=test_params['R'],
                               B=test_params['B'], mu0=test_params['mu0'], V0=test_params['V0'])
pickle.dump(test_model, open(basepath + "ml_project/small_test/model.pkl",'w'))
# test_model = pickle.load(open(basepath + "ml_project/small_test/model.pkl",'r'))

# Sample the SSM
T = 5000
Z,Y = test_model.sample(T, U=train['U'][0:T].T)

# See if we can learn the SSM, starting with the fit LR model
lr_test = DynamicRegression(fit_offset=False)
lr_test.fit(train['Y'], train['U'])
test_model_learn = LinearGaussianSSM(A=np.copy(lr_test.coefficients[1:].T), B=np.copy(lr_test.coefficients[0:1].T), C=np.eye(3))
test_model_learn.fit_em(Y, train['U'][0:T].T, max_iters=500, tol=1., exclude_list=['C'], diagonal_covariance=True)
pickle.dump(test_model_learn, open(basepath + "ml_project/small_test/model_learn_5000.pkl",'w'))
