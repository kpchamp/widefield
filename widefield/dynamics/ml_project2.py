from widefield.dynamics.ssm import LinearGaussianSSM
from widefield.regression.linregress import LinearRegression, RecurrentRegression
import pickle
import tables as tb
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from pykalman import KalmanFilter

# Import
basepath = "/suppscr/riekesheabrown/kpchamp/data/m187474/150804/"
summary_data_path = basepath + "SummaryTimeSeries.pkl"
summary_data = pd.read_pickle(summary_data_path)

# replace NaNs with 0s
summary_data['behavioral_measurables']['running_speed'][np.where(np.isnan(summary_data['behavioral_measurables']['running_speed']))[0]] = 0

include_orientations = False
if include_orientations:
    # Get stimulus orientations into array
    stimulus_contrasts = np.squeeze(summary_data['session_dataframe'].as_matrix(columns=['Contrast']))
    stimulus_orientations = np.zeros(summary_data['stimulus'].shape)
    stimulus_orientations[np.where(summary_data['stimulus'] != 0)[0]] = np.squeeze(summary_data['session_dataframe'].as_matrix(columns=['Ori']))[np.where(stimulus_contrasts != 0)[0]]

    # Get regressors into format for regression
    regressor_data = np.vstack((stimulus_contrasts, stimulus_orientations, summary_data['behavioral_measurables']['licking'],
                                summary_data['behavioral_measurables']['rewards'], summary_data['behavioral_measurables']['running_speed'])).T
    X_labels = ['stimulus contrast', 'stimulus orientation', 'licking', 'rewards', 'running speed']
else:
    # Get regressors into format for regression
    regressor_data = np.vstack((summary_data['stimulus'], summary_data['behavioral_measurables']['licking'],
                                summary_data['behavioral_measurables']['rewards'], summary_data['behavioral_measurables']['running_speed'])).T
    X_labels = ['stimulus contrast', 'licking', 'rewards', 'running speed']


n_time_steps = summary_data['ROIs_F'][summary_data['ROIs_F'].keys()[0]].size
n_regions = len(summary_data['ROIs_F'].keys())
region_data = np.zeros((n_time_steps, n_regions))
region_labels = []
for i,key in enumerate(sorted(summary_data['ROIs_F'].keys())):
    region_data[:, i] = (summary_data['ROIs_F'][key] - summary_data['ROIs_F0'][key]) / summary_data['ROIs_F0'][key]
    region_labels.append(key)

# Construct training and test sets
region_data_train = {'Y': region_data[20000:25000, :], 'X': regressor_data[20000:25000, :], 'X_labels': X_labels, 'Y_labels': region_labels}
region_data_test = {'Y': region_data[183000:-20000, :], 'X': regressor_data[183000:-20000, :], 'X_labels': X_labels, 'Y_labels': region_labels}


# print "Doing linear regression"
# lr = LinearRegression(use_design_matrix=False)
# lr.fit(region_data_train['Y'][1:], region_data_train['Y'][:-1])
# pickle.dump(lr,open(basepath + "ml_project/lr.pkl",'w'))
lr = pickle.load(open(basepath + "ml_project/lr.pkl",'r'))

# print "Fitting LGSSM"
# # Fit EM parameters for the model, based on the sampled data
# model = LinearGaussianSSM(A=np.copy(lr.coefficients.T), C=np.eye(21))
# model.fit_em(region_data_train['Y'].T, max_iters=1000, exclude_list=['C'])
# pickle.dump(model,open(basepath + "ml_project/lgssm.pkl",'w'))
# model = pickle.load(open(basepath + "ml_project/lgssm.pkl",'r'))

print "Doing linear regression - supervised case"
# lr2 = RecurrentRegression(use_design_matrix=False)
# lr2.fit(region_data_train['Y'], region_data_train['X'])
# pickle.dump(lr2,open(basepath + "ml_project/lr_supervised.pkl",'w'))
lr2 = pickle.load(open(basepath + "ml_project/lr_supervised.pkl",'r'))

print "Fitting LGSSM - supervised case"
# Fit EM parameters for the model, based on the sampled data
model = LinearGaussianSSM(A=np.copy(lr.coefficients[region_data_train['X'].shape[1]:].T), B=np.copy(lr.coefficients[0:region_data_train['X'].shape[1]].T), C=np.eye(21))
model.fit_em(region_data_train['Y'].T, region_data_train['X'].T, max_iters=1000, exclude_list=['C'])
pickle.dump(model,open(basepath + "ml_project/lgssm_supervised.pkl",'w'))
# model = pickle.load(open(basepath + "ml_project/lgssm_supervised.pkl",'r'))
