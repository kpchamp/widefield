from widefield.regression.linregress import linear_regression, recurrent_regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

basepath = "/suppscr/riekesheabrown/kpchamp/data/m187474/150804/"
path = basepath + "SummaryTimeSeries.pkl"
data = pd.read_pickle(path)

# replace NaNs with 0s
data['behavioral_measurables']['running_speed'][np.where(np.isnan(data['behavioral_measurables']['running_speed']))[0]] = 0

# get fluorescence data into df/f format and matrix for regression
n_timesteps = data['ROIs_F'][data['ROIs_F'].keys()[0]].size
n_regions = len(data['ROIs_F'].keys())
Y = np.zeros((n_timesteps, n_regions))
Y_labels = []
for i,key in enumerate(sorted(data['ROIs_F'].keys())):
    Y[:,i] = (data['ROIs_F'][key] - data['ROIs_F0'][key])/data['ROIs_F0'][key]
    Y_labels.append(key)

# load pairs to exclude - want to run recurrent regression without using paired left/right region
excludePairs = np.load(basepath + 'regression/excludePairs.npy')

# get stimulus orientations into array
# stimulus_contrasts = np.squeeze(data['session_dataframe'].as_matrix(columns=['Contrast']))
# stimulus_orientations = np.zeros(data['stimulus'].shape)
# stimulus_orientations[np.where(data['stimulus']!=0)[0]] = np.squeeze(data['session_dataframe'].as_matrix(columns=['Ori']))[np.where(stimulus_contrasts!=0)[0]]

# get regressors into format for regression; leave out pupil information for now
X = np.vstack((data['stimulus'], data['behavioral_measurables']['licking'],
               data['behavioral_measurables']['rewards'], data['behavioral_measurables']['running_speed'])).T
X_labels = ['stimulus contrast', 'licking', 'rewards', 'running speed']

training_data = {'Y': Y[20000:183000,:], 'X': X[20000:183000,:], 'X_labels': X_labels, 'Y_labels': Y_labels}
test_data = {'Y': Y[183000:-20000,:], 'X': X[183000:-20000,:], 'X_labels': X_labels, 'Y_labels': Y_labels}

# based on cross-correlations, convolve over 5 seconds
lr1 = linear_regression(use_design_matrix=True, convolution_length=400)
lr1.fit(training_data['Y'], training_data['X'])
lr2 = recurrent_regression(use_design_matrix=True, convolution_length=400)
lr2.fit(training_data['Y'], training_data['X'])
#pickle.dump(training_data,open(basepath + 'regression/train.pkl', 'w'))
#pickle.dump(test_data,open(basepath + 'regression/test.pkl', 'w'))
pickle.dump(lr1, open(basepath + 'regression/regression_results_400.pkl','w'))
pickle.dump(lr2, open(basepath + 'regression/regression_results_recurrent_400.pkl','w'))
#pickle.dump(lr3, open(basepath + 'regression/regression_results_recurrent_noPairs_400.pkl','w'))
