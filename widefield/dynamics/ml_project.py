from widefield.dynamics.ssm import LinearGaussianSSM
from widefield.regression.linregress import LinearRegression
import pickle
import tables as tb
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

# Import
basepath = "/suppscr/riekesheabrown/kpchamp/data/m187474/150804/"
summary_data_path = basepath + "SummaryTimeSeries.pkl"
summary_data = pd.read_pickle(summary_data_path)
image_data_path = basepath + "data_detrend_mask.h5"

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

# Load data
tb_open = tb.open_file(image_data_path, 'r')
image_data_all = tb_open.root.data[:].T
image_data_train = tb_open.root.data[:,20000:21000].T
image_data_test = tb_open.root.data[:,31000:32000].T
tb_open.close()

pca_model = PCA(n_components=10)
pca_model.fit(image_data_all)
pickle.dump(pca_model,open(basepath + "ml_project/pca.pkl",'w'))
pca_data_train = pca_model.transform(image_data_train)
pca_data_test = pca_model.transform(image_data_test)

print "Doing linear regression"
lr = LinearRegression(use_design_matrix=False)
lr.fit(pca_data_train[1:], pca_data_train[:-1])
pickle.dump(lr,open(basepath + "ml_project/lr.pkl",'w'))

print "Fitting LGSSM"
# Fit EM parameters for the model, based on the sampled data
model = LinearGaussianSSM(A=np.copy(lr.coefficients), C=np.eye(10))
model.fit_em(pca_data_train, max_iters=1000, exclude_list=['C'])
pickle.dump(model,open(basepath + "ml_project/lgssm.pkl",'w'))
