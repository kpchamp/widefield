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
summary_data_path = basepath + "SummaryTimeSeries.pkl"

construct_dataset = False
if construct_dataset:
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
        U_labels = ['stimulus contrast', 'stimulus orientation', 'licking', 'rewards', 'running speed']
    else:
        # Get regressors into format for regression
        regressor_data = np.vstack((summary_data['stimulus'], summary_data['behavioral_measurables']['licking'],
                                    summary_data['behavioral_measurables']['rewards'], summary_data['behavioral_measurables']['running_speed'])).T
        U_labels = ['stimulus contrast', 'licking', 'rewards', 'running speed']


    n_time_steps = summary_data['ROIs_F'][summary_data['ROIs_F'].keys()[0]].size
    n_regions = len(summary_data['ROIs_F'].keys())
    region_data = np.zeros((n_time_steps, n_regions))
    region_labels = []
    for i,key in enumerate(sorted(summary_data['ROIs_F'].keys())):
        region_data[:, i] = (summary_data['ROIs_F'][key] - summary_data['ROIs_F0'][key]) / summary_data['ROIs_F0'][key]
        region_labels.append(key)

    # Construct training and test sets
    train = {'Y': region_data[20000:70000, :], 'U': regressor_data[20000:70000, :], 'U_labels': U_labels, 'Y_labels': region_labels}
    test = {'Y': region_data[70000:-20000, :], 'U': regressor_data[70000:-20000, :], 'U_labels': U_labels, 'Y_labels': region_labels}
    pickle.dump(train, open(basepath + "ml_project/train.pkl",'w'))
    pickle.dump(test, open(basepath + "ml_project/test.pkl",'w'))
else:
    train = pickle.load(open(basepath + "ml_project/train.pkl",'r'))
    test = pickle.load(open(basepath + "ml_project/test.pkl",'r'))

fit_original_model = False
fit_original_LR = False

if fit_original_LR:
    print >>open('progress.txt','a'), "Doing linear regression - basic model"
    lr1 = DynamicRegression(fit_offset=False)
    lr1.fit(train['Y'])
    pickle.dump(lr1,open(basepath + "ml_project/lr.pkl",'w'))
elif fit_original_model:
    lr1 = pickle.load(open(basepath + "ml_project/lr.pkl",'r'))

if fit_original_model:
    print >>open('progress.txt','a'), "Fitting SSM - basic model"
    # Fit EM parameters for the model, based on the sampled data
    #model1 = LinearGaussianSSM(A=np.copy(lr1.coefficients.T), C=np.eye(21))
    model1 = pickle.load(open(basepath + "ml_project/ssm_diagonal.pkl",'r'))
    model1.fit_em(train['Y'].T, max_iters=1000, tol=0.1, exclude_list=['C'], diagonal_covariance=True)
    pickle.dump(model1,open(basepath + "ml_project/ssm_diagonal.pkl",'w'))
#else:
#    model1 = pickle.load(open(basepath + "ml_project/ssm_diagonal.pkl",'r'))

fit_input_model = False
fit_input_LR = False
if fit_input_LR:
    print >>open('progress.txt','a'), "Doing linear regression - input model"
    lr2 = DynamicRegression(fit_offset=False)
    lr2.fit(train['Y'], train['U'])
    pickle.dump(lr2,open(basepath + "ml_project/lr_input.pkl",'w'))
elif fit_input_model:
    lr2 = pickle.load(open(basepath + "ml_project/lr_input.pkl",'r'))

if fit_input_model:
    print >>open('progress.txt','a'), "Fitting SSM - input model"
    # Fit EM parameters for the model, based on the sampled data
    # model2 = LinearGaussianSSM(A=np.copy(lr2.coefficients[4:].T), B=np.copy(lr2.coefficients[0:4].T), C=np.eye(21))
    model2 = pickle.load(open(basepath + "ml_project/ssm_input_diagonal.pkl",'r'))
    start_time = time.time()
    model2.fit_em(train['Y'].T, train['U'].T, max_iters=1000, tol=0.1, exclude_list=['C'], diagonal_covariance=True)
    pickle.dump(model2,open(basepath + "ml_project/ssm_input_diagonal.pkl",'w'))
    end_time = time.time()
    print >>open('progress.txt','a'), "EM took %f seconds" % (end_time-start_time)
#else:
#    model2 = pickle.load(open(basepath + "ml_project/ssm_input_diagonal.pkl",'r'))

fit_bilinear_model = True
fit_bilinear_LR = False
if fit_bilinear_LR:
    print >>open('progress.txt','a'), "Doing linear regression - bilinear model"
    lr3 = BilinearRegression(fit_offset=False)
    lr3.fit(train['Y'], train['U'])
    pickle.dump(lr3,open(basepath + "ml_project/lr_bilinear.pkl",'w'))
elif fit_bilinear_model:
    lr3 = pickle.load(open(basepath + "ml_project/lr_bilinear.pkl",'r'))

if fit_bilinear_model:
    print >>open('progress.txt','a'), "Fitting SSM - bilinear model"
    # Fit EM parameters for the model, based on the sampled data
    residual_variance = np.mean((lr3.reconstruct(train['Y'],train['U']) - train['Y'][1:])**2, axis=0)
    model3 = BilinearGaussianSSM(A=np.copy(lr3.coefficients[4:25].T), B=np.copy(lr3.coefficients[0:4].T),
                                 D=np.copy(np.stack(np.split(lr3.coefficients[25:].T,4,axis=1),axis=0)), C=np.eye(21),
                                 Q=np.diag(residual_variance), R=np.diag(residual_variance))
    #model3 = pickle.load(open(basepath + "ml_project/ssm_bilinear_diagonal.pkl",'r'))
    start_time = time.time()
    model3.fit_em(train['Y'].T, train['U'].T, max_iters=100, tol=0.1, exclude_list=['C'], diagonal_covariance=True)
    pickle.dump(model3,open(basepath + "ml_project/ssm_bilinear_diagonal_lowNoiseStart.pkl",'w'))
    end_time = time.time()
    print >>open('progress.txt','a'), "EM took %f seconds" % (end_time-start_time)
#else:
#    model3 = pickle.load(open(basepath + "ml_project/ssm_bilinear_diagonal.pkl",'r'))
