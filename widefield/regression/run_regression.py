from widefield.regression.linregress import LinearRegression, RecurrentRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import tables as tb
from sklearn.decomposition import PCA, FactorAnalysis, FastICA


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


# -------------- REGIONAL REGRESSION --------------
run_regional_regression = False
save_region_files = True
load_region_files = True

if run_regional_regression:
    # Get fluorescence data into df/f format and matrix for regression
    n_time_steps = summary_data['ROIs_F'][summary_data['ROIs_F'].keys()[0]].size
    n_regions = len(summary_data['ROIs_F'].keys())
    region_data = np.zeros((n_time_steps, n_regions))
    region_labels = []
    for i,key in enumerate(sorted(summary_data['ROIs_F'].keys())):
        region_data[:, i] = (summary_data['ROIs_F'][key] - summary_data['ROIs_F0'][key]) / summary_data['ROIs_F0'][key]
        region_labels.append(key)

    # load pairs to exclude - want to run recurrent regression without using paired left/right region
    excludePairs = np.load(basepath + 'regression/regions/excludePairs.npy')

    # Construct training and test sets
    region_data_train = {'Y': region_data[20000:183000, :], 'X': regressor_data[20000:183000, :], 'X_labels': X_labels, 'Y_labels': region_labels}
    region_data_test = {'Y': region_data[183000:-20000, :], 'X': regressor_data[183000:-20000, :], 'X_labels': X_labels, 'Y_labels': region_labels}

    # Run regular and recurrent regressions.
    # based on cross-correlations, convolve over 4 seconds
    lr1_regions = LinearRegression(use_design_matrix=True, convolution_length=400)
    lr1_regions.fit(region_data_train['Y'], region_data_train['X'])
    lr2_regions = RecurrentRegression(use_design_matrix=True, convolution_length=400, recurrent_convolution_length=400)
    lr2_regions.fit(region_data_train['Y'], region_data_train['X'])
    lr3_regions = RecurrentRegression(use_design_matrix=True, convolution_length=400, recurrent_convolution_length=1)
    lr3_regions.fit(region_data_train['Y'], region_data_train['X'])

    if save_region_files:
        pickle.dump(region_data_train, open(basepath + "regression/regions/train.pkl",'w'))
        pickle.dump(region_data_test, open(basepath + "regression/regions/test.pkl",'w'))
        pickle.dump(lr1_regions, open(basepath + "regression/regions/results_nonrecurrent.pkl",'w'))
        pickle.dump(lr2_regions, open(basepath + "regression/regions/results_recurrent.pkl", 'w'))
        pickle.dump(lr3_regions, open(basepath + "regression/regions/results_halfrecurrent.pkl", 'w'))

if load_region_files:
    region_data_train = pickle.load(open(basepath + "regression/regions/train.pkl",'r'))
    region_data_test = pickle.load(open(basepath + "regression/regions/test.pkl",'r'))
    lr1_regions = pickle.load(open(basepath + "regression/regions/results_nonrecurrent.pkl", 'r'))
    lr2_regions = pickle.load(open(basepath + "regression/regions/results_recurrent.pkl", 'r'))
    lr3_regions = pickle.load(open(basepath + "regression/regions/results_halfrecurrent.pkl", 'r'))


def create_region_plot():
    f = {}
    f['percent_error'] = np.zeros((3,len(region_data_test['Y_labels'])))
    f['percent_error'][0] = lr1_regions.compute_loss_percentage(region_data_test['Y'], region_data_test['X'])
    f['percent_error'][1] = lr2_regions.compute_loss_percentage(region_data_test['Y'], region_data_test['X'])
    f['percent_error'][2] = lr3_regions.compute_loss_percentage(region_data_test['Y'], region_data_test['X'])
    f['bar_width'] = 0.3
    f['idxs'] = np.arange(len(region_data_test['Y_labels']))
    f['category_labels'] = ['regression', 'dynamics', 'dynamics-filter']
    f['region_labels'] = region_data_test['Y_labels']
    f['code'] = """
plt.figure()
plt.bar(f['idxs'], f['percent_error'][0], f['bar_width'], color='r',label=f['category_labels'][0])
plt.bar(f['idxs']+f['bar_width'], f['percent_error'][1], f['bar_width'], color='b', label=f['category_labels'][1])
plt.bar(f['idxs']+2*f['bar_width'], f['percent_error'][2], f['bar_width'], color='g', label=f['category_labels'][2])
plt.title('error as percentage of variance - regression on regions')
plt.xticks(f['idxs']+f['bar_width'], f['region_labels'], rotation='vertical')
plt.legend()
plt.tight_layout()
"""
    return f


# -------------- PCA Regression --------------
run_pca_regression = False
save_pca_files = True
load_pca_files = True

if run_pca_regression:
    # Load data
    tb_open = tb.open_file(image_data_path, 'r')
    image_data_train = tb_open.root.data[:,20000:183000].T
    image_data_test = tb_open.root.data[:,183000:-20000].T
    tb_open.close()

    pca_model = PCA(n_components=20)
    pca_model.fit(image_data_train)

    pca_data_train = {'Y': pca_model.transform(image_data_train), 'X': regressor_data[20000:183000, :], 'X_labels': X_labels}
    pca_data_test = {'Y': pca_model.transform(image_data_test), 'X': regressor_data[183000:-20000, :], 'X_labels': X_labels}

    lr1_pca = LinearRegression(use_design_matrix=True, convolution_length=400)
    lr1_pca.fit(pca_data_train['Y'], pca_data_train['X'])
    lr2_pca = RecurrentRegression(use_design_matrix=True, convolution_length=400, recurrent_convolution_length=400)
    lr2_pca.fit(pca_data_train['Y'], pca_data_train['X'])
    lr3_pca = RecurrentRegression(use_design_matrix=True, convolution_length=400, recurrent_convolution_length=1)
    lr3_pca.fit(pca_data_train['Y'], pca_data_train['X'])

    if save_pca_files:
        pickle.dump(pca_model, open(basepath + "regression/pca/pca_model.pkl",'w'))
        pickle.dump(pca_data_train, open(basepath + "regression/pca/train.pkl",'w'))
        pickle.dump(pca_data_test, open(basepath + "regression/pca/test.pkl",'w'))
        pickle.dump(lr1_pca, open(basepath + "regression/pca/results_nonrecurrent.pkl", 'w'))
        pickle.dump(lr2_pca, open(basepath + "regression/pca/results_recurrent.pkl", 'w'))
        pickle.dump(lr3_pca, open(basepath + "regression/pca/results_halfrecurrent.pkl", 'w'))

if load_pca_files:
    pca_model = pickle.load(open(basepath + "regression/pca/pca_model.pkl",'r'))
    pca_data_train = pickle.load(open(basepath + "regression/pca/train.pkl",'r'))
    pca_data_test = pickle.load(open(basepath + "regression/pca/test.pkl",'r'))
    lr1_pca = pickle.load(open(basepath + "regression/pca/results_nonrecurrent.pkl", 'r'))
    lr2_pca = pickle.load(open(basepath + "regression/pca/results_recurrent.pkl", 'r'))
    lr3_pca = pickle.load(open(basepath + "regression/pca/results_halfrecurrent.pkl", 'r'))

def create_pca_plot():
    f = {}
    f['percent_error'] = np.zeros((3,pca_data_test['Y'].shape[1]))
    f['percent_error'][0] = lr1_pca.compute_loss_percentage(pca_data_test['Y'], pca_data_test['X'])
    f['percent_error'][1] = lr3_pca.compute_loss_percentage(pca_data_test['Y'], pca_data_test['X'])
    f['percent_error'][2] = lr2_pca.compute_loss_percentage(pca_data_test['Y'], pca_data_test['X'])
    f['bar_width'] = 0.3
    f['idxs'] = np.arange(pca_data_test['Y'].shape[1])
    f['category_labels'] = ['regression', 'dynamics', 'dynamics-filter']
    f['code'] = """
plt.figure()
plt.bar(f['idxs'], f['percent_error'][0], f['bar_width'], color='r',label=f['category_labels'][0])
plt.bar(f['idxs']+f['bar_width'], f['percent_error'][1], f['bar_width'], color='b', label=f['category_labels'][1])
plt.bar(f['idxs']+2*f['bar_width'], f['percent_error'][2], f['bar_width'], color='g', label=f['category_labels'][2])
plt.title('error as percentage of variance - regression on 20 PCA components')
plt.legend()
plt.tight_layout()
"""
    return f

# -------------- Factor Analysis Regression --------------
run_fa_regression = False
save_fa_files = False
load_fa_files = False

if run_fa_regression:
    if not run_pca_regression:
        # Load data
        tb_open = tb.open_file(image_data_path, 'r')
        image_data_train = tb_open.root.data[:,20000:183000].T
        image_data_test = tb_open.root.data[:,183000:-20000].T
        tb_open.close()

    fa_model = FactorAnalysis(n_components=20)
    fa_model.fit(image_data_train)

    fa_data_train = {'Y': fa_model.transform(image_data_train), 'X': regressor_data[20000:183000, :], 'X_labels': X_labels}
    fa_data_test = {'Y': fa_model.transform(image_data_test), 'X': regressor_data[183000:-20000, :], 'X_labels': X_labels}

    lr1_fa = LinearRegression(use_design_matrix=True, convolution_length=400)
    lr1_fa.fit(fa_data_train['Y'], fa_data_train['X'])
    lr2_fa = RecurrentRegression(use_design_matrix=True, convolution_length=400, recurrent_convolution_length=400)
    lr2_fa.fit(fa_data_train['Y'], fa_data_train['X'])

    if save_fa_files:
        pickle.dump(fa_model, open(basepath + "regression/fa/fa_model.pkl",'w'))
        pickle.dump(fa_data_train, open(basepath + "regression/fa/train.pkl",'w'))
        pickle.dump(fa_data_test, open(basepath + "regression/fa/test.pkl",'w'))
        pickle.dump(lr1_fa, open(basepath + "regression/fa/results_nonrecurrent.pkl", 'w'))
        pickle.dump(lr2_fa, open(basepath + "regression/fa/results_recurrent.pkl", 'w'))

if load_fa_files:
    fa_model = pickle.load(open(basepath + "regression/fa/fa_model.pkl",'r'))
    fa_data_train = pickle.load(open(basepath + "regression/fa/train.pkl",'r'))
    fa_data_test = pickle.load(open(basepath + "regression/fa/test.pkl",'r'))
    lr1_fa = pickle.load(open(basepath + "regression/fa/results_nonrecurrent.pkl", 'r'))
    lr2_fa = pickle.load(open(basepath + "regression/fa/results_recurrent.pkl", 'r'))

def create_fa_plot():
    f = {}
    f['percent_error'] = np.zeros((2,fa_data_test['Y'].shape[1]))
    f['percent_error'][0] = lr1_fa.compute_loss_percentage(fa_data_test['Y'], fa_data_test['X'])
    f['percent_error'][1] = lr2_fa.compute_loss_percentage(fa_data_test['Y'], fa_data_test['X'])
    f['bar_width'] = 0.4
    f['idxs'] = np.arange(fa_data_test['Y'].shape[1])
    f['category_labels'] = ['nonrecurrent', 'recurrent']
    f['code'] = """
plt.figure()
plt.bar(f['idxs'], f['percent_error'][0], f['bar_width'], color='r',label=f['category_labels'][0])
plt.bar(f['idxs']+f['bar_width'], f['percent_error'][1], f['bar_width'], color='b', label=f['category_labels'][1])
plt.title('error as percentage of variance - regression on 20 FA components')
plt.legend()
plt.tight_layout()
"""
    return f


# -------------- ICA --------------
run_ica = False
save_ica_files = True
load_ica_files = True
plot_ica_components = False

if run_ica:
    pca_data_train_whiten = pca_data_train['Y'][:,0:10]/np.sqrt(pca_model.explained_variance_[0:10])
    pca_data_test_whiten = pca_data_test['Y'][:,0:10]/np.sqrt(pca_model.explained_variance_[0:10])

    ica_model = FastICA(whiten=False)
    ica_data_train = {'Y': ica_model.fit_transform(pca_data_train_whiten), 'X': regressor_data[20000:183000, :], 'X_labels': X_labels}
    ica_data_test = {'Y': ica_model.transform(pca_data_test_whiten), 'X': regressor_data[183000:-20000, :], 'X_labels': X_labels}

    lr1_ica = LinearRegression(use_design_matrix=True, convolution_length=400)
    lr1_ica.fit(ica_data_train['Y'], ica_data_train['X'])
    lr2_ica = RecurrentRegression(use_design_matrix=True, convolution_length=400, recurrent_convolution_length=400)
    lr2_ica.fit(ica_data_train['Y'], ica_data_train['X'])
    lr3_ica = RecurrentRegression(use_design_matrix=True, convolution_length=400, recurrent_convolution_length=1)
    lr3_ica.fit(ica_data_train['Y'], ica_data_train['X'])

    if save_ica_files:
        pickle.dump(ica_model, open(basepath + "regression/ica/ica_model.pkl",'w'))
        pickle.dump(ica_data_train, open(basepath + "regression/ica/train.pkl",'w'))
        pickle.dump(ica_data_test, open(basepath + "regression/ica/test.pkl",'w'))
        pickle.dump(lr1_ica, open(basepath + "regression/ica/results_nonrecurrent.pkl", 'w'))
        pickle.dump(lr2_ica, open(basepath + "regression/ica/results_recurrent.pkl", 'w'))
        pickle.dump(lr3_ica, open(basepath + "regression/ica/results_halfrecurrent.pkl", 'w'))

if load_ica_files:
    ica_model = pickle.load(open(basepath + "regression/ica/ica_model.pkl",'r'))
    ica_data_train = pickle.load(open(basepath + "regression/ica/train.pkl",'r'))
    ica_data_test = pickle.load(open(basepath + "regression/ica/test.pkl",'r'))
    lr1_ica = pickle.load(open(basepath + "regression/ica/results_nonrecurrent.pkl", 'r'))
    lr2_ica = pickle.load(open(basepath + "regression/ica/results_recurrent.pkl", 'r'))
    lr3_ica = pickle.load(open(basepath + "regression/ica/results_halfrecurrent.pkl", 'r'))

def create_ica_plot():
    f = {}
    f['percent_error'] = np.zeros((3,ica_data_test['Y'].shape[1]))
    f['percent_error'][0] = lr1_ica.compute_loss_percentage(ica_data_test['Y'], ica_data_test['X'])
    f['percent_error'][1] = lr3_ica.compute_loss_percentage(ica_data_test['Y'], ica_data_test['X'])
    f['percent_error'][2] = lr2_ica.compute_loss_percentage(ica_data_test['Y'], ica_data_test['X'])
    f['bar_width'] = 0.3
    f['idxs'] = np.arange(ica_data_test['Y'].shape[1])
    f['category_labels'] = ['regression', 'dynamics', 'dynamics-filter']
    f['code'] = """
plt.figure()
plt.bar(f['idxs'], f['percent_error'][0], f['bar_width'], color='r',label=f['category_labels'][0])
plt.bar(f['idxs']+f['bar_width'], f['percent_error'][1], f['bar_width'], color='b', label=f['category_labels'][1])
plt.bar(f['idxs']+2*f['bar_width'], f['percent_error'][2], f['bar_width'], color='g', label=f['category_labels'][2])
plt.title('error as percentage of variance - regression on 10 ICA components')
plt.legend()
plt.tight_layout()
"""
    return f

# if plot_ica_components:
#     for i in range(10):
#         plt.subplot(2,5,i+1)
#         plt.imshow(pca_components[:,i].reshape(128,128),interpolation='none')
#         plt.clim([-0.02,0.02])
#         plt.axis('off')
