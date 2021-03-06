from widefield.regression.linregress import LinearRegression, DynamicRegression
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
    lr_If_regions = LinearRegression(convolution_length=400)
    lr_If_regions.fit(region_data_train['Y'], region_data_train['X'])
    lr_D_regions = DynamicRegression(dynamic_convolution_length=1)
    lr_D_regions.fit(region_data_train['Y'])
    lr_Df_regions = DynamicRegression(dynamic_convolution_length=400)
    lr_Df_regions.fit(region_data_train['Y'])
    lr_DIf_regions = DynamicRegression(convolution_length=400, dynamic_convolution_length=1)
    lr_DIf_regions.fit(region_data_train['Y'], region_data_train['X'])
    lr_DfIf_regions = DynamicRegression(convolution_length=400, dynamic_convolution_length=400)
    lr_DfIf_regions.fit(region_data_train['Y'], region_data_train['X'])


    if save_region_files:
        pickle.dump(region_data_train, open(basepath + "regression/regions/train.pkl",'w'))
        pickle.dump(region_data_test, open(basepath + "regression/regions/test.pkl",'w'))
        pickle.dump(lr_If_regions, open(basepath + "regression/regions/lr_If.pkl",'w'))
        pickle.dump(lr_D_regions, open(basepath + "regression/regions/lr_D.pkl", 'w'))
        pickle.dump(lr_Df_regions, open(basepath + "regression/regions/lr_Df.pkl", 'w'))
        pickle.dump(lr_DIf_regions, open(basepath + "regression/regions/lr_DIf.pkl", 'w'))
        pickle.dump(lr_DfIf_regions, open(basepath + "regression/regions/lr_DfIf.pkl", 'w'))

if load_region_files:
    region_data_train = pickle.load(open(basepath + "regression/regions/train.pkl",'r'))
    region_data_test = pickle.load(open(basepath + "regression/regions/test.pkl",'r'))
    lr_If_regions = pickle.load(open(basepath + "regression/regions/lr_If.pkl", 'r'))
    lr_D_regions = pickle.load(open(basepath + "regression/regions/lr_D.pkl", 'r'))
    lr_Df_regions = pickle.load(open(basepath + "regression/regions/lr_Df.pkl", 'r'))
    lr_DIf_regions = pickle.load(open(basepath + "regression/regions/lr_DIf.pkl", 'r'))
    lr_DfIf_regions = pickle.load(open(basepath + "regression/regions/lr_DfIf.pkl", 'r'))


def create_region_plot():
    f = {}
    f['percent_error'] = np.zeros((5,len(region_data_test['Y_labels'])))
    f['percent_error'][0] = lr_If_regions.compute_loss_percentage(region_data_test['Y'], region_data_test['X'])
    f['percent_error'][1] = lr_D_regions.compute_loss_percentage(region_data_test['Y'])
    f['percent_error'][2] = lr_Df_regions.compute_loss_percentage(region_data_test['Y'])
    f['percent_error'][3] = lr_DIf_regions.compute_loss_percentage(region_data_test['Y'], region_data_test['X'])
    f['percent_error'][4] = lr_DfIf_regions.compute_loss_percentage(region_data_test['Y'], region_data_test['X'])
    f['bar_width'] = 0.15
    f['idxs'] = np.arange(len(region_data_test['Y_labels']))
    f['category_labels'] = ['If', 'D', 'Df', 'DIf', 'DfIf']
    f['region_labels'] = region_data_test['Y_labels']
    f['code'] = """
plt.figure()
plt.bar(f['idxs'], f['percent_error'][0], f['bar_width'], color='r',label=f['category_labels'][0])
plt.bar(f['idxs']+f['bar_width'], f['percent_error'][1], f['bar_width'], color='b', label=f['category_labels'][1])
plt.bar(f['idxs']+2*f['bar_width'], f['percent_error'][2], f['bar_width'], color='g', label=f['category_labels'][2])
plt.bar(f['idxs']+3*f['bar_width'], f['percent_error'][3], f['bar_width'], color='y', label=f['category_labels'][3])
plt.bar(f['idxs']+4*f['bar_width'], f['percent_error'][4], f['bar_width'], color='k', label=f['category_labels'][4])
plt.title('error as percentage of variance - regression on regions')
plt.xticks(f['idxs']+2*f['bar_width'], f['region_labels'], rotation='vertical')
plt.legend()
plt.tight_layout()
"""
    return f


def get_region_reconstructions(idxs):
    reconstructions = np.empty((6,idxs.size,len(region_data_test['Y_labels'])))
    reconstructions[0] = region_data_test['Y'][idxs]
    reconstructions[1] = lr_If_regions.reconstruct(region_data_test['X'])[idxs]
    reconstructions[2] = lr_D_regions.reconstruct(region_data_test['Y'])[idxs-1]
    reconstructions[3] = lr_Df_regions.reconstruct(region_data_test['Y'])[idxs-1]
    reconstructions[4] = lr_DIf_regions.reconstruct(region_data_test['Y'],region_data_test['X'])[idxs-1]
    reconstructions[5] = lr_DfIf_regions.reconstruct(region_data_test['Y'],region_data_test['X'])[idxs-1]
    recon_labels = ['data', 'If', 'D', 'Df', 'DIf', 'DfIf']
    return reconstructions, recon_labels

def make_regressor_plot_values(data,time):
    nonzeros = np.where(data != 0)[0]
    n = nonzeros.size
    values = np.empty((n,4))
    for i in range(n):
        values[i,0:2] = time[nonzeros[i]]
        values[i,2] = 0
        values[i,3] = data[nonzeros[i]]
    return values


def create_region_reconstruction_plot(idxs):
    f = {}
    f['time'] = np.arange(idxs.size)*0.01
    f['reconstructions'], f['recon_labels'] = get_region_reconstructions(idxs)
    f['stim'] = make_regressor_plot_values(region_data_test['X'][idxs,0],f['time'])
    f['lick'] = make_regressor_plot_values(region_data_test['X'][idxs,1],f['time'])
    f['reward'] = make_regressor_plot_values(region_data_test['X'][idxs,2],f['time'])
    f['region_labels'] = region_data_test['Y_labels']
    f['xlim'] = [f['time'][0],f['time'][-1]]
    f['ylim'] = [1.2*np.min(f['reconstructions']), 1.2*np.max(f['reconstructions'])]
    f['color'] = np.random.permutation(plt.cm.rainbow(np.linspace(0,1,9)))
    f['code'] = """
fig = plt.figure()
for i in range(len(f['region_labels'])):
    ax = fig.add_subplot(5,5,i+1)
    for j in range(f['reconstructions'].shape[0]):
        ax.plot(f['time'],f['reconstructions'][j,:,i],c=f['color'][j],label=f['recon_labels'][j])
    ax.set_ylim(f['ylim'])
    ax.set_xlim(f['xlim'])
    ax.set_title(f['region_labels'][i],fontsize=8)
    for j in range(f['stim'].shape[0]):
        ax.plot(f['stim'][j,0:2],f['stim'][j,2:],c=f['color'][f['reconstructions'].shape[0]])
    for j in range(f['lick'].shape[0]):
        ax.plot(f['lick'][j,0:2],f['lick'][j,2:],c=f['color'][1+f['reconstructions'].shape[0]])
    for j in range(f['reward'].shape[0]):
        ax.plot(f['reward'][j,0:2],f['reward'][j,2:],c=f['color'][2+f['reconstructions'].shape[0]])
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    if i==18:
        ax.legend(fontsize=8, loc='lower center', bbox_to_anchor=(-0.5,-1.5,1,1))
plt.tight_layout(pad=0.1)
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

    pca_model = PCA(n_components=10)
    pca_model.fit(image_data_train)

    pca_data_train = {'Y': pca_model.transform(image_data_train), 'X': regressor_data[20000:183000, :], 'X_labels': X_labels}
    pca_data_test = {'Y': pca_model.transform(image_data_test), 'X': regressor_data[183000:-20000, :], 'X_labels': X_labels}

    lr_If_pca = LinearRegression(convolution_length=400)
    lr_If_pca.fit(pca_data_train['Y'], pca_data_train['X'])
    lr_D_pca = DynamicRegression(dynamic_convolution_length=1)
    lr_D_pca.fit(pca_data_train['Y'])
    lr_Df_pca = DynamicRegression(dynamic_convolution_length=400)
    lr_Df_pca.fit(pca_data_train['Y'])
    lr_DIf_pca = DynamicRegression(convolution_length=400, dynamic_convolution_length=1)
    lr_DIf_pca.fit(pca_data_train['Y'], pca_data_train['X'])
    lr_DfIf_pca = DynamicRegression(convolution_length=400, dynamic_convolution_length=400)
    lr_DfIf_pca.fit(pca_data_train['Y'], pca_data_train['X'])

    if save_pca_files:
        pickle.dump(pca_model, open(basepath + "regression/pca/pca_model.pkl",'w'))
        pickle.dump(pca_data_train, open(basepath + "regression/pca/train.pkl",'w'))
        pickle.dump(pca_data_test, open(basepath + "regression/pca/test.pkl",'w'))
        pickle.dump(lr_If_pca, open(basepath + "regression/pca/lr_If_pca.pkl", 'w'))
        pickle.dump(lr_D_pca, open(basepath + "regression/pca/lr_D_pca.pkl", 'w'))
        pickle.dump(lr_Df_pca, open(basepath + "regression/pca/lr_Df_pca.pkl", 'w'))
        pickle.dump(lr_DIf_pca, open(basepath + "regression/pca/lr_DIf_pca.pkl", 'w'))
        pickle.dump(lr_DfIf_pca, open(basepath + "regression/pca/lr_DfIf_pca.pkl", 'w'))

if load_pca_files:
    pca_model = pickle.load(open(basepath + "regression/pca/pca_model.pkl",'r'))
    pca_data_train = pickle.load(open(basepath + "regression/pca/train.pkl",'r'))
    pca_data_test = pickle.load(open(basepath + "regression/pca/test.pkl",'r'))
    lr_If_pca = pickle.load(open(basepath + "regression/pca/lr_If_pca.pkl", 'r'))
    lr_D_pca = pickle.load(open(basepath + "regression/pca/lr_D_pca.pkl", 'r'))
    lr_Df_pca = pickle.load(open(basepath + "regression/pca/lr_Df_pca.pkl", 'r'))
    lr_DIf_pca = pickle.load(open(basepath + "regression/pca/lr_DIf_pca.pkl", 'r'))
    lr_DfIf_pca = pickle.load(open(basepath + "regression/pca/lr_DfIf_pca.pkl", 'r'))

def create_pca_plot():
    f = {}
    f['percent_error'] = np.zeros((5,pca_data_test['Y'].shape[1]))
    f['percent_error'][0] = lr_If_pca.compute_loss_percentage(pca_data_test['Y'], pca_data_test['X'])
    f['percent_error'][1] = lr_D_pca.compute_loss_percentage(pca_data_test['Y'])
    f['percent_error'][2] = lr_Df_pca.compute_loss_percentage(pca_data_test['Y'])
    f['percent_error'][3] = lr_DIf_pca.compute_loss_percentage(pca_data_test['Y'], pca_data_test['X'])
    f['percent_error'][4] = lr_DfIf_pca.compute_loss_percentage(pca_data_test['Y'], pca_data_test['X'])
    f['bar_width'] = 0.15
    f['idxs'] = np.arange(pca_data_test['Y'].shape[1])
    f['category_labels'] = ['If', 'D', 'Df', 'DIf', 'DfIf']
    f['code'] = """
plt.figure()
plt.bar(f['idxs'], f['percent_error'][0], f['bar_width'], color='r',label=f['category_labels'][0])
plt.bar(f['idxs']+f['bar_width'], f['percent_error'][1], f['bar_width'], color='b', label=f['category_labels'][1])
plt.bar(f['idxs']+2*f['bar_width'], f['percent_error'][2], f['bar_width'], color='g', label=f['category_labels'][2])
plt.bar(f['idxs']+3*f['bar_width'], f['percent_error'][3], f['bar_width'], color='y', label=f['category_labels'][3])
plt.bar(f['idxs']+4*f['bar_width'], f['percent_error'][4], f['bar_width'], color='k', label=f['category_labels'][4])
plt.title('error as percentage of variance - regression on 10 PCA components')
plt.legend()
plt.tight_layout()
"""
    return f


def get_pca_reconstructions(idxs):
    reconstructions = np.empty((6,idxs.size,pca_data_test['Y'].shape[1]))
    reconstructions[0] = pca_data_test['Y'][idxs]
    reconstructions[1] = lr_If_pca.reconstruct(pca_data_test['X'])[idxs]
    reconstructions[2] = lr_D_pca.reconstruct(pca_data_test['Y'])[idxs-1]
    reconstructions[3] = lr_Df_pca.reconstruct(pca_data_test['Y'])[idxs-1]
    reconstructions[4] = lr_DIf_pca.reconstruct(pca_data_test['Y'],pca_data_test['X'])[idxs-1]
    reconstructions[5] = lr_DfIf_pca.reconstruct(pca_data_test['Y'],pca_data_test['X'])[idxs-1]
    recon_labels = ['data', 'If', 'D', 'Df', 'DIf', 'DfIf']
    return reconstructions, recon_labels


def create_pca_reconstruction_plot(idxs):
    f = {}
    f['time'] = np.arange(idxs.size)*0.01
    f['reconstructions'], f['recon_labels'] = get_pca_reconstructions(idxs)
    f['stim'] = make_regressor_plot_values(pca_data_test['X'][idxs,0],f['time'])
    f['lick'] = make_regressor_plot_values(pca_data_test['X'][idxs,1],f['time'])
    f['reward'] = make_regressor_plot_values(pca_data_test['X'][idxs,2],f['time'])
    f['xlim'] = [f['time'][0],f['time'][-1]]
    f['ylim'] = [1.2*np.min(f['reconstructions']), 1.2*np.max(f['reconstructions'])]
    f['color'] = np.random.permutation(plt.cm.rainbow(np.linspace(0,1,9)))
    f['code'] = """
fig = plt.figure()
for i in range(f['reconstructions'].shape[2]):
    ax = fig.add_subplot(2,5,i+1)
    for j in range(f['reconstructions'].shape[0]):
        ax.plot(f['time'],f['reconstructions'][j,:,i],c=f['color'][j],label=f['recon_labels'][j])
    ax.set_ylim(f['ylim'])
    ax.set_xlim(f['xlim'])
    for j in range(f['stim'].shape[0]):
        ax.plot(f['stim'][j,0:2],f['stim'][j,2:],c=f['color'][f['reconstructions'].shape[0]])
    for j in range(f['lick'].shape[0]):
        ax.plot(f['lick'][j,0:2],f['lick'][j,2:],c=f['color'][1+f['reconstructions'].shape[0]])
    for j in range(f['reward'].shape[0]):
        ax.plot(f['reward'][j,0:2],f['reward'][j,2:],c=f['color'][2+f['reconstructions'].shape[0]])
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    if i==9:
        ax.legend(fontsize=8, loc='lower right')
plt.tight_layout(pad=0.1)
"""
    return f


# -------------- ICA --------------
run_ica = False
save_ica_files = True
load_ica_files = True
plot_ica_components = False

if run_ica:
    pca_data_train_whiten = pca_data_train['Y']/np.sqrt(pca_model.explained_variance_)
    pca_data_test_whiten = pca_data_test['Y']/np.sqrt(pca_model.explained_variance_)

    ica_model = FastICA(whiten=False)
    ica_data_train = {'Y': ica_model.fit_transform(pca_data_train_whiten), 'X': regressor_data[20000:183000, :], 'X_labels': X_labels}
    ica_data_test = {'Y': ica_model.transform(pca_data_test_whiten), 'X': regressor_data[183000:-20000, :], 'X_labels': X_labels}

    lr_If_ica = LinearRegression(convolution_length=400)
    lr_If_ica.fit(ica_data_train['Y'], ica_data_train['X'])
    lr_D_ica = DynamicRegression(dynamic_convolution_length=1)
    lr_D_ica.fit(ica_data_train['Y'])
    lr_Df_ica = DynamicRegression(dynamic_convolution_length=400)
    lr_Df_ica.fit(ica_data_train['Y'])
    lr_DIf_ica = DynamicRegression(convolution_length=400, dynamic_convolution_length=1)
    lr_DIf_ica.fit(ica_data_train['Y'], ica_data_train['X'])
    lr_DfIf_ica = DynamicRegression(convolution_length=400, dynamic_convolution_length=400)
    lr_DfIf_ica.fit(ica_data_train['Y'], ica_data_train['X'])

    if save_ica_files:
        pickle.dump(ica_model, open(basepath + "regression/ica/ica_model.pkl",'w'))
        pickle.dump(ica_data_train, open(basepath + "regression/ica/train.pkl",'w'))
        pickle.dump(ica_data_test, open(basepath + "regression/ica/test.pkl",'w'))
        pickle.dump(lr_If_ica, open(basepath + "regression/ica/lr_If_ica.pkl", 'w'))
        pickle.dump(lr_D_ica, open(basepath + "regression/ica/lr_D_ica.pkl", 'w'))
        pickle.dump(lr_Df_ica, open(basepath + "regression/ica/lr_Df_ica.pkl", 'w'))
        pickle.dump(lr_DIf_ica, open(basepath + "regression/ica/lr_DIf_ica.pkl", 'w'))
        pickle.dump(lr_DfIf_ica, open(basepath + "regression/ica/lr_DfIf_ica.pkl", 'w'))

if load_ica_files:
    ica_model = pickle.load(open(basepath + "regression/ica/ica_model.pkl",'r'))
    ica_data_train = pickle.load(open(basepath + "regression/ica/train.pkl",'r'))
    ica_data_test = pickle.load(open(basepath + "regression/ica/test.pkl",'r'))
    lr_If_ica = pickle.load(open(basepath + "regression/ica/lr_If_ica.pkl", 'r'))
    lr_D_ica = pickle.load(open(basepath + "regression/ica/lr_D_ica.pkl", 'r'))
    lr_Df_ica = pickle.load(open(basepath + "regression/ica/lr_Df_ica.pkl", 'r'))
    lr_DIf_ica = pickle.load(open(basepath + "regression/ica/lr_DIf_ica.pkl", 'r'))
    lr_DfIf_ica = pickle.load(open(basepath + "regression/ica/lr_DfIf_ica.pkl", 'r'))


def create_ica_plot():
    f = {}
    f['percent_error'] = np.zeros((5,ica_data_test['Y'].shape[1]))
    f['percent_error'][0] = lr_If_ica.compute_loss_percentage(ica_data_test['Y'], ica_data_test['X'])
    f['percent_error'][1] = lr_D_ica.compute_loss_percentage(ica_data_test['Y'])
    f['percent_error'][2] = lr_Df_ica.compute_loss_percentage(ica_data_test['Y'])
    f['percent_error'][3] = lr_DIf_ica.compute_loss_percentage(ica_data_test['Y'], ica_data_test['X'])
    f['percent_error'][4] = lr_DfIf_ica.compute_loss_percentage(ica_data_test['Y'], ica_data_test['X'])
    f['bar_width'] = 0.15
    f['idxs'] = np.arange(ica_data_test['Y'].shape[1])
    f['category_labels'] = ['If', 'D', 'Df', 'DIf', 'DfIf']
    f['code'] = """
plt.figure()
plt.bar(f['idxs'], f['percent_error'][0], f['bar_width'], color='r',label=f['category_labels'][0])
plt.bar(f['idxs']+f['bar_width'], f['percent_error'][1], f['bar_width'], color='b', label=f['category_labels'][1])
plt.bar(f['idxs']+2*f['bar_width'], f['percent_error'][2], f['bar_width'], color='g', label=f['category_labels'][2])
plt.bar(f['idxs']+3*f['bar_width'], f['percent_error'][3], f['bar_width'], color='y', label=f['category_labels'][3])
plt.bar(f['idxs']+4*f['bar_width'], f['percent_error'][4], f['bar_width'], color='k', label=f['category_labels'][4])
plt.title('error as percentage of variance - regression on 10 ICA components')
plt.legend()
plt.tight_layout()
"""
    return f


def get_ica_reconstructions(idxs):
    reconstructions = np.empty((6,idxs.size,ica_data_test['Y'].shape[1]))
    reconstructions[0] = ica_data_test['Y'][idxs]
    reconstructions[1] = lr_If_ica.reconstruct(ica_data_test['X'])[idxs]
    reconstructions[2] = lr_D_ica.reconstruct(ica_data_test['Y'])[idxs-1]
    reconstructions[3] = lr_Df_ica.reconstruct(ica_data_test['Y'])[idxs-1]
    reconstructions[4] = lr_DIf_ica.reconstruct(ica_data_test['Y'],ica_data_test['X'])[idxs-1]
    reconstructions[5] = lr_DfIf_ica.reconstruct(ica_data_test['Y'],ica_data_test['X'])[idxs-1]
    recon_labels = ['data', 'If', 'D', 'Df', 'DIf', 'DfIf']
    return reconstructions, recon_labels


def create_ica_reconstruction_plot(idxs):
    f = {}
    f['time'] = np.arange(idxs.size)*0.01
    f['reconstructions'], f['recon_labels'] = get_ica_reconstructions(idxs)
    f['stim'] = make_regressor_plot_values(ica_data_test['X'][idxs,0],f['time'])
    f['lick'] = make_regressor_plot_values(ica_data_test['X'][idxs,1],f['time'])
    f['reward'] = make_regressor_plot_values(ica_data_test['X'][idxs,2],f['time'])
    f['xlim'] = [f['time'][0],f['time'][-1]]
    f['ylim'] = [1.2*np.min(f['reconstructions']), 1.2*np.max(f['reconstructions'])]
    f['color'] = np.random.permutation(plt.cm.rainbow(np.linspace(0,1,9)))
    f['code'] = """
fig = plt.figure()
for i in range(f['reconstructions'].shape[2]):
    ax = fig.add_subplot(2,5,i+1)
    for j in range(f['reconstructions'].shape[0]):
        ax.plot(f['time'],f['reconstructions'][j,:,i],c=f['color'][j],label=f['recon_labels'][j])
    ax.set_ylim(f['ylim'])
    ax.set_xlim(f['xlim'])
    for j in range(f['stim'].shape[0]):
        ax.plot(f['stim'][j,0:2],f['stim'][j,2:],c=f['color'][f['reconstructions'].shape[0]])
    for j in range(f['lick'].shape[0]):
        ax.plot(f['lick'][j,0:2],f['lick'][j,2:],c=f['color'][1+f['reconstructions'].shape[0]])
    for j in range(f['reward'].shape[0]):
        ax.plot(f['reward'][j,0:2],f['reward'][j,2:],c=f['color'][2+f['reconstructions'].shape[0]])
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    if i==9:
        ax.legend(fontsize=8, loc='lower right')
plt.tight_layout(pad=0.1)
"""
    return f


# if plot_ica_components:
#     for i in range(10):
#         plt.subplot(2,5,i+1)
#         plt.imshow(pca_components[:,i].reshape(128,128),interpolation='none')
#         plt.clim([-0.02,0.02])
#         plt.axis('off')
