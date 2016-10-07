from widefield.regression.linregress import *
import pandas as pd
import matplotlib.pyplot as plt
import pickle

path = "/suppscr/riekesheabrown/kpchamp/data/m187474/150804/SummaryTimeSeries.pkl"
#path = "/Users/kpchamp/Dropbox (uwamath)/backup/research/python/notebooks/SummaryTimeSeries.pkl"
data = pd.read_pickle(path)
data['behavioral_measurables']['running_speed'][np.where(np.isnan(data['behavioral_measurables']['running_speed']))[0]] = 0

n_timesteps = data['ROIs_F'][data['ROIs_F'].keys()[0]].size
n_regions = len(data['ROIs_F'].keys())
Y = np.zeros((n_timesteps, n_regions))
for i,key in enumerate(data['ROIs_F'].keys()):
    Y[:,i] = (data['ROIs_F'][key] - data['ROIs_F0'][key])/data['ROIs_F0'][key]

#Y = Y[0:365115]

#X = data['behavioral_measurables']['pupil_azimuth']
#phi = create_design_matrix(np.reshape(X, (-1,1)))
#beta = fit_lr(Y, phi)
#beta2 = fit_lr_analytic(Y, X)

#X = np.vstack((data['stimulus'][0:365115], data['behavioral_measurables']['licking'][0:365115],
#               data['behavioral_measurables']['rewards'][0:365115], data['behavioral_measurables']['pupil_azimuth'],
              #data['behavioral_measurables']['pupil_diameter_mm'], data['behavioral_measurables']['pupil_elevation'])).T
X = np.vstack((data['stimulus'], data['behavioral_measurables']['licking'], data['behavioral_measurables']['rewards'],
               data['behavioral_measurables']['running_speed'])).T
phi = create_design_matrix(X, type='convolution', convolution_length=500)
G1 = fit_lr(Y, phi, method='least squares')
#G2 = fit_lr(Y, phi, method='gradient descent')

regress = {}
regress['G'] = G1
regress['phi'] = phi
regress['Y'] = Y
pickle.dump(regress, open("/suppscr/riekesheabrown/kpchamp/data/m187474/150804/regress_t500.pkl"))

# for region in range(5):
#     plt.subplot(2,5,region+1)
#     plt.plot(G[0:100,region])
#     plt.subplot(2,5,region+6)
#     plt.plot(G[100:,region])
# plt.show()
