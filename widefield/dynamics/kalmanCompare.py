import scipy.io as io
from widefield.dynamics.lds import *
from pykalman import KalmanFilter

ss = 4
os = 2
F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
Q = 0.001*np.eye(ss)
R = 1*np.eye(os)
initmu = np.array([8, 10, 1, 0])
initV = 1*np.eye(ss)

data = io.loadmat('/Users/kpchamp/Dropbox (uwamath)/backup/research/python/ldsData.mat')
x = data['x']
y = data['y']
T = x.shape[1]

kf = KalmanFilter(n_dim_state=4, n_dim_obs=2)
kf.em(y.T, em_vars={'transition_matrices', 'transition_covariance', 'observation_matrices', 'observation_covariance',
                    'initial_state_mean', 'initial_state_covariance'}, n_iter=25)
kf
