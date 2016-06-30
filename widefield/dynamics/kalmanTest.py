import scipy.io as io
from widefield.dynamics.lds import *

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

lds = lds_model(y, 4, max_iters=25)
# lds = lds_model(F, H, Q, R, initmu, initV)
xfilt, Vfilt, LL = lds.kalman_filter(y)
# xsmooth, Vsmooth, J = lds.kalman_smoothing(y)

# plt.figure()
# plt.plot(y[0,:], y[1,:], 'go', linewidth=3, markersize=12)
# plt.plot(xfilt[0,:], xfilt[1,:], 'rx-', linewidth=3, markersize=12)
# plt.axis('equal')
# plt.title('filtered results')
# plt.show()
#
# plt.figure()
# plt.plot(y[0,:], y[1,:], 'go', linewidth=3, markersize=12)
# plt.plot(xsmooth[0,:], xsmooth[1,:], 'rx-', linewidth=3, markersize=12)
# plt.axis('equal')
# plt.title('smoothed results')
# plt.show()
