import numpy as np
import scipy.linalg as la


def create_design_matrix(X, type=None, convolution_length=1):
    n_samples, n_regressors = X.shape
    if type is None:
        phi = np.hstack([X, np.ones((n_samples,1))])
    elif type == 'convolution':
        if convolution_length > n_samples:
            raise ValueError("convolution_length=%d cannot be greater than n_samples=%d" % (convolution_length,n_samples))
        phi = np.zeros((n_samples, n_regressors*convolution_length+1))
        for k in range(n_regressors):
            for j in range(convolution_length):
                phi[j:, k*convolution_length + j] = X[0:n_samples-j, k]
        phi[:,-1] += 1
    else:
        raise ValueError("type=%s is not a valid type" % type)
    return phi


def fit_lr(Y, phi):
    beta = np.zeros((phi.shape[1], Y.shape[1]))
    for i in range(Y.shape[1]):
        beta[:,i] = np.squeeze(leastsquares(np.reshape(Y[:,i], (-1,1)), phi))
    return beta


def leastsquares(y, phi):
    u,s,v = la.svd(phi, full_matrices=False)
    return (v*1/s).dot(u.T.dot(y))


# Note: This function fits a linear regression in the case where you only have one regressor
# and want to find the function G that is convolved with your regressor.
def fit_lr_analytic(Y, x):
    Ypad = zeropad(Y, n_zeros=100)
    Xpad = zeropad(x, n_zeros=100)
    Yft = np.fft.rfft(Ypad, axis=0)
    Xft = np.fft.rfft(Xpad)
    G = np.fft.irfft((Yft.T/Xft).T, axis=0)
    return G


def zeropad(x, n_zeros=1):
    if len(x.shape) == 1:
        xout = np.zeros(x.shape[0] + 2*n_zeros)
        xout[n_zeros:-n_zeros] = x
    else:
        xout = np.zeros((x.shape[0] + 2*n_zeros,x.shape[1]))
        xout[n_zeros:-n_zeros,:] = x
    return xout
