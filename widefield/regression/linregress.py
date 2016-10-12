import numpy as np
import scipy.linalg as la


class linear_regression:
    def __init__(self, fit_offset=True, use_design_matrix=False, convolution_length=1):
        self.fit_offset = fit_offset
        self.use_design_matrix = use_design_matrix
        if fit_offset:
            self.offset = None
        if use_design_matrix:
            self.convolution_length = convolution_length
        self.coefficients = None
        self.training_loss = None

    def create_design_matrix(self, X):
        n_samples, n_regressors = X.shape
        if self.convolution_length > n_samples:
            raise ValueError("convolution_length=%d cannot be greater than n_samples=%d" % (self.convolution_length,n_samples))
        design_matrix = np.zeros((n_samples, n_regressors*self.convolution_length))
        for k in range(n_regressors):
            for j in range(self.convolution_length):
                design_matrix[j:, k*self.convolution_length + j] = X[0:n_samples-j, k]
        return design_matrix

    def fit(self, Y, Xin, method='least squares'):
        n_samples, n_features = Y.shape
        if self.use_design_matrix:
            X = self.create_design_matrix(Xin)
        else:
            X = Xin
        n_regressors = X.shape[1]
        if self.fit_offset:
            Y_mean = np.mean(Y, axis=0)
            X_mean = np.mean(X, axis=0)
        else:
            Y_mean = np.zeros(n_features)
            X_mean = np.zeros(n_regressors)
        self.coefficients = np.zeros((n_regressors, n_features))
        X_centered = X - X_mean
        Y_centered = Y - Y_mean
        for i in range(n_features):
            if method == 'least squares':
                #beta[:,i] = np.squeeze(leastsquares(np.reshape(Y[:,i], (-1,1)), X))
                self.coefficients[:,i] = la.lstsq(X_centered, Y_centered[:,i])[0]
            elif method == 'gradient descent':
                self.coefficients[:,i] = np.squeeze(self.gradient_descent(X_centered, Y_centered[:,i]))
        if self.fit_offset:
            self.offset = Y_mean - X_mean.dot(self.coefficients)
        self.training_loss = self.compute_loss_percentage(Y, self.reconstruct(Xin))

    # def leastsquares(self, y, phi):
    #     u,s,v = la.svd(phi, full_matrices=False)
    #     return (v*1/s).dot(u.T.dot(y))

    def gradient_descent(self, X, y, start=None, learning_rate=0.1, tolerance=0.00001):
        n_samples = y.size
        if start is None:
            coefficients = np.zeros(X.shape[1])
        gradient = -2./n_samples*(y - X.dot(coefficients)).dot(X)
        while la.norm(gradient, np.inf) >= tolerance:
            gradient = 2./n_samples*(X.dot(coefficients) - y).dot(X)
            coefficients -= learning_rate*gradient
        return coefficients

    def reconstruct(self, Xin):
        if self.use_design_matrix:
            X = self.create_design_matrix(Xin)
        else:
            X = Xin
        if self.fit_offset:
            return X.dot(self.coefficients) + self.offset
        else:
            return X.dot(self.coefficients)

    def compute_loss_percentage(self, Y, Y_recon):
        return np.sum((Y - Y_recon)**2, axis=0)/Y.shape[0]/np.var(Y, axis=0)

    # def zeropad(x, n_zeros=1):
    #     if len(x.shape) == 1:
    #         xout = np.zeros(x.shape[0] + 2*n_zeros)
    #         xout[n_zeros:-n_zeros] = x
    #     else:
    #         xout = np.zeros((x.shape[0] + 2*n_zeros,x.shape[1]))
    #         xout[n_zeros:-n_zeros,:] = x
    #     return xout
    #
    # # Note: This function fits a linear regression in the case where you only have one regressor
    # # and want to find the function G that is convolved with your regressor.
    # def fit_lr_analytic(self, Y, x):
    #     Ypad = self.zeropad(Y, n_zeros=100)
    #     Xpad = self.zeropad(x, n_zeros=100)
    #     Yft = np.fft.rfft(Ypad, axis=0)
    #     Xft = np.fft.rfft(Xpad)
    #     G = np.fft.irfft((Yft.T/Xft).T, axis=0)
    #     return G


class recurrent_regression:
    def __init__(self, fit_offset=True, use_design_matrix=False, convolution_length=1, recurrent_convolution_length=1):
        self.fit_offset = fit_offset
        self.use_design_matrix = use_design_matrix
        if fit_offset:
            self.offset = None
        if use_design_matrix:
            self.convolution_length = convolution_length
            self.recurrent_convolution_length = recurrent_convolution_length
        self.coefficients = None
        self.training_loss = None

    def create_design_matrix(self, X):
        n_samples, n_regressors = X.shape
        if self.convolution_length > n_samples:
            raise ValueError("convolution_length=%d cannot be greater than n_samples=%d" % (self.convolution_length,n_samples))
        design_matrix = np.zeros((n_samples, n_regressors*self.convolution_length))
        for k in range(n_regressors):
            for j in range(self.convolution_length):
                design_matrix[j:, k*self.convolution_length + j] = X[0:n_samples-j, k]
        return design_matrix

    def fit(self, Y, Xin, method='least squares'):
        n_samples, n_features = Y.shape
        if self.use_design_matrix:
            X = self.create_design_matrix(np.concatenate(Xin, Y))
        else:
            X = np.concatenate(Xin, Y)
        n_regressors = X.shape[1]
        if self.fit_offset:
            Y_mean = np.mean(Y, axis=0)
            X_mean = np.mean(X, axis=0)
        else:
            Y_mean = np.zeros(n_features)
            X_mean = np.zeros(n_regressors)
        self.coefficients = np.zeros((n_regressors, n_features))
        X_centered = X - X_mean
        Y_centered = Y - Y_mean
        for i in range(n_features):
            idxs = np.concatenate((np.arange(self.convolution_length*(Xin.shape[1]+i)), np.arange(self.convolution_length*(Xin.shape[1]+i+1),n_regressors)))
            if method == 'least squares':
                self.coefficients[idxs,i] = la.lstsq(X_centered[:,idxs], Y_centered[:,i])[0]
        if self.fit_offset:
            self.offset = Y_mean - X_mean.dot(self.coefficients)
        self.training_loss = self.compute_loss_percentage(Y, self.reconstruct(Xin))

    def reconstruct(self, Y, Xin):
        if self.use_design_matrix:
            X = self.create_design_matrix(np.concatenate(Xin, Y))
        else:
            X = np.concatenate(Xin, Y)
        if self.fit_offset:
            return X.dot(self.coefficients) + self.offset
        else:
            return X.dot(self.coefficients)

    def compute_loss_percentage(self, Y, Y_recon):
        return np.sum((Y - Y_recon)**2, axis=0)/Y.shape[0]/np.var(Y, axis=0)
