import numpy as np
import scipy.linalg as la


class LinearRegression:
    def __init__(self, fit_offset=True, convolution_length=1):
        self.fit_offset = fit_offset
        if fit_offset:
            self.offset = None
        self.convolution_length = convolution_length
        self.coefficients = None
        self.training_loss = None

    def create_design_matrix(self, X):
        n_samples, n_regressors = X.shape
        if self.convolution_length > n_samples:
            raise ValueError("convolution_length=%d cannot be greater than n_samples=%d" % (self.convolution_length,n_samples))
        design_matrix = np.zeros((n_samples, int(self.fit_offset) + n_regressors*self.convolution_length))
        design_matrix[:,0] += 1.
        for k in range(n_regressors):
            for j in range(self.convolution_length):
                design_matrix[j:, int(self.fit_offset) + k*self.convolution_length + j] = X[0:n_samples-j, k]
        return design_matrix

    def fit(self, Y, Xin, method='least squares'):
        n_samples, n_features = Y.shape
        X = self.create_design_matrix(Xin)
        n_regressors = X.shape[1]
        self.coefficients = np.zeros((n_regressors, n_features))
        for i in range(n_features):
            if method == 'least squares':
                self.coefficients[:,i] = la.lstsq(X, Y[:,i])[0]
            elif method == 'gradient descent':
                self.coefficients[:,i] = np.squeeze(self.gradient_descent(X, Y[:,i]))
        if self.fit_offset:
            self.offset = self.coefficients[0]
        self.training_loss = self.compute_loss_percentage(Y, Xin)

    def gradient_descent(self, X, y, start=None, learning_rate=0.1, tolerance=0.00001):
        raise NotImplementedError("haven't suffieciently tested this implementation")
        n_samples = y.size
        if start is None:
            coefficients = np.zeros(X.shape[1])
        gradient = -2./n_samples*(y - X.dot(coefficients)).dot(X)
        while la.norm(gradient, np.inf) >= tolerance:
            gradient = 2./n_samples*(X.dot(coefficients) - y).dot(X)
            coefficients -= learning_rate*gradient
        return coefficients

    def reconstruct(self, Xin):
        X = self.create_design_matrix(Xin)
        return X.dot(self.coefficients)

    def compute_loss_percentage(self, Y, Xin):
        return np.mean((Y - self.reconstruct(Xin))**2, axis=0)/np.var(Y, axis=0)

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


class DynamicRegression:
    def __init__(self, fit_offset=True, convolution_length=1, dynamic_convolution_length=1):
        self.fit_offset = fit_offset
        if fit_offset:
            self.offset = None
        self.convolution_length = convolution_length
        self.dynamic_convolution_length = dynamic_convolution_length
        self.coefficients = None
        self.training_loss = None

    def create_design_matrix(self, Y, X=None):
        if X is not None:
            n_inputs = X.shape[1]
        else:
            n_inputs = 0
        n_samples, n_features = Y.shape
        if self.convolution_length > n_samples:
            raise ValueError("convolution_length=%d cannot be greater than n_samples=%d" % (self.convolution_length, n_samples))
        design_matrix = np.zeros((n_samples-1, int(self.fit_offset) + n_inputs*self.convolution_length
                                  + n_features*self.dynamic_convolution_length))
        design_matrix[:,0] += 1.
        for k in range(n_inputs):
            for j in range(self.convolution_length):
                design_matrix[j:, int(self.fit_offset) + k*self.convolution_length + j] = X[0:n_samples-j-1, k]
        for k in range(n_features):
            for j in range(self.dynamic_convolution_length):
                design_matrix[j:, int(self.fit_offset) + n_inputs*self.convolution_length + k*self.dynamic_convolution_length + j] = Y[0:n_samples - j - 1, k]
        return design_matrix

    def fit(self, Y, Xin=None, method='least squares'):
        n_samples, n_features = Y.shape
        if Xin is not None:
            X = self.create_design_matrix(Y, X=Xin)
        else:
            X = self.create_design_matrix(Y)
        n_regressors = X.shape[1]
        self.coefficients = np.zeros((n_regressors, n_features))
        for i in range(n_features):
            if method == 'least squares':
                self.coefficients[:,i] = la.lstsq(X, Y[1:,i])[0]
        if self.fit_offset:
            self.offset = self.coefficients[0]
        self.training_loss = self.compute_loss_percentage(Y, Xin)

    def reconstruct(self, Y, Xin=None):
        if Xin is not None:
            X = self.create_design_matrix(Y, X=Xin)
        else:
            X = self.create_design_matrix(Y)
        return X.dot(self.coefficients)

    def compute_loss_percentage(self, Y, Xin=None):
        return np.mean((Y[1:] - self.reconstruct(Y, Xin))**2, axis=0)/np.var(Y[1:], axis=0)


class BilinearRegression:
    def __init__(self, fit_offset=True, convolution_length=1, dynamic_convolution_length=1,
                 bilinear_convolution_length=1):
        self.fit_offset = fit_offset
        if fit_offset:
            self.offset = None
        self.convolution_length = convolution_length
        self.dynamic_convolution_length = dynamic_convolution_length
        self.bilinear_convolution_length = bilinear_convolution_length
        self.coefficients = None
        self.training_loss = None

    def create_design_matrix(self, Y, X=None):
        if X is not None:
            n_inputs = X.shape[1]
        else:
            n_inputs = 0
        n_samples, n_features = Y.shape
        if self.convolution_length > n_samples:
            raise ValueError("convolution_length=%d cannot be greater than n_samples=%d" % (self.convolution_length, n_samples))
        design_matrix = np.zeros((n_samples-1, int(self.fit_offset) + n_inputs*self.convolution_length +
                                  n_features*self.dynamic_convolution_length +
                                  n_inputs*n_features*self.bilinear_convolution_length))
        design_matrix[:,0] += 1.
        for k in range(n_inputs):
            for j in range(self.convolution_length):
                design_matrix[j:, int(self.fit_offset) + k*self.convolution_length + j] = X[0:n_samples-j-1, k]
        for k in range(n_features):
            for j in range(self.dynamic_convolution_length):
                design_matrix[j:, int(self.fit_offset) + n_inputs*self.convolution_length +
                                  k*self.dynamic_convolution_length + j] = Y[0:n_samples - j - 1, k]
        for k in range(n_inputs):
            for j in range(n_features):
                for i in range(self.bilinear_convolution_length):
                    design_matrix[i:, int(self.fit_offset) + n_inputs*self.convolution_length +
                                      n_features*self.dynamic_convolution_length +
                                      (k*n_features+j)*self.bilinear_convolution_length +
                                      i] = X[0:n_samples-i-1, k] * Y[0:n_samples-i-1, j]
        return design_matrix

    def fit(self, Y, Xin=None, method='least squares'):
        n_samples, n_features = Y.shape
        if Xin is not None:
            X = self.create_design_matrix(Y, X=Xin)
        else:
            X = self.create_design_matrix(Y)
        n_regressors = X.shape[1]
        self.coefficients = np.zeros((n_regressors, n_features))
        for i in range(n_features):
            if method == 'least squares':
                self.coefficients[:,i] = la.lstsq(X, Y[1:,i])[0]
        if self.fit_offset:
            self.offset = self.coefficients[0]
        self.training_loss = self.compute_loss_percentage(Y, Xin)

    def reconstruct(self, Y, Xin=None):
        if Xin is not None:
            X = self.create_design_matrix(Y, X=Xin)
        else:
            X = self.create_design_matrix(Y)
        return X.dot(self.coefficients)

    def compute_loss_percentage(self, Y, Xin=None):
        return np.mean((Y[1:] - self.reconstruct(Y, Xin))**2, axis=0)/np.var(Y[1:], axis=0)


def create_convolution_matrix(X, convolution_length):
    n_samples, n_features = X.shape
    convolution_matrix = np.zeros((n_samples, n_features*convolution_length))
    for i in range(convolution_length):
        convolution_matrix += np.kron(np.eye(n_samples,k=-i).dot(X), np.eye(1,convolution_length,i))
    return convolution_matrix
