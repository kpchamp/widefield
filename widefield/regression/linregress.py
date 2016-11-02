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
        else:
            self.convolution_length = 1
        self.coefficients = None
        self.training_loss = None


    # def create_design_matrix(self, X):
    #     n_samples, n_regressors = X.shape
    #     if self.convolution_length > n_samples:
    #         raise ValueError("convolution_length=%d cannot be greater than n_samples=%d" % (self.convolution_length,n_samples))
    #     if self.fit_offset:
    #         return np.concatenate((np.ones((n_samples-1,1)),
    #                                self.create_convolution_matrix(X[1:], self.convolution_length)))
    #     else:
    #         return self.create_convolution_matrix(X[1:], self.convolution_length)

    def create_design_matrix(self, X):
        n_samples, n_regressors = X.shape
        if self.convolution_length > n_samples:
            raise ValueError("convolution_length=%d cannot be greater than n_samples=%d" % (self.convolution_length,n_samples))
        design_matrix = np.zeros((n_samples, 1 + n_regressors*self.convolution_length))
        design_matrix[:,0] += 1.
        for k in range(n_regressors):
            for j in range(self.convolution_length):
                design_matrix[j:, 1 + k*self.convolution_length + j] = X[0:n_samples-j, k]
        return design_matrix

    def fit(self, Y, Xin, method='least squares'):
        n_samples, n_features = Y.shape
        if self.use_design_matrix:
            X = self.create_design_matrix(Xin)
        else:
            X = Xin
        n_regressors = X.shape[1]
        self.coefficients = np.zeros((n_regressors, n_features))
        for i in range(n_features):
            if method == 'least squares':
                self.coefficients[:,i] = la.lstsq(X, Y[:,i])[0]
            elif method == 'gradient descent':
                self.coefficients[:,i] = np.squeeze(self.gradient_descent(X, Y[:,i]))
        if self.fit_offset:
            self.offset = self.coefficients[0]
        self.training_loss = self.compute_loss_percentage(Y, self.reconstruct(Xin))

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
        return X.dot(self.coefficients)

    def compute_loss_percentage(self, Y, Y_recon):
        return np.mean((Y - Y_recon)**2, axis=0)/np.var(Y, axis=0)

    def create_convolution_matrix(self, X, convolution_length):
        n_samples, n_features = X.shape
        convolution_matrix = np.zeros((n_samples, n_features*convolution_length))
        for i in range(convolution_length):
            convolution_matrix += np.kron(np.eye(n_samples,k=-i).dot(X), np.eye(1,convolution_length,i))
        return convolution_matrix

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
        self.convolution_length = convolution_length
        self.recurrent_convolution_length = recurrent_convolution_length
        self.coefficients = None
        self.training_loss = None

    def create_design_matrix(self, X, Y):
        n_samples, n_regressors = X.shape
        n_features = Y.shape[1]
        if self.convolution_length > n_samples:
            raise ValueError("convolution_length=%d cannot be greater than n_samples=%d" % (self.convolution_length,n_samples))
        # if self.fit_offset:
        #     return np.concatenate((np.ones((n_samples-1,1)),
        #                            self.create_convolution_matrix(X[1:], self.convolution_length),
        #                            self.create_convolution_matrix(Y[:-1], self.recurrent_convolution_length)))
        # else:
        #     return np.concatenate((self.create_convolution_matrix(X[1:], self.convolution_length),
        #                            self.create_convolution_matrix(Y[:-1], self.recurrent_convolution_length)))
        design_matrix = np.zeros((n_samples-1, int(self.fit_offset) + n_regressors*self.convolution_length
                                  + n_features*self.recurrent_convolution_length))
        design_matrix[:,0] += 1.
        for k in range(n_regressors):
            for j in range(self.convolution_length):
                design_matrix[j:, 1 + k*self.convolution_length + j] = X[1:n_samples-j, k]
        for k in range(n_features):
            for j in range(self.recurrent_convolution_length):
                design_matrix[j:, 1 + n_regressors*self.convolution_length + k*self.recurrent_convolution_length + j] = Y[0:n_samples-j-1, k]
        return design_matrix

    def fit(self, Y, Xin, method='least squares', excludePairs=None):
        n_samples, n_features = Y.shape
        # if self.fit_offset:
        #     Y_mean = np.mean(Y, axis=0)
        #     X_mean = np.mean(Xin, axis=0)
        # else:
        #     Y_mean = np.zeros(n_features)
        #     X_mean = np.zeros(Xin.shape[1])
        # X_centered = Xin - X_mean
        # Y_centered = Y - Y_mean
        if self.use_design_matrix:
            X = self.create_design_matrix(Xin, Y)
        else:
            X = np.concatenate((Xin[1:], Y[:-1]), axis=1)
        n_regressors = X.shape[1]
        self.coefficients = np.zeros((n_regressors, n_features))
        for i in range(n_features):
            if excludePairs is None:
                idxs = np.concatenate((np.arange(int(self.fit_offset)+self.convolution_length*(Xin.shape[1])
                                                 +self.recurrent_convolution_length*i),
                                       np.arange(int(self.fit_offset)+self.convolution_length*(Xin.shape[1])
                                                 +self.recurrent_convolution_length*(i+1),n_regressors)))
            else:
                idxs = np.concatenate((np.arange(int(self.fit_offset)+self.convolution_length*(Xin.shape[1])
                                                 +self.recurrent_convolution_length*min(i,excludePairs[i])),
                                       np.arange(int(self.fit_offset)+self.convolution_length*(Xin.shape[1])
                                                 +self.recurrent_convolution_length*(min(i,excludePairs[i])+1),
                                                 int(self.fit_offset)+self.convolution_length*(Xin.shape[1])
                                                 +self.recurrent_convolution_length*(max(i,excludePairs[i]))),
                                       np.arange(int(self.fit_offset)+self.convolution_length*(Xin.shape[1])
                                                 +self.recurrent_convolution_length*(max(i,excludePairs[i])+1),
                                                 int(self.fit_offset)+n_regressors)))
            if method == 'least squares':
                self.coefficients[idxs,i] = la.lstsq(X[:,idxs], Y[1:,i])[0]
        if self.fit_offset:
            self.offset = self.coefficients[0]
        self.training_loss = self.compute_loss_percentage(Y[1:], self.reconstruct(Y, Xin))

    def reconstruct(self, Y, Xin):
        if self.use_design_matrix:
            X = self.create_design_matrix(Xin, Y)
        else:
            X = np.concatenate((Xin[1:], Y[:-1]), axis=1)
        return X.dot(self.coefficients)

    def compute_loss_percentage(self, Y, Y_recon):
        return np.mean((Y - Y_recon)**2, axis=0)/np.var(Y, axis=0)

    # def create_coefficient_convolution_matrix(self, feature_idx, regressor_idx, n_samples):
    #     r = np.zeros(n_samples)
    #     r[0:self.convolution_length] = self.coefficients[np.arange(self.convolution_length*regressor_idx,
    #                                                                self.convolution_length*(regressor_idx+1)), feature_idx]
    #     c = np.zeros(n_samples)
    #     c[0] = self.coefficients[self.convolution_length*regressor_idx, feature_idx]
    #     return la.toeplitz(c, r=r)

    # def create_convolution_matrix(self, X, convolution_length):
    #     n_samples, n_features = X.shape
    #     convolution_matrix = np.zeros((n_samples, n_features*convolution_length))
    #     for i in range(convolution_length):
    #         convolution_matrix += np.kron(np.eye(n_samples,k=-i).dot(X), np.eye(1,convolution_length,i))
    #     return convolution_matrix
