import numpy as np
import scipy.linalg as la
from widefield.tools.alignment import reshape_trial_to_sequence, reshape_sequence_to_trial


class LinearRegression:
    def __init__(self, fit_offset=True, convolution_length=1):
        self.fit_offset = fit_offset
        if fit_offset:
            self.offset = None
        self.convolution_length = convolution_length
        self.coefficients = None
        self.training_r2 = None
        self.multiple_trials = None

    # def create_design_matrix(self, X):
    #     n_samples, n_regressors = X.shape
    #     if self.convolution_length > n_samples:
    #         raise ValueError("convolution_length=%d cannot be greater than n_samples=%d" % (self.convolution_length,n_samples))
    #     design_matrix = np.zeros((n_samples, int(self.fit_offset) + n_regressors*self.convolution_length))
    #     design_matrix[:,0] += 1.
    #     for k in range(n_regressors):
    #         for j in range(self.convolution_length):
    #             design_matrix[j:, int(self.fit_offset) + k*self.convolution_length + j] = X[0:n_samples-j, k]
    #     return design_matrix

    def fit(self, Xin, Yin, method='least squares'):
        # Y is what we're trying to predict; X are what we use to predict
        self.multiple_trials = (Yin.ndim == 3)
        X, Y = self.construct_data_matrices(Xin, Yin)
        n_samples, n_outputs = Y.shape
        n_inputs = X.shape[1]
        self.coefficients = np.zeros((n_inputs, n_outputs))
        for i in range(n_outputs):
            if method == 'least squares':
                self.coefficients[:,i] = la.lstsq(X, Y[:,i])[0]
            elif method == 'gradient descent':
                self.coefficients[:,i] = np.squeeze(self.gradient_descent(X, Y[:,i]))
        if self.fit_offset:
            self.offset = self.coefficients[0]
        self.training_r2 = self.compute_rsquared(Xin, Yin)

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
        X = self.construct_data_matrices(Xin)
        Y_recon = X.dot(self.coefficients)
        if Xin.ndim == 3:
            return Y_recon.reshape((Xin.shape[0], Xin.shape[1], Y_recon.shape[-1]))
        else:
            return Y_recon

    def compute_rsquared(self, Xin, Yin, by_output=False):
        X, Y = self.construct_data_matrices(Xin, Yin)
        if by_output:
            return 1. - np.var(Y - X.dot(self.coefficients), axis=0)/np.var(Y, axis=0)
        else:
            return 1. - np.var(Y - X.dot(self.coefficients))/np.var(Y)

    def construct_data_matrices(self, Xin, Yin=None):
        if self.multiple_trials:
            if Xin.ndim != 3:
                raise ValueError("input matrix must be 3 dimensions")
            n_trials, n_samples, n_inputs = Xin.shape
            if self.convolution_length > n_samples:
                raise ValueError("convolution_length=%d cannot be greater than n_samples=%d" % (self.convolution_length,n_samples))
            X = np.zeros((n_trials*n_samples, int(self.fit_offset) + n_inputs*self.convolution_length))
            for i in range(n_trials):
                X[i*n_samples:(i+1)*n_samples] = self.create_convolution_matrix(Xin[i])
            if Yin is not None:
                if Yin.ndim != 3:
                    raise ValueError("output matrix must be 3 dimensions")
                Y = reshape_trial_to_sequence(Yin)
                return X, Y
            return X
        else:
            if Xin.ndim != 2:
                raise ValueError("input matrix must be 2 dimensions")
            n_samples, n_inputs = Xin.shape
            if self.convolution_length > n_samples:
                raise ValueError("convolution_length=%d cannot be greater than n_samples=%d" % (self.convolution_length,n_samples))
            X = self.create_convolution_matrix(Xin)
            if Yin is not None:
                if Yin.ndim != 2:
                    raise ValueError("output matrix must be 2 dimensions")
                Y = Yin
                return X, Y
            return X

    def create_convolution_matrix(self, X):
        n_samples, n_inputs = X.shape
        design_matrix = np.zeros((n_samples, int(self.fit_offset) + n_inputs*self.convolution_length))
        design_matrix[:,0] += 1.
        for k in range(n_inputs):
            for j in range(self.convolution_length):
                design_matrix[j:, int(self.fit_offset) + k*self.convolution_length + j] = X[0:n_samples-j, k]
        return design_matrix


class DynamicRegression:
    def __init__(self, fit_offset=True, convolution_length=1):
        self.fit_offset = fit_offset
        if fit_offset:
            self.offset = None
        self.convolution_length = convolution_length
        self.coefficients = None
        self.training_r2 = None

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

    def fit(self, Yin, Xin=None, method='least squares', input_inclusions=None):
        # inclusions should be a list of numpy arrays--list element i includes inputs to be included for output i
        self.multiple_trials = (Yin.ndim == 3)
        output_matrix, input_matrix = self.construct_data_matrices(Yin, Xin)
        n_outputs = output_matrix.shape[1]
        n_inputs = input_matrix.shape[1]
        self.coefficients = np.zeros((n_inputs, n_outputs))
        for i in range(n_outputs):
            if method == 'least squares':
                if input_inclusions is None:
                    self.coefficients[:,i] = la.lstsq(input_matrix, output_matrix[:,i])[0]
                else:
                    if input_inclusions[i].size == 0:
                        print "no inclusions for ",i
                        input_idxs = np.concatenate((np.arange(n_outputs),np.array([n_inputs-1])))
                        if i==10:
                            print input_idxs
                        self.coefficients[input_idxs,i] = la.lstsq(input_matrix[:,input_idxs], output_matrix[:,i])[0]
                    else:
                        input_idxs = np.arange(n_outputs)
                        for j in input_inclusions[i]:
                            input_idxs = np.concatenate((input_idxs,n_outputs+np.arange(j*self.convolution_length,(j+1)*self.convolution_length)))
                        input_idxs = np.concatenate((input_idxs,np.array([n_inputs-1])))
                        print input_idxs
                        self.coefficients[input_idxs,i] = la.lstsq(input_matrix[:,input_idxs], output_matrix[:,i])[0]
        if self.fit_offset:
            self.offset = self.coefficients[0]
        self.training_r2 = self.compute_rsquared(Yin, Xin)

    def reconstruct(self, Yin, Xin=None):
        output_matrix, input_matrix = self.construct_data_matrices(Yin, Xin)
        output_reconstructed = np.dot(input_matrix, self.coefficients)
        if Yin.ndim == 3:
            n_trials = Yin.shape[0]
            return reshape_sequence_to_trial(output_reconstructed, n_trials)
        else:
            return output_reconstructed

    def compute_rsquared(self, Yin, Xin=None):
        if Yin.ndim == 3:
            true_increments = Yin[:,1:,:] - Yin[:,:-1,:]
            output_reconstructed = self.reconstruct(Yin, Xin)
            predicted_increments = output_reconstructed - Yin[:,:-1,:]
        else:
            true_increments = Yin[1:] - Yin[:-1]
            output_reconstructed = self.reconstruct(Yin, Xin)
            predicted_increments = output_reconstructed - Yin[:-1]
        return 1. - np.var(predicted_increments - true_increments)/np.var(true_increments)

    def compute_rsquared_data(self, Yin, Xin=None):
        output_reconstructed = self.reconstruct(Yin, Xin)
        if Yin.ndim == 3:
            return 1. - np.var(Yin[:,1:,:] - output_reconstructed)/np.var(Yin[:,1:,:])
        else:
            return 1. - np.var(Yin[1:] - output_reconstructed)/np.var(Yin[1:])

    def construct_data_matrices(self, Yin, Xin=None):
        if self.multiple_trials:
            if Yin.ndim != 3:
                raise ValueError("data matrix must be 3 dimensions")
            n_trials, n_samples, n_outputs = Yin.shape
            Yleft = reshape_trial_to_sequence(Yin[:,1:,:])
            Yright = reshape_trial_to_sequence(Yin[:,:-1,:])
            if Xin is not None:
                n_inputs = Xin.shape[2]
                if self.convolution_length > n_samples:
                    raise ValueError("convolution_length=%d cannot be greater than n_samples=%d" % (self.convolution_length,n_samples))
                X = np.zeros((n_trials*(n_samples-1), int(self.fit_offset) + n_inputs*self.convolution_length))
                for i in range(n_trials):
                    X[i*(n_samples-1):(i+1)*(n_samples-1)] = self.create_convolution_matrix(Xin[i])[:-1]
                return Yleft, np.concatenate((Yright, X), axis=1)
            else:
                if self.fit_offset:
                    return Yleft, np.concatenate((Yright, np.ones((Yright.shape[0],1))),axis=1)
                else:
                    return Yleft, Yright
        else:
            if Yin.ndim != 2:
                raise ValueError("input matrix must be 2 dimensions")
            n_samples, n_outputs = Yin.shape
            Yleft = Yin[1:]
            Yright = Yin[:-1]
            if Xin is not None:
                if self.convolution_length > n_samples:
                    raise ValueError("convolution_length=%d cannot be greater than n_samples=%d" % (self.convolution_length,n_samples))
                X = self.create_convolution_matrix(Xin)
                return Yleft, np.concatenate((Yright, X), axis=1)
            else:
                if self.fit_offset:
                    return Yleft, np.concatenate((Yright, np.ones((Yright.shape[0],1))),axis=1)
                else:
                    return Yleft, Yright

    def create_convolution_matrix(self, X):
        n_samples, n_inputs = X.shape
        design_matrix = np.zeros((n_samples, int(self.fit_offset) + n_inputs*self.convolution_length))
        design_matrix[:,0] += 1.
        for k in range(n_inputs):
            for j in range(self.convolution_length):
                design_matrix[j:, int(self.fit_offset) + k*self.convolution_length + j] = X[0:n_samples-j, k]
        return design_matrix

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
        self.training_r2 = None

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
                                      i] = X[0:n_samples-i-1, k] * Y[i:n_samples-1, j]
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
        self.training_r2 = self.compute_rsquared(Y, Xin)

    def reconstruct(self, Y, Xin=None):
        if Xin is not None:
            X = self.create_design_matrix(Y, X=Xin)
        else:
            X = self.create_design_matrix(Y)
        return X.dot(self.coefficients)

    def compute_rsquared(self, Y, Xin=None, by_region=False):
        true_increments = Y[1:] - Y[:-1]
        Y_recon = self.reconstruct(Y, Xin)
        Y_dot = Y_recon - Y[:-1]
        if by_region:
            return 1. - np.var(Y_dot - true_increments, axis=0)/np.var(true_increments, axis=0)
        else:
            return 1. - np.var(Y_dot - true_increments)/np.var(true_increments)

    def compute_rsquared_data(self, Y, Xin=None, by_region=False):
        Y_recon = self.reconstruct(Y, Xin)
        if by_region:
            return 1. - np.var(Y_recon - Y[1:], axis=0)/np.var(Y[1:], axis=0)
        else:
            return 1. - np.var(Y_recon - Y[1:])/np.var(Y[1:])


def create_convolution_matrix(X, convolution_length):
    n_samples, n_features = X.shape
    convolution_matrix = np.zeros((n_samples, n_features*convolution_length))
    for i in range(convolution_length):
        convolution_matrix += np.kron(np.eye(n_samples,k=-i).dot(X), np.eye(1,convolution_length,i))
    return convolution_matrix
