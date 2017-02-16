import numpy as np
import scipy.linalg as la
from widefield.tools.alignment import reshape_trial_to_sequence, reshape_sequence_to_trial


class DynamicModeDecomposition:
    def __init__(self):
        self.dynamics_rank = None
        self.n_features = None
        self.n_inputs = None
        self.multiple_trials = None
        return

    def fit(self, X, U=None, dynamics_rank=None):
        self.multiple_trials = (X.ndim == 3)
        self.n_features = X.shape[-1]
        if U is not None:
            self.n_inputs = U.shape[-1]
            X_left, X_right, U_right = self.construct_data_matrices(X, U)
            Omega = np.concatenate((X_right, U_right), axis=0)
        else:
            self.n_inputs = 0
            X_left, X_right = self.construct_data_matrices(X)
            Omega = X_right
        # if X.ndim == 3:
        #     # Format which allows for multiple trials - dimensions TRIALS x TIMESTEPS x FEATURES
        #     self.multiple_trials = True
        #     n_samples, n_trials, self.n_features = X.shape
        #     X_left = reshape_trial_to_sequence(X, left_truncation=1).T
        #     X_right = reshape_trial_to_sequence(X, right_truncation=1).T
        # elif X.ndim == 2:
        #     # One continuous time series - dimensions TIMESTEPS x FEATURES
        #     self.multiple_trials = False
        #     n_trials = 1
        #     n_samples, self.n_features = X.shape
        #     X_left = X[1:].T
        #     X_right = X[:-1].T
        # else:
        #     raise ValueError("data matrix must be 2 or 3 dimensions")
        # if U is not None:
        #     if self.multiple_trials:
        #         if U.ndim == 3:
        #             self.n_inputs = U.shape[2]
        #             # U_right = U[:,:-1,:].reshape((n_samples-1, n_inputs)).T
        #             Omega = np.concatenate((X_right, reshape_trial_to_sequence(U, right_truncation=1).T), axis=0)
        #             # Omega = np.concatenate((X_right, U[:,:-1,:].reshape((n_samples-n_trials, self.n_inputs)).T), axis=0)
        #         else:
        #             raise ValueError("data matrix must be 3 dimensions")
        #     else:
        #         if U.ndim == 2:
        #             self.n_inputs = U.shape[1]
        #             Omega = np.concatenate((X_right, U[:-1,:].T), axis=0)
        #         else:
        #             raise ValueError("data matrix must be 2 dimensions")
        # else:
        #     self.n_inputs = 0
        #     n_inputs = 0
        #     Omega = X_right
        if dynamics_rank is not None:
            self.dynamics_rank = dynamics_rank
        else:
            self.dynamics_rank = self.n_features+self.n_inputs
        U_right,s_right,V_right = la.svd(Omega, full_matrices=False)
        U1 = U_right[:self.n_features, :self.dynamics_rank]
        U_left,s_left,V_left = la.svd(X_left, full_matrices=False)
        # Compute matrix products that are reused for A and B
        tmp = np.dot(X_left, V_right[:self.dynamics_rank].T*(1./s_right[:self.dynamics_rank]))
        self.dim_reduction_coefficients = U_left[:,:self.dynamics_rank]
        self.A = np.dot(tmp, U1.T)
        self.A_reduced = np.dot(U_left[:,:self.dynamics_rank].T, self.A)
        if self.n_inputs != 0:
            U2 = U_right[self.n_features:, :self.dynamics_rank]
            self.B = np.dot(tmp, U2.T)
            self.B_reduced = np.dot(U_left[:,:self.dynamics_rank].T, self.B)

    def reconstruct(self, X, U=None):
        if self.n_inputs > 0:
            X_left, X_right, U_right = self.construct_data_matrices(X, U)
            X_recon = np.dot(self.A, X_right) + np.dot(self.B, U_right)
        else:
            X_left, X_right = self.construct_data_matrices(X)
            X_recon = np.dot(self.A, X_right)
        X_dot = X_recon - X_right
        return X_recon, X_dot

    def calculate_rsquared(self, X, U=None):
        if self.n_inputs > 0:
            X_left, X_right, U_right = self.construct_data_matrices(X, U)
            X_recon = np.dot(self.A, X_right) + np.dot(self.B, U_right)
        else:
            X_left, X_right = self.construct_data_matrices(X)
            X_recon = np.dot(self.A, X_right)
        X_dot = X_recon - X_right
        X_dot_true = X_left - X_right
        return 1. - np.var(X_dot - X_dot_true)/np.var(X_dot_true)

    def construct_data_matrices(self, X, U=None):
        if self.multiple_trials:
            if X.ndim == 3:
                n_samples, n_trials, n_features = X.shape
                if n_features != self.n_features:
                    raise ValueError("wrong number of features")
                X_left = reshape_trial_to_sequence(X, left_truncation=1).T
                X_right = reshape_trial_to_sequence(X, right_truncation=1).T
            else:
                raise ValueError("data matrix must be 3 dimensions")
        else:
            if X.ndim == 2:
                n_samples, n_features = X.shape
                if n_features != self.n_features:
                    raise ValueError("wrong number of features")
                X_left = X[1:].T
                X_right = X[:-1].T
            else:
                raise ValueError("data matrix must be 2 dimensions")
        if self.n_inputs > 0:
            if U is None:
                raise ValueError("missing input matrix")
            if self.multiple_trials:
                if U.ndim == 3:
                    n_samples, n_trials, n_inputs = U.shape
                    if n_inputs != self.n_inputs:
                        raise ValueError("wrong number of inputs")
                    U_right = reshape_trial_to_sequence(X, right_truncation=1).T
                    # Omega = np.concatenate((X_right, U[:,:-1,:].reshape((self.n_samples-self.n_trials, n_inputs)).T), axis=0)
                else:
                    raise ValueError("data matrix must be 3 dimensions")
            else:
                if U.ndim == 2:
                    n_inputs = U.shape[1]
                    if n_inputs != self.n_inputs:
                        raise ValueError("wrong number of inputs")
                    U_right = U[:-1,:].T
                    # Omega = np.concatenate((X_right, U[:-1,:].T), axis=0)
                else:
                    raise ValueError("data matrix must be 2 dimensions")
            return X_left, X_right, U_right
        else:
            return X_left, X_right

