import numpy as np
import scipy.linalg as la


class DynamicModeDecomposition:
    def __init__(self):
        self.dynamics_rank = None
        return

    def fit(self, X, U=None, dynamics_rank=None):
        if X.ndim == 3:
            # Format which allows for multiple trials - dimensions TRIALS x TIMESTEPS x FEATURES
            multiple_trials = True
            n_trials, n_timesteps, n_features = X.shape
            n_samples = n_trials*n_timesteps
            X_left = X[:,1:,:].reshape((n_samples-n_trials, n_features)).T
            X_right = X[:,:-1,:].reshape((n_samples-n_trials, n_features)).T
        elif X.ndim == 2:
            # One continuous time series - dimensions TIMESTEPS x FEATURES
            multiple_trials = False
            n_samples, n_features = X.shape
            X_left = X[1:].T
            X_right = X[:-1].T
        else:
            raise ValueError("data matrix must be 2 or 3 dimensions")
        if U is not None:
            if multiple_trials and U.ndim == 3:
                n_inputs = U.shape[2]
                # U_right = U[:,:-1,:].reshape((n_samples-1, n_inputs)).T
                Omega = np.concatenate((X_right, U[:,:-1,:].reshape((n_samples-n_trials, n_inputs)).T), axis=0)
        else:
            n_inputs = 0
            Omega = X_right
        if dynamics_rank is not None:
            self.dynamics_rank = dynamics_rank
        else:
            self.dynamics_rank = n_features+n_inputs
        U_right,s_right,V_right = la.svd(Omega, full_matrices=False)
        U1 = U_right[:n_features, :self.dynamics_rank]
        U2 = U_right[n_features:, :self.dynamics_rank]
        U_left,s_left,V_left = la.svd(X_left, full_matrices=False)
        # Compute matrix products that are reused for A and B
        tmp = np.dot(X_left, V_right[:self.dynamics_rank].T*(1./s_right[:self.dynamics_rank]))
        self.A = np.dot(tmp, U1.T)
        self.B = np.dot(tmp, U2.T)
        self.A_reduced = np.dot(U_left[:,:self.dynamics_rank].T, self.A)
        self.B_reduced = np.dot(U_left[:,:self.dynamics_rank].T, self.B)
        