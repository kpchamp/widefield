import numpy as np
import scipy.linalg as la
from scipy.stats import multivariate_normal
from cvxopt import matrix, solvers
import warnings


class LinearGaussianSSM:
    # Linear Gaussian state space model with optional control
    def __init__(self, *args, **kwargs):
        if 'n_dim_state' in kwargs and 'n_dim_obs' in kwargs:
            self.n_dim_states = kwargs['n_dim_state']
            self.n_dim_observations = kwargs['n_dim_obs']
        elif 'C' in kwargs:
            self.n_dim_observations, self.n_dim_states = kwargs['C'].shape
        else:
            raise TypeError('must specify state space and observation space sizes in args')

        if 'A' in kwargs:
            self.A = kwargs['A']
        else:
            self.A = np.eye(self.n_dim_states)
        if 'C' in kwargs:
            self.C = kwargs['C']
        else:
            self.C = np.eye(self.n_dim_observations, self.n_dim_states)
        if 'Q' in kwargs:
            self.Q = kwargs['Q']
        else:
            self.Q = np.eye(self.n_dim_states)
        if 'R' in kwargs:
            self.R = kwargs['R']
        else:
            self.R = np.eye(self.n_dim_observations)
        if 'mu0' in kwargs:
            self.mu0 = kwargs['mu0']
        else:
            self.mu0 = np.zeros(self.n_dim_states)
        if 'V0' in kwargs:
            self.V0 = kwargs['V0']
        else:
            self.V0 = np.eye(self.n_dim_states)

        # Optional control
        if 'B' in kwargs:
            self.B = kwargs['B']
            self.n_dim_control = kwargs['B'].shape[0]
        elif 'n_dim_control' in kwargs:
            self.B = np.ones((self.n_dim_states, kwargs['n_dim_control']))
            self.n_dim_control = kwargs['n_dim_control']
        else:
            self.B = None

        self.mean_observation = None
        self.mean_input = None
        self.LL = None
        self.fitting_em = False

    # Fit the parameters of the LDS model using EM
    def fit_em(self, Yin, Uin=None, max_iters=10, tol=.01, exclude_list=None, diagonal_covariance=False):
        self.fitting_em = True
        n_samples = Yin.shape[1]
        self.mean_observation = np.mean(Yin, axis=1)
        if Uin is not None:
            self.mean_input = np.mean(Uin, axis=1)
        Y = (np.copy(Yin).T - self.mean_observation).T
        U = (np.copy(Uin).T - self.mean_input).T

        if exclude_list is None:
            exclude_list = []

        self.LL = []
        for i in range(max_iters):
            # E step - run Kalman smoothing algorithm
            mu_smooth, V_smooth, J = self.kalman_smoothing(Y,U)

            # self.LL[i] = self.complete_log_likelihood(Y, mu_smooth, U)
            # print self.LL[i]
            if i>0:
                LL_diff = self.LL[i] - self.LL[i-1]
                if LL_diff < 0:
                    warnings.warn("log likelihood increased on iteration %d - numerical instability or bug detected" % i, RuntimeWarning)
                if np.abs(LL_diff) < tol:
                    break

            # E step - compute expectations
            # Variables are defined as follows:
            #   Psum_all   = \sum_{t=1}^T P_t
            #   Psum1      = \sum_{t=1}^{T-1} P_t
            #   Psum2      = \sum_{t=2}^T P_t
            #   Psum_ttm1  = \sum_{t=2}^T P_{t,t-1}
            #   UXsum      = \sum_{t=1}^{T-1} \hat{x}_t u_t'
            #   UXsum_ttm1 = \sum_{t=2}^T \hat{x}_t u_{t-1}'
            #   Usum       = \sum_{t=1}^{T-1} u_t u_t'
            # P_t and P_{t,t-1} are as defined in Ghahramani - Parameter Estimation for LDSs
            Psum_all = np.zeros((self.n_dim_states, self.n_dim_states))
            Psum1 = np.zeros((self.n_dim_states, self.n_dim_states))
            Psum2 = np.zeros((self.n_dim_states, self.n_dim_states))
            Psum_ttm1 = np.zeros((self.n_dim_states, self.n_dim_states))
            for t in range(n_samples):
                P_t = V_smooth[t] + np.outer(mu_smooth[:,t],mu_smooth[:,t])
                Psum_all += P_t
                if t != 0:
                    Psum2 += P_t
                else:
                    P1 = P_t
                    # tmp_mean = np.mean(mu_smooth[:,t])
                    # P1 = V_smooth[t] + np.outer(mu_smooth[:,t] - tmp_mean, mu_smooth[:,t] - tmp_mean)
                    # modified to match Gharahmani code
                if t != n_samples-1:
                    Psum1 += P_t
                    # not sure why but needed to transpose first term to match with other code:
                    # Psum_ttm1 += J[t].dot(V_smooth[t+1]) + np.outer(mu_smooth[:,t+1],mu_smooth[:,t])
                    Psum_ttm1 += J[t].dot(V_smooth[t+1]).T + np.outer(mu_smooth[:,t+1],mu_smooth[:,t])
            if self.B is not None:
                UXsum = np.dot(mu_smooth[:,:-1], U[:,:-1].T)
                UXsum_ttm1 = np.dot(mu_smooth[:,1:], U[:,:-1].T)
                Usum = np.dot(U[:,:-1], U[:,:-1].T)
                # Murphy version???
                # UXsum = np.dot(mu_smooth[:,:-1], U[:,1:].T)
                # UXsum_ttm1 = np.dot(mu_smooth[:,1:], U[:,1:].T)
                # Usum = np.dot(U[:,1:], U[:,1:].T)

            # M step - update parameters
            if 'mu0' not in exclude_list:
                self.mu0 = mu_smooth[:,0]
            if 'V0' not in exclude_list:
                self.V0 = P1 - np.outer(mu_smooth[:,0],mu_smooth[:,0])   # NOTE: different from Ghamahrani code but seems consistent with paper
                # self.V0 = P1   # modified to match Ghamahrani code
            if 'C' not in exclude_list:
                self.C = Y.dot(mu_smooth.T).dot(la.inv(Psum_all))
                if 'R' not in exclude_list:
                    self.R = (Y.dot(Y.T) - self.C.dot(mu_smooth).dot(Y.T))/n_samples
            else:
                if 'R' not in exclude_list:
                    self.R = (Y.dot(Y.T) - np.dot(self.C,mu_smooth).dot(Y.T) - np.dot(Y,np.dot(self.C,mu_smooth).T) + self.C.dot(Psum_all).dot(self.C.T))/n_samples
                # self.R = np.diag(np.diag((Y.dot(Y.T) - self.C.dot(mu_smooth).dot(Y.T))/n_samples))
            if self.B is None:
                # Ghahramani version of updates
                if 'A' not in exclude_list:
                    self.A = Psum_ttm1.dot(la.inv(Psum1))
                    if 'Q' not in exclude_list:
                        self.Q = (Psum2 - self.A.dot(Psum_ttm1.T))/(n_samples-1.)
                else:
                    if 'Q' not in exclude_list:
                        self.Q = (Psum2 - self.A.dot(Psum_ttm1.T) - Psum_ttm1.dot(self.A) + self.A.dot(Psum1.dot(self.A.T)))/(n_samples-1.)
                    # self.Q = np.diag(np.diag((Psum2 - self.A.dot(Psum_ttm1.T))/(n_samples-1.)))
                # bishop version of updates
                # self.A = Psum_ttm1.dot(la.inv(Psum1))
                # self.C = Y.dot(Ez.T).dot(la.inv(Psum_all))
                # self.R = np.diag(Y.dot(Y.T) - self.C.dot(Ez).dot(Y.T) - Y.dot(Ez.T.dot(self.C.T)) + self.C.dot(Psum_all).dot(self.C.T))/n_samples
            else:
                if U is None:
                    raise ValueError('control term U must not be None')
                T1 = np.concatenate((np.concatenate((Psum1, UXsum), axis=1), np.concatenate((UXsum.T, Usum), axis=1)), axis=0)
                T2 = np.concatenate((Psum_ttm1, UXsum_ttm1), axis=1)
                T3 = np.dot(T2, la.inv(T1))
                # Atmp = (Psum_ttm1 - self.B.dot(UXsum.T)).dot(la.inv(Psum1))
                # Btmp = (UXsum_ttm1 - self.A.dot(UXsum)).dot(la.inv(Usum))
                if 'A' not in exclude_list:
                    self.A = T3[:,:self.n_dim_states]
                if 'B' not in exclude_list:
                    self.B = T3[:,self.n_dim_states:]
                if 'A' not in exclude_list and 'B' not in exclude_list:
                    if 'Q' not in exclude_list:
                        self.Q = (Psum2 - self.A.dot(Psum_ttm1.T) - self.B.dot(UXsum_ttm1.T))/(n_samples-1.)
                else:
                    if 'Q' not in exclude_list:
                        self.Q = (Psum2 - self.A.dot(Psum_ttm1.T) - Psum_ttm1.dot(self.A.T) + self.A.dot(Psum1).dot(self.A.T)
                                  - self.B.dot(UXsum_ttm1.T) - UXsum_ttm1.dot(self.B.T) + self.B.dot(UXsum.T).dot(self.A.T)
                                  + self.A.dot(UXsum).dot(self.B.T) + self.B.dot(Usum).dot(self.B.T))

            # optionally diagonalize the covariance matrices
            if diagonal_covariance:
                self.Q = self.Q*np.eye(self.Q.shape[0])
                self.R = self.R*np.eye(self.R.shape[0])

        self.fitting_em = False

    def kalman_filter(self, Y, U=None):
        n_samples = Y.shape[1]

        mu_filter = np.zeros((self.n_dim_states, n_samples))
        V_filter = np.zeros((n_samples, self.n_dim_states, self.n_dim_states))

        mu_predict = self.mu0
        # Murphy version - I don't think this is what I want
        # if self.B is not None:
        #     if U is None:
        #         raise ValueError('control term U must not be None')
        #     mu_predict += self.B.dot(U[:,0])
        V_predict = self.V0
        LL = 0
        const = (2.*np.pi)**(-self.n_dim_observations/2.)
        for t in range(n_samples):
            print >>open('progress.txt','a'), "filtering time %d" % t
            e = Y[:,t] - self.C.dot(mu_predict)

            # Invert S using dpotrf
            S = self.C.dot(V_predict).dot(self.C.T) + self.R
            # K = V_predict.dot(self.C.T).dot(la.lapack.flapack.dpotri(la.lapack.flapack.dpotrf(S)[0])[0])

            K = V_predict.dot(self.C.T).dot(la.inv(S))
            mu_filter[:,t] = mu_predict + K.dot(e)
            V_filter[t] = V_predict - K.dot(self.C).dot(V_predict)
            #LL += multivariate_normal.logpdf(e, mean=np.zeros(e.shape), cov=self.C.dot(V_predict).dot(self.C.T) + self.R)
            LL += multivariate_normal.logpdf(e, mean=np.zeros(e.shape), cov=S)
            if t != n_samples-1:
                mu_predict = self.A.dot(mu_filter[:,t])
                if self.B is not None:
                    if U is None:
                        raise ValueError('control term U must not be None')
                    mu_predict += self.B.dot(U[:,t])
                    # mu_predict += self.B.dot(U[:,t+1 ])   # Murphy version
                V_predict = self.A.dot(V_filter[t]).dot(self.A.T) + self.Q

        LL += n_samples*const
        if self.fitting_em:
            self.LL.append(LL)
        return mu_filter, V_filter

    def kalman_smoothing(self, Y, U=None):
        n_samples = Y.shape[1]
        mu_filter, V_filter = self.kalman_filter(Y,U)

        mu_smooth = np.zeros((self.n_dim_states, n_samples))
        V_smooth = np.zeros((n_samples, self.n_dim_states, self.n_dim_states))
        J = np.zeros((n_samples - 1, self.n_dim_states, self.n_dim_states))

        mu_smooth[:,-1] = mu_filter[:,-1]
        V_smooth[-1] = V_filter[-1]
        for t in range(n_samples-2,-1,-1):
            print >>open('progress.txt','a'), "smoothing time %d" % t
            mu_predict = self.A.dot(mu_filter[:,t])
            if self.B is not None:
                if U is None:
                    raise ValueError('control term U must not be None')
                mu_predict += self.B.dot(U[:,t])
            V_predict = self.A.dot(V_filter[t]).dot(self.A.T) + self.Q
            J[t] = V_filter[t].dot(self.A.T).dot(la.inv(V_predict))
            mu_smooth[:,t] = mu_filter[:,t] + J[t].dot(mu_smooth[:,t+1] - mu_predict)
            V_smooth[t] = V_filter[t] + J[t].dot(V_smooth[t+1] - V_predict).dot(J[t].T)

        return mu_smooth, V_smooth, J

    def sample(self, T, U=None, state_noise_samples=None, obs_noise_samples=None):
        # These parameters allow you to input pre-sampled noise.
        if state_noise_samples is None:
            state_noise_samples = np.random.multivariate_normal(np.zeros(self.n_dim_states), self.Q, T).T
        if obs_noise_samples is None:
            obs_noise_samples = np.random.multivariate_normal(np.zeros(self.n_dim_observations), self.R, T).T

        x = np.zeros((self.n_dim_states, T))
        y = np.zeros((self.n_dim_observations, T))

        x[:,0] = self.mu0
        # if self.B is not None:
        #     if U is None:
        #         raise ValueError('control term U must not be None')
        #     x[:,0] += self.B.dot(U[:,0])
        y[:,0] = self.C.dot(self.mu0) + obs_noise_samples[:,0]
        for t in range(1,T):
            x[:,t] = self.A.dot(x[:,t-1]) + state_noise_samples[:,t]
            if self.B is not None:
                if U is None:
                    raise ValueError('control term U must not be None')
                x[:,t] += self.B.dot(U[:,t-1])
            y[:,t] = self.C.dot(x[:,t]) + obs_noise_samples[:,t]

        return x, y

    def complete_log_likelihood(self, Y, X, U=None):
        n_samples = Y.shape[1]

        ll = 0.

        observation_diff = Y - np.dot(self.C,X)
        state_diff = X[:,1:] - np.dot(self.A,X[:,:-1])
        if self.B is not None:
            if U is None:
                raise ValueError('control term U must not be None')
            state_diff -= np.dot(self.B,U[:,1:])
            M0 = np.outer(X[:,0] - (self.mu0 + self.B.dot(U[:,0])), X[:,0] - (self.mu0 + self.B.dot(U[:,0])))
        else:
            M0 = np.outer(X[:,0] - self.mu0, X[:,0] - self.mu0)
        N = np.dot(observation_diff, observation_diff.T)
        M = np.dot(state_diff, state_diff.T)

        ll += np.linalg.slogdet(self.V0)[1] + np.trace(np.dot(M0, la.inv(self.V0)))
        ll += (n_samples-1)*np.linalg.slogdet(self.Q)[1] + np.trace(np.dot(M,la.inv(self.Q)))
        ll += n_samples*np.linalg.slogdet(self.R)[1] + np.trace(np.dot(N,la.inv(self.R)))
        ll *= -0.5

        return ll