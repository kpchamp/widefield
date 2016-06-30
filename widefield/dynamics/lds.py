import numpy as np
import scipy.linalg as la
from scipy.stats import multivariate_normal


class lds_model:
    def __init__(self, *args, **kwargs):
        if len(args) == 6:
            # Initialize model where parameters are known.
            self.A = args[0]
            self.C = args[1]
            self.Q = args[2]
            self.R = args[3]
            self.mu0 = args[4]
            self.V0 = args[5]

            self.n_states = self.A.shape[0]
            self.n_observations = self.C.shape[1]
        elif len(args) == 2:
            # Initialize model where parameters are unknown. In this
            # case, parameters will be fit using EM.
            Y = args[0]
            self.n_observations = Y.shape[0]
            self.n_states = args[1]

            self.A = None
            self.C = None
            self.Q = None
            self.R = None
            self.mu0 = None
            self.V0 = None

            if 'max_iters' in kwargs:
                max_iters = kwargs['max_iters']
            else:
                max_iters = 1000

            self.fit_em(Y, max_iters)
        else:
            raise TypeError('Wrong number of arguments')

    # Fit the parameters of the LDS model using EM
    def fit_em(self, Y, max_iters):
        n_samples = Y.shape[1]

        self.A = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = 0.001*np.eye(self.n_states)
        self.R = 1*np.eye(self.n_observations)
        self.mu0 = np.array([8, 10, 1, 0])
        self.V0 = 1*np.eye(self.n_states)

        # initialize parameters
        # self.mu0 = np.random.rand(self.n_states)
        # self.V0 = np.random.rand(self.n_states, self.n_states)
        # self.A = np.random.rand(self.n_states, self.n_states)
        # #tmp = np.random.rand(self.n_states, self.n_states)
        # self.Q = np.eye(self.n_states)
        # self.C = np.random.rand(self.n_observations, self.n_states)
        # tmp = np.eye(self.n_observations)*np.random.rand(self.n_observations)
        # self.R = tmp**2

        for i in range(max_iters):
            # E step - run Kalman smoothing algorithm
            mu_smooth, V_smooth, J = self.kalman_smoothing(Y)
            Ez = mu_smooth
            # Ezttm1 = np.zeros((n_samples-1, self.n_states, self.n_observations))
            # Eztt = np.zeros((n_samples, self.n_states, self.n_observations))
            # for t in range(n_samples):
            #     if t != n_samples:
            #         Ezttm1[t] = J[t].dot(V_smooth[t+1]) + mu_smooth[:,t+1].dot(mu_smooth[:,t].T)
            #     Eztt[t] = V_smooth[t] + mu_smooth[:,t].dot(mu_smooth[:,t].T)

            Psum_all = np.zeros((self.n_states, self.n_states))
            Psum1 = np.zeros((self.n_states, self.n_states))
            Psum2 = np.zeros((self.n_states, self.n_states))
            Psum_ttm1 = np.zeros((self.n_states, self.n_states))
            for t in range(n_samples):
                tmp1 = V_smooth[t] + mu_smooth[:,t].dot(mu_smooth[:,t].T)
                Psum_all += tmp1
                if t != 0:
                    Psum2 += tmp1
                else:
                    P1 = tmp1
                if t != n_samples:
                    Psum1 += tmp1
                    if t != n_samples-1:
                        Psum_ttm1 += J[t].dot(V_smooth[t+1]) + mu_smooth[:,t+1].dot(mu_smooth[:,t].T)

            # M step - update parameters
            self.mu0 = Ez[:,0]
            self.V0 = P1 - Ez[:,0].dot(Ez[:,0].T)
            # Ghahramani version of updates
            self.A = Psum_ttm1.dot(la.inv(Psum1))
            self.Q = (Psum2 - self.A.dot(Psum_ttm1.T))/(n_samples-1)
            self.C = Y.dot(Ez.T).dot(la.inv(Psum_all))
            self.R = (Y.dot(Y.T) - self.C.dot(Ez).dot(Y.T))/n_samples
            # bishop version of updates
            # self.A = Psum_ttm1.dot(la.inv(Psum1))
            # self.Q = (Psum2 - self.A.dot(Psum_ttm1.T) - Psum_ttm1.dot(self.A) + self.A.dot(Psum1).dot(self.A.T))/(n_samples-1)
            # self.C = Y.dot(Ez.T).dot(la.inv(Psum_all))
            # self.R = (Y.dot(Y.T) - self.C.dot(Ez).dot(Y.T) - Y.dot(Ez.T.dot(self.C.T)) + self.C.dot(Psum_all).dot(self.C.T))/n_samples

    def kalman_filter(self, Y):
        n_samples = Y.shape[1]

        mu_filter = np.zeros((self.n_states, n_samples))
        V_filter = np.zeros((n_samples, self.n_states, self.n_states))

        mu_predict = self.mu0
        V_predict = self.V0
        LL = 0
        for t in range(n_samples):
            e = Y[:,t] - self.C.dot(mu_predict)
            S = self.C.dot(V_predict).dot(self.C.T) + self.R
            K = V_predict.dot(self.C.T).dot(la.inv(S))
            mu_filter[:,t] = mu_predict + K.dot(e)
            V_filter[t] = V_predict - K.dot(self.C).dot(V_predict)
            LL += multivariate_normal.logpdf(e, mean=np.zeros(e.shape), cov=S)
            if t != n_samples:
                mu_predict = self.A.dot(mu_filter[:,t].T)
                V_predict = self.A.dot(V_filter[t]).dot(self.A.T) + self.Q

        return mu_filter, V_filter, LL

    def kalman_smoothing(self, Y):
        n_samples = Y.shape[1]
        mu_filter, V_filter, LL = self.kalman_filter(Y)

        mu_smooth = np.zeros((self.n_states, n_samples))
        V_smooth = np.zeros((n_samples, self.n_states, self.n_states))
        J = np.zeros((n_samples-1, self.n_states, self.n_states))

        mu_smooth[:,-1] = mu_filter[:,-1]
        V_smooth[-1] = V_filter[-1]
        for t in range(n_samples-2,-1,-1):
            mu_predict = self.A.dot(mu_filter[:,t])
            V_predict = self.A.dot(V_filter[t]).dot(self.A.T) + self.Q
            J[t] = V_filter[t].dot(self.A.T).dot(la.inv(V_predict))
            mu_smooth[:,t] = mu_filter[:,t] + J[t].dot(mu_smooth[:,t+1] - mu_predict)
            V_smooth[t] = V_filter[t] + J[t].dot(V_smooth[t+1] - V_predict).dot(J[t].T)

        return mu_smooth, V_smooth, J