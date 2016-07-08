import numpy as np
import scipy.linalg as la
from scipy.stats import multivariate_normal
from cvxopt import matrix, solvers


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

            self.n_dim_obs, self.n_dim_state = self.C.shape
        elif len(args) == 2:
            # Initialize model where parameters are unknown. In this
            # case, parameters will be fit using EM.
            Y = args[0]
            self.n_dim_obs = Y.shape[0]
            self.n_dim_state = args[1]

            self.A = np.eye(self.n_dim_state)
            self.C = np.eye(self.n_dim_obs, self.n_dim_state)
            self.Q = np.eye(self.n_dim_state)
            self.R = np.ones(self.n_dim_obs)
            self.mu0 = np.zeros(self.n_dim_state)
            self.V0 = np.eye(self.n_dim_state)
        else:
            raise TypeError('Wrong number of arguments')

    # Fit the parameters of the LDS model using EM
    def fit_em(self, Y, max_iters=10):
        n_samples = Y.shape[1]

        for i in range(max_iters):
            # E step - run Kalman smoothing algorithm
            mu_smooth, V_smooth, J, LL = self.kalman_smoothing(Y)

            Psum_all = np.zeros((self.n_dim_state, self.n_dim_state))
            Psum1 = np.zeros((self.n_dim_state, self.n_dim_state))
            Psum2 = np.zeros((self.n_dim_state, self.n_dim_state))
            Psum_ttm1 = np.zeros((self.n_dim_state, self.n_dim_state))
            for t in range(n_samples):
                tmp1 = V_smooth[t] + np.outer(mu_smooth[:,t],mu_smooth[:,t])
                Psum_all += tmp1
                if t != 0:
                    Psum2 += tmp1
                else:
                    P1 = tmp1
                if t != n_samples-1:
                    Psum1 += tmp1
                    # not sure why but needed to transpose first term to match with other code:
                    # Psum_ttm1 += J[t].dot(V_smooth[t+1]) + np.outer(mu_smooth[:,t+1],mu_smooth[:,t])
                    Psum_ttm1 += J[t].dot(V_smooth[t+1]).T + np.outer(mu_smooth[:,t+1],mu_smooth[:,t])

            # M step - update parameters
            self.mu0 = mu_smooth[:,0]
            self.V0 = P1 - np.outer(mu_smooth[:,0],mu_smooth[:,0])
            # Ghahramani version of updates
            self.A = Psum_ttm1.dot(la.inv(Psum1))
            self.C = Y.dot(mu_smooth.T).dot(la.inv(Psum_all))
            self.R = np.diag((Y.dot(Y.T) - self.C.dot(mu_smooth).dot(Y.T))/n_samples)
            # bishop version of updates
            # self.A = Psum_ttm1.dot(la.inv(Psum1))
            # self.C = Y.dot(Ez.T).dot(la.inv(Psum_all))
            # self.R = np.diag(Y.dot(Y.T) - self.C.dot(Ez).dot(Y.T) - Y.dot(Ez.T.dot(self.C.T)) + self.C.dot(Psum_all).dot(self.C.T))/n_samples

    def fit_constrained(self, Y):
        # NOTE: this method fails because of a memory error in forming P
        n_samples = Y.shape[1]

        C, s, V = la.svd(Y, full_matrices=False)
        Z = s*V.T

        num_conditions = 0
        P = matrix(np.kron(np.eye(self.n_dim_obs),Y[:,0:-1].dot(Y[:,0:-1].T)))
        q = matrix(np.flatten(Y[:,0:-1].dot(Y[:,1:].T)))
        sol = solvers.qp(P,q)
        A = np.reshape(sol['x'], (self.n_dim_state,self.n_dim_state))
        rho = np.max(la.eigvals(A))

        while rho > 1:
            num_conditions += 1
            U,s,V = la.svd(A, full_matrices=False)
            Gold = G
            G = np.zeros(num_conditions, self.n_dim_state**2)
            G[0:-1,:] = Gold
            G[-1,:] = np.dot(U[:,0], V[:,0].T)
            h = matrix(np.ones(num_conditions))

            sol = solvers.qp(P,q,matrix(G),h)
            A = np.reshape(sol['x'], (self.n_dim_state,self.n_dim_state))
            rho = np.max(la.eigvals(A))

    def kalman_filter(self, Y):
        n_samples = Y.shape[1]

        mu_filter = np.zeros((self.n_dim_state, n_samples))
        V_filter = np.zeros((n_samples, self.n_dim_state, self.n_dim_state))

        mu_predict = self.mu0
        V_predict = self.V0
        LL = 0
        for t in range(n_samples):
            print >>open('progress.txt','a'), "filtering time %d" % t
            e = Y[:,t] - self.C.dot(mu_predict)

            # Invert S using dpotrf
            # S = self.C.dot(V_predict).dot(self.C.T) + self.R
            # K = V_predict.dot(self.C.T).dot(la.lapack.flapack.dpotri(la.lapack.flapack.dpotrf(S)[0])[0])

            # Invert S using matrix inversion lemma
            Vinv = la.inv(V_predict)
            # Vinv = la.lapack.flapack.dpotri(la.lapack.flapack.dpotrf(V_predict)[0])[0]   # incorrect
            Rinv = 1/self.R
            Sinv = np.diag(Rinv) - (self.C.T*Rinv).T.dot(la.inv(Vinv + (self.C.T*Rinv).dot(self.C))).dot(self.C.T*Rinv)
            # print np.max(Sinv - la.inv(self.C.dot(V_predict).dot(self.C.T) + np.diag(self.R)))

            K = V_predict.dot(self.C.T).dot(Sinv)
            mu_filter[:,t] = mu_predict + K.dot(e)
            V_filter[t] = V_predict - K.dot(self.C).dot(V_predict)
            # LL += multivariate_normal.logpdf(e, mean=np.zeros(e.shape), cov=self.C.dot(V_predict).dot(self.C.T) + np.diag(self.R))
            # LL += multivariate_normal.logpdf(e, mean=np.zeros(e.shape), cov=S)
            if t != n_samples:
                mu_predict = self.A.dot(mu_filter[:,t].T)
                V_predict = self.A.dot(V_filter[t]).dot(self.A.T) + self.Q

        return mu_filter, V_filter, LL

    def kalman_smoothing(self, Y):
        n_samples = Y.shape[1]
        mu_filter, V_filter, LL = self.kalman_filter(Y)

        mu_smooth = np.zeros((self.n_dim_state, n_samples))
        V_smooth = np.zeros((n_samples, self.n_dim_state, self.n_dim_state))
        J = np.zeros((n_samples-1, self.n_dim_state, self.n_dim_state))

        mu_smooth[:,-1] = mu_filter[:,-1]
        V_smooth[-1] = V_filter[-1]
        for t in range(n_samples-2,-1,-1):
            print >>open('progress.txt','a'), "smoothing time %d" % t
            mu_predict = self.A.dot(mu_filter[:,t])
            V_predict = self.A.dot(V_filter[t]).dot(self.A.T) + self.Q
            J[t] = V_filter[t].dot(self.A.T).dot(la.inv(V_predict))
            mu_smooth[:,t] = mu_filter[:,t] + J[t].dot(mu_smooth[:,t+1] - mu_predict)
            V_smooth[t] = V_filter[t] + J[t].dot(V_smooth[t+1] - V_predict).dot(J[t].T)

        return mu_smooth, V_smooth, J, LL

    def sample(self, T):
        state_noise_samples = np.random.multivariate_normal(np.zeros(self.n_dim_state), self.Q, T).T
        obs_noise_samples = np.random.multivariate_normal(np.zeros(self.n_dim_obs), np.diag(self.R), T).T

        x = np.zeros((self.n_dim_state, T))
        y = np.zeros((self.n_dim_obs, T))

        x[:,0] = self.mu0
        y[:,0] = self.C.dot(self.mu0) + obs_noise_samples[:,0]
        for t in range(1,T):
            x[:,t] = self.A.dot(x[:,t-1]) + state_noise_samples[:,t]
            y[:,t] = self.C.dot(x[:,t]) + obs_noise_samples[:,t]

        return x, y