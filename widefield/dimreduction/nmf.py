import numpy as np
from scipy.optimize import nnls
# from sklearn.decomposition import NMF
from sklearn.decomposition.cdnmf_fast import _update_cdnmf_fast


class NMF:
    def __init__(self, n_components=None, sparsity=None, sparsity_penalty=1., regularization=None, regularization_penalty=1.):
        self.n_components = n_components
        self.sparsity = sparsity
        self.sparsity_penalty = sparsity_penalty
        self.regularization = regularization
        self.regularization_penalty = regularization_penalty
        #raise NotImplementedError("NMF not implemented yet")

    def fit(self, X, shuffle=False, max_iter=200, tol=1e-4):
        # Fit X = W*H, implementing coordinate descent as in scikit-learn implementation
        n_samples, n_features = X.shape
        if self.n_components is None:
            self.n_components = min(n_samples, n_features)

        avg = np.sqrt(X.mean() / self.n_components)
        Ht = avg * np.random.randn(n_features, self.n_components)
        W = avg * np.random.randn(n_samples, self.n_components)
        np.abs(Ht, Ht)
        np.abs(W, W)

        l1_H, l2_H, l1_W, l2_W = 0, 0, 0, 0
        if self.sparsity in ('both', 'components'):
            l1_H = self.sparsity_penalty
        if self.sparsity in ('both', 'transformation'):
            l1_W = self.sparsity_penalty
        if self.regularization in ('both', 'components'):
            l2_H = self.regularization_penalty
        if self.regularization in ('both', 'transformation'):
            l2_W = self.regularization_penalty

        objective = np.inf
        for i in range(max_iter):
            violation = 0.

            # -------------- Update W --------------
            HHt = np.dot(Ht.T, Ht)
            XHt = np.dot(X, Ht)

            # L2 regularization corresponds to increase of the diagonal of HHt
            if l2_W != 0.:
                # adds l2_reg only on the diagonal
                HHt.flat[::self.n_components + 1] += l2_W
            # L1 regularization corresponds to decrease of each element of XHt
            if l1_W != 0.:
                XHt -= l1_W

            if shuffle:
                permutation = np.random.permutation(self.n_components)
            else:
                permutation = np.arange(self.n_components)
            permutation = np.asarray(permutation, dtype=np.intp)
            violation += _update_cdnmf_fast(W, HHt, XHt, permutation)

            objective_new = np.sum((X - np.dot(W,Ht.T))**2) + l1_H*np.sum(np.sum(np.abs(Ht),axis=1)**2) + l1_W*np.sum(np.sum(np.abs(W),axis=1)**2) + l2_H*np.sum(Ht**2) + l2_W*np.sum(W**2)
            if objective_new > objective:
                print "warning: objective value increased"
            objective = objective_new

            # -------------- Update H --------------
            WWt = np.dot(W.T, W)
            XWt = np.dot(X.T, W)

            # L2 regularization corresponds to increase of the diagonal of HHt
            if l2_H != 0.:
                # adds l2_reg only on the diagonal
                WWt.flat[::self.n_components + 1] += l2_H
            # L1 regularization corresponds to decrease of each element of XHt
            if l1_H != 0.:
                XWt -= l1_H

            if shuffle:
                permutation = np.random.permutation(self.n_components)
            else:
                permutation = np.arange(self.n_components)
            permutation = np.asarray(permutation, dtype=np.intp)
            violation += _update_cdnmf_fast(Ht, WWt, XWt, permutation)

            objective_new = np.sum((X - np.dot(W,Ht.T))**2) + l1_H*np.sum(np.sum(np.abs(Ht),axis=1)**2) + l1_W*np.sum(np.sum(np.abs(W),axis=1)**2) + l2_H*np.sum(Ht**2) + l2_W*np.sum(W**2)
            if objective_new > objective:
                print "warning: objective value increased"
            objective = objective_new

            if i == 0:
                violation_init = violation
            elif violation > violation_last:
                print("increase in violation")
            violation_last = violation

            if violation_init == 0:
                break

            if violation / violation_init <= tol:
                print("Converged at iteration", i + 1)
                break


        self.components = Ht
        return W

    def fit_nnls(self, Xin):
        # Fit X = W*H, using NNLS as in ``Spare Non-Negative Matrix Factorization for Clustering", Kim and Park
        n_samples, n_features = Xin.shape
        if self.n_components is None:
            self.n_components = min(n_samples, n_features)

        # Randomly initialize W and H
        avg = np.sqrt(Xin.mean() / self.n_components)
        H = avg * np.random.randn(n_features, self.n_components)
        W = avg * np.random.randn(n_samples, self.n_components)
        np.abs(H, H)
        np.abs(W, W)
        # Hn = np.sqrt(np.sum(H**2, axis=0))
        # H /= Hn
        # W *= Hn

        l1_H, l2_H, l1_W, l2_W = 0, 0, 0, 0
        if self.sparsity in ('both', 'components'):
            l1_H = self.sparsity_penalty
        if self.sparsity in ('both', 'transformation'):
            l1_W = self.sparsity_penalty
        if self.regularization in ('both', 'components'):
            l2_H = self.regularization_penalty
        if self.regularization in ('both', 'transformation'):
            l2_W = self.regularization_penalty

        objective = np.inf
        for i in range(self.max_iter):
            # ---------- UPDATE W ----------
            Wnew = np.empty(W.shape)
            for i in range(n_samples):
                Wnew[i] = nnls(np.concatenate((H, np.sqrt(l1_W)*np.ones((1,self.n_components)), np.sqrt(l2_W)*np.eye(self.n_components)),axis=0),
                     np.hstack((Xin[i], np.zeros(self.n_components+1))))[0]
            W = Wnew

            objective_new = np.sum((Xin - np.dot(W,H.T))**2) + l1_H*np.sum(np.sum(np.abs(H),axis=1)**2) + l1_W*np.sum(np.sum(np.abs(W),axis=1)**2) + l2_H*np.sum(H**2) + l2_W*np.sum(W**2)
            if objective_new > objective:
                print "warning: objective value increased"
            objective = objective_new

            # ---------- UPDATE H ----------
            Hnew = np.empty(H.shape)
            for i in range(n_features):
                Hnew[i] = nnls(np.concatenate((W, np.sqrt(l1_H)*np.ones((1,self.n_components)), np.sqrt(l2_H)*np.eye(self.n_components)),axis=0),
                     np.hstack((Xin[:,i], np.zeros(self.n_components+1))))[0]
            H = Hnew
            # normalize H
            # H = Hnew/np.sqrt(np.sum(Hnew**2, axis=0))

            objective_new = np.sum((Xin - np.dot(W,H.T))**2) + l1_H*np.sum(np.sum(np.abs(H),axis=1)**2) + l1_W*np.sum(np.sum(np.abs(W),axis=1)**2) + l2_H*np.sum(H**2) + l2_W*np.sum(W**2)
            if objective_new > objective:
                print "warning: objective value increased on iteration %d" % i
                break
            objective = objective_new

        self.components = H
        return W


class SparseNMF:
    # Reference: "Sparse NMF, half-baked or well done?"
    def __init__(self, n_components=None, cf='KL', beta=1.0, sparsity_penalty=0.0, max_iter=200, tol=1e-4):
        self.n_components = n_components
        self.sparsity_penalty = sparsity_penalty
        self.max_iter = max_iter
        self.beta = beta
        self.flr = 1e-9
        self.tol = 1e-4

    def fit(self, X, h_ind=None, w_ind=None, display=False):
        n_samples, n_features = X.shape
        if self.n_components is None:
            self.n_components = min(n_samples, n_features)

        # initialize W and H
        W = np.random.rand(n_samples, self.n_components)
        H = np.random.rand(self.n_components, n_features)

        # sparsity per matrix entry - still figuring out what this means
        # if length(params.sparsity) == 1
        # params.sparsity = ones(r, n) * params.sparsity;
        # elseif size(params.sparsity, 2) == 1
        # params.sparsity = repmat(params.sparsity, 1, n);
        # end
        sparsity = np.ones((self.n_components, n_features))*self.sparsity_penalty

        # Normalize the columns of W and rescale H accordingly
        Wn = np.sqrt(np.sum(W**2,axis=0))
        W /= Wn
        H = (H.T*Wn).T

        lam = np.maximum(np.dot(W,H), self.flr)
        last_cost = np.inf

        obj_div = np.zeros(self.max_iter)
        obj_cost = np.zeros(self.max_iter)

        if h_ind is None:
            h_ind = np.ones(self.n_components, dtype=int)
        if w_ind is None:
            w_ind = np.ones(self.n_components, dtype=int)
        update_h = np.sum(h_ind)
        update_w = np.sum(w_ind)

        for i in range(self.max_iter):
            # H updates
            if update_h > 0:
                if self.beta == 1:
                    dph = (np.sum(W[:, h_ind], axis=0) + sparsity.T).T
                    dph = np.maximum(dph, self.flr)
                    dmh = np.dot(W[:, h_ind].T, (X / lam))
                    H[h_ind] = H[h_ind, :]*dmh/dph
                elif self.beta == 2:
                    dph = np.dot(W[:, h_ind].T, lam) + sparsity
                    dph = np.maximum(dph, self.flr)
                    dmh = np.dot(W[:, h_ind].T,X)
                    H[h_ind] = H[h_ind, :]* dmh / dph
                else:
                    dph = np.dot(W[:, h_ind].T, lam**(self.beta - 1.0)) + sparsity
                    dph = np.maximum(dph, self.flr)
                    dmh = np.dot(W[:, h_ind].T, (X * lam**(self.beta - 2.0)))
                    H[h_ind, :] = H[h_ind, :] * dmh / dph

                # # Normalize the rows of H
                # H = (H.T/np.sqrt(np.sum(H**2,axis=1))).T
                lam = np.maximum(np.dot(W,H), self.flr)

            # W updates
            if update_w > 0:
                if self.beta == 1:
                    dpw = np.sum(H[w_ind, :], axis=1).T + np.sum(np.dot((X / lam), H[w_ind, :].T) * W[:, w_ind], axis=0)*W[:, w_ind]
                    dpw = np.maximum(dpw, self.flr)
                    dmw = np.dot(X/lam, H[w_ind, :].T) + np.sum(np.sum(H[w_ind, :],axis=1).T*W[:, w_ind],axis=0)*W[:, w_ind]
                    W[:, w_ind] = W[:,w_ind] * dmw / dpw
                elif self.beta == 2:
                    dpw = np.dot(lam, H[w_ind, :].T) + np.sum(np.dot(X, H[w_ind, :].T) * W[:, w_ind],axis=0)*W[:, w_ind]
                    dpw = np.maximum(dpw, self.flr)
                    dmw = np.dot(X, H[w_ind, :].T) + np.sum(np.dot(lam, H[w_ind, :].T) * W[:, w_ind],axis=0)*W[:, w_ind]
                    W[:, w_ind] = W[:,w_ind] * dmw / dpw
                else:
                    dpw = np.dot(lam**(self.beta - 1.),H[w_ind, :].T) + np.sum(np.dot(X * lam**(self.beta - 2.), H[w_ind, :].T) * W[:, w_ind],axis=0)*W[:, w_ind]
                    dpw = np.maximum(dpw, self.flr)
                    dmw = np.dot(X * lam**(self.beta - 2.), H[w_ind, :].T) + np.sum(np.dot(lam**(self.beta - 1.), H[w_ind, :].T) * W[:, w_ind], axis=0)*W[:, w_ind]
                    W[:, w_ind] = W[:,w_ind] * dmw / dpw

                # Normalize the columns of W
                W = W/np.sqrt(np.sum(W**2,axis=0))
                lam = np.maximum(np.dot(W,H), self.flr)

            # Compute the objective function
            if self.beta == 1:
                div = np.sum(X * np.log(X / lam) - X + lam)
            elif self.beta == 2:
                div = np.sum((X - lam)**2)
            elif self.beta == 0:
                div = np.sum(X / lam - np.log( X / lam) - 1.)
            else:
                div = np.sum(X**self.beta + (self.beta - 1.)*lam**self.beta - self.beta * X * lam**(self.beta - 1.)) / (self.beta * (self.beta - 1.))
            cost = div + np.sum(sparsity * H)

            obj_div[i] = div
            obj_cost[i] = cost

            if display:
                print "iteration %d div = %.3e cost = %.3e\n" % (i, div, cost)

            # Convergence check
            if i > 1:
                e = np.abs(cost - last_cost) / last_cost
            if cost >= last_cost:
                print "cost increased on iteration %d" % i
                break
            elif last_cost - cost < self.tol:
                break

            last_cost = cost

        self.components = H
        return W

    def infer_latent(self, X, w_ind=None):
        n_samples, n_features = X.shape
        W = np.random.rand(n_samples, self.n_components)
        W /= np.sqrt(np.sum(W**2,axis=0))
        H = self.components
        lam = np.maximum(np.dot(W,H), self.flr)

        if w_ind is None:
            w_ind = np.ones(self.n_components, dtype=int)

        for i in range(self.max_iter):
            if self.beta == 1:
                dpw = np.sum(H[w_ind, :], axis=1).T + np.sum(np.dot((X / lam), H[w_ind, :].T) * W[:, w_ind], axis=0)*W[:, w_ind]
                dpw = np.maximum(dpw, self.flr)
                dmw = np.dot(X/lam, H[w_ind, :].T) + np.sum(np.sum(H[w_ind, :],axis=1).T*W[:, w_ind],axis=0)*W[:, w_ind]
                W[:, w_ind] = W[:,w_ind] * dmw / dpw
            elif self.beta == 2:
                dpw = np.dot(lam, H[w_ind, :].T) + np.sum(np.dot(X, H[w_ind, :].T) * W[:, w_ind],axis=0)*W[:, w_ind]
                dpw = np.maximum(dpw, self.flr)
                dmw = np.dot(X, H[w_ind, :].T) + np.sum(np.dot(lam, H[w_ind, :].T) * W[:, w_ind],axis=0)*W[:, w_ind]
                W[:, w_ind] = W[:,w_ind] * dmw / dpw
            else:
                dpw = np.dot(lam**(self.beta - 1.),H[w_ind, :].T) + np.sum(np.dot(X * lam**(self.beta - 2.), H[w_ind, :].T) * W[:, w_ind],axis=0)*W[:, w_ind]
                dpw = np.maximum(dpw, self.flr)
                dmw = np.dot(X * lam**(self.beta - 2.), H[w_ind, :].T) + np.sum(np.dot(lam**(self.beta - 1.), H[w_ind, :].T) * W[:, w_ind], axis=0)*W[:, w_ind]
                W[:, w_ind] = W[:,w_ind] * dmw / dpw

            W /= np.sqrt(np.sum(W**2,axis=0))
            lam = np.maximum(np.dot(W,H), self.flr)

        return W

    def reconstruct(self, X, W=None):
        if W is None:
            W = self.infer_latent(X)
        return np.dot(W, self.components)