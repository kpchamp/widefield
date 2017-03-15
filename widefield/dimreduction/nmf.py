import numpy as np
import scipy.linalg as la
from scipy.optimize import nnls
# from sklearn.decomposition import NMF
from sklearn.decomposition.cdnmf_fast import _update_cdnmf_fast
from sklearn.decomposition.nmf import _initialize_nmf


class NMF:
    def __init__(self, n_components=None, sparsity=None, sparsity_penalty=1., regularization=None, regularization_penalty=1.):
        self.n_components = n_components
        self.sparsity = sparsity
        self.sparsity_penalty = sparsity_penalty
        self.regularization = regularization
        self.regularization_penalty = regularization_penalty
        #raise NotImplementedError("NMF not implemented yet")

    def fit(self, X, shuffle=False, max_iter=200, tol=1e-4, verbose=False, W=None, H=None):
        # Fit X = W*H, implementing coordinate descent as in scikit-learn implementation
        n_samples, n_features = X.shape
        if self.n_components is None:
            self.n_components = min(n_samples, n_features)

        # Initialize using sklearn method
        W, Ht = _initialize_nmf(X, self.n_components)
        H = (Ht.T).copy(order='C')
        # Determine whether or not to initialize matrices randomly
        # avg = np.sqrt(X.mean() / self.n_components)
        # if H is None:
        #     H = avg * np.random.randn(n_features, self.n_components)
        #     np.abs(H, H)
        # else:
        #     H = np.copy(H)
        # if W is None:
        #     W = avg * np.random.randn(n_samples, self.n_components)
        #     np.abs(W, W)
        # else:
        #     W = np.copy(W)

        l1_H, l2_H, l1_W, l2_W = 0, 0, 0, 0
        if self.sparsity in ('both', 'components'):
            l1_H = self.sparsity_penalty
        if self.sparsity in ('both', 'transformation'):
            l1_W = self.sparsity_penalty
        if self.regularization in ('both', 'components'):
            l2_H = self.regularization_penalty
        if self.regularization in ('both', 'transformation'):
            l2_W = self.regularization_penalty

        for i in range(max_iter):
            violation = 0.

            # Update W
            violation += self.nmf_iteration_update(X, W, H, l1_W, l2_W, shuffle)

            # objective_new = np.sum((X - np.dot(W,Ht.T))**2) + l1_H*np.sum(np.sum(np.abs(Ht),axis=1)**2) + l1_W*np.sum(np.sum(np.abs(W),axis=1)**2) + l2_H*np.sum(Ht**2) + l2_W*np.sum(W**2)
            # if objective_new > objective:
            #     print "warning: objective value increased"
            # objective = objective_new

            # Update H
            violation += self.nmf_iteration_update(X.T, H, W, l1_H, l2_H, shuffle)

            if i == 0:
                violation_init = violation

            if violation_init == 0:
                break

            if verbose:
                print("violation:", violation / violation_init)

            if violation / violation_init <= tol:
                print("Converged at iteration", i + 1)
                break

        self.components = H
        return W

    def nmf_iteration_update(self, X, W, H, l1_reg, l2_reg, shuffle):
        n_components = H.shape[1]

        HHt = np.dot(H.T, H)
        XHt = np.dot(X, H)

        # L2 regularization corresponds to increase of the diagonal of HHt
        if l2_reg != 0.:
            # adds l2_reg only on the diagonal
            HHt.flat[::n_components + 1] += l2_reg
        # L1 regularization corresponds to decrease of each element of XHt
        if l1_reg != 0.:
            XHt -= l1_reg

        if shuffle:
            permutation = np.random.permutation(n_components)
        else:
            permutation = np.arange(n_components)
        # The following seems to be required on 64-bit Windows w/ Python 3.5.
        permutation = np.asarray(permutation, dtype=np.intp)
        return _update_cdnmf_fast(W, HHt, XHt, permutation)

    def infer_latent(self, X, W=None, max_iter=100, shuffle=False, tol=1e-4, verbose=False):
        # Fit X = W*H, using H as already found from fitting above
        n_samples, n_features = X.shape
        if self.n_components is None:
            self.n_components = min(n_samples, n_features)

        # Determine whether or not to initialize W randomly
        avg = np.sqrt(X.mean() / self.n_components)
        if W is None:
            W = avg * np.random.randn(n_samples, self.n_components)
            np.abs(W, W)
        else:
            W = np.copy(W)

        l1_W, l2_W = 0, 0
        if self.sparsity in ('both', 'transformation'):
            l1_W = self.sparsity_penalty
        if self.regularization in ('both', 'transformation'):
            l2_W = self.regularization_penalty

        for i in range(max_iter):
            violation = 0.

            # Update W
            violation += self.nmf_iteration_update(X, W, self.components, l1_W, l2_W, shuffle)

            if i == 0:
                violation_init = violation

            if violation_init == 0:
                break

            if verbose:
                print("violation:", violation / violation_init)

            if violation / violation_init <= tol:
                print("Converged at iteration", i + 1)
                break

        return W

    def reconstruct(self, X, W=None):
        if W is None:
            W = self.infer_latent(X)
        return np.dot(W, self.components.T)

    def initialize_nmf(self, X, n_components):
        n_samples, n_features = X.shape
        W = np.empty((n_samples, n_components))
        H = np.empty((n_features, n_components))

        U,s,V = la.svd(X, full_matrices=False)
        W[:,0] = np.sqrt(s[0])*np.abs(U[:,0])
        H[:,0] = np.sqrt(s[0])*np.abs(V[0])
        for i in range(1, n_components):
            x = U[:,i]
            y = V[i]
            xp = np.maximum(x,0)
            xn = np.abs(np.minimum(x,0))
            yp = np.maximum(y,0)
            yn = np.abs(np.minimum(y,0))
            xpnrm = np.sqrt(np.sum(xp**2))
            ypnrm = np.sqrt(np.sum(yp**2))
            mp = xpnrm*ypnrm
            xnnrm = np.sqrt(np.sum(xn**2))
            ynnrm = np.sqrt(np.sum(yn**2))
            mn = xnnrm*ynnrm
            if mp > mn:
                u = xp/xpnrm
                v = yp/ypnrm
                sigma = mp
            else:
                u = xn/xnnrm
                v = yn/ynnrm
                sigma = mn
            W[:,i] = np.sqrt(s[i]*sigma)*u
            H[:,i] = np.sqrt(s[i]*sigma)*v

        return W, H

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


class SemiNMF:
    def __init__(self, n_components=None, sparsity_penalty=1., regularization_penalty=1.):
        self.n_components = n_components
        self.sparsity_penalty = sparsity_penalty
        self.regularization_penalty = regularization_penalty
        #raise NotImplementedError("NMF not implemented yet")

    def fit(self, X, shuffle=False, max_iter=200, tol=1e-4, verbose=False, W=None, H=None):
        # Fit X = W*H, implementing coordinate descent as in scikit-learn implementation
        n_samples, n_features = X.shape
        if self.n_components is None:
            self.n_components = min(n_samples, n_features)

        # Initialize matrices
        W, H = self.initialize_nmf(X, self.n_components)
        # avg = np.sqrt(X.mean() / self.n_components)
        # if H is None:
        #     H = avg * np.random.randn(n_features, self.n_components)
        #     np.abs(H, H)
        # else:
        #     H = np.copy(H)
        # if W is None:
        #     W = avg * np.random.randn(n_samples, self.n_components)
        # else:
        #     W = np.copy(W)

        lambda1 = self.sparsity_penalty
        lambda2 = self.regularization_penalty

        obj_last = np.sum((X - np.dot(W,H.T))**2)/2. + lambda2/2.*np.sum(W**2) + lambda1*np.sum(np.abs(H))
        for i in range(max_iter):
            grad_W = np.dot(np.dot(W,H.T) - X, H)
            lipschitz_W = np.sum(np.dot(H,H.T)**2)

            W = (W - 1./lipschitz_W*grad_W)/(1. + lambda2/lipschitz_W)

            grad_H = np.dot(W.T, np.dot(W,H.T) - X).T
            lipschitz_H = np.sum(np.dot(W,W.T)**2)

            H = np.maximum((H - 1./lipschitz_H*grad_H) - lambda1/lipschitz_H, 0.0)

            obj = np.sum((X - np.dot(W,H.T))**2)/2. + lambda2/2.*np.sum(W**2) + lambda1*np.sum(np.abs(H))
            if obj_last - obj < 0:
                print "warning: objective function increased on iteration %d" % i
            if obj_last - obj < tol:
                print "converged on iteration %d" % i
            if verbose:
                print obj_last - obj
            obj_last = obj

        self.components = H
        return W

    def initialize_nmf(self, X, n_components):
        n_samples, n_features = X.shape
        W = np.empty((n_samples, n_components))
        H = np.empty((n_features, n_components))

        U,s,V = la.svd(X, full_matrices=False)
        W[:,0] = np.sqrt(s[0])*U[:,0]
        H[:,0] = np.sqrt(s[0])*np.abs(V[0])
        for i in range(1, n_components):
            y = V[i]
            yp = np.maximum(y,0)
            yn = np.abs(np.minimum(y,0))
            ypnrm = np.sqrt(np.sum(yp**2))
            ynnrm = np.sqrt(np.sum(yn**2))
            if ypnrm > ynnrm:
                v = yp/ypnrm
                sigma = ypnrm
            else:
                v = yn/ynnrm
                sigma = ynnrm
            W[:,i] = np.sqrt(s[i]*sigma)*U[:,i]
            H[:,i] = np.sqrt(s[i]*sigma)*v

        return W,H

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