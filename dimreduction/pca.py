import numpy as np
import scipy.linalg as la
import scipy.special as special

# center columns
def centerCols(X):
    mu = np.mean(X, axis=0)
    return X - mu


class pca_model:
    def __init__(self, X, n_components=None, fitWith='SVD', max_iters=1000):
        self.n_samples, self.n_features = X.shape

        # Center data
        self.mean = np.mean(X, axis=0)
        X -= self.mean

        if n_components is None:
            self.n_components = min(self.n_features,self.n_samples)
        elif not 0 <= n_components <= self.n_features:
            raise ValueError("n_components=%r invalid for n_features=%d" % (n_components, self.n_features))
        else:
            self.n_components = n_components

        self.evals = None
        self.evecs = None
        self.components = None
        self.EMerr = None

        if fitWith == 'SVD':
            self.fitSVD(X)
        elif fitWith == 'EM':
            self.fitEM(X, max_iters)
        elif fitWith == 'EM_noconstraint':
            self.fitEM_noconstraint(X, max_iters)

    def fitSVD(self, X):
        U, S, V = la.svd(X, full_matrices=False)
        self.evals = (S ** 2) / self.n_samples
        self.evecs = V.T
        self.components = self.evecs[:,:self.n_components]

    def fitEM(self, X, max_iters):
        # NOTE: PCA model fit with EM may not have all values for evals and evecs
        W = np.random.randn(self.n_features, self.n_components)

        err = []
        for i in range(0, max_iters):
            # new constrained algorithm - gives W orthogonal but not orthonormal
            Minv = la.lapack.flapack.dtrtri(np.tril(np.dot(W.T,W)).T)[0].T
            XW = np.dot(X,W)
            Z = np.dot(Minv, XW.T)
            ZZt = np.triu(np.dot(Z,Z.T))
            SW = np.dot(X.T, XW)
            W = np.dot(SW, np.dot(Minv.T, la.lapack.flapack.dtrtri(ZZt)[0]))

            err.append(la.norm(X.T - np.dot(W, Z)))
            if i > 0 and np.abs(err[i] - err[i-1]) < 1e-6:
                break

        self.components = W/np.sqrt(np.sum(np.abs(W)**2,axis=0))   # normalize
        self.evals = np.diag(np.dot(np.dot(W.T,X.T),np.dot(X,W)))/self.n_samples
        self.EMerr = np.array(err)

    def fitEM_noconstraint(self, X, max_iters):
        # NOTE: PCA model fit with EM will not have values for evals and evecs
        W = np.random.randn(self.n_features, self.n_components)

        err = []
        for i in range(0, max_iters):
            U,S,V = la.svd(W, full_matrices=False)
            M = np.dot(V.T*(S**2),V)
            Minv = np.dot(V.T*(1./(S**2)),V)
            XW = np.dot(X, W)
            SW = np.dot(X.T, XW)  # d x n matrix
            W = np.dot(SW, np.dot(la.inv(np.dot(W.T, SW)),M))
            Z = np.dot(Minv, XW.T)

            err.append(la.norm(X.T - np.dot(W, Z)))
            if i > 0 and np.abs(err[i] - err[i-1]) < 1e-6:
                break

        self.components = W
        self.EMerr = np.array(err)

    def inferLatent(self,X):
        return np.dot(X,self.components)

    def reconstruct(self,X):
        Z = self.inferLatent(X)
        Xnew = np.dot(Z,self.components.T)
        return Xnew

    def changeNumComponents(self,n_components):
        if not 1 <= n_components <= self.n_features:
            raise ValueError("n_components=%r invalid for n_features=%d" % (n_components, self.n_features))
        self.n_components = n_components
        self.components = self.evecs[:,:n_components]
        return self


class ppca_model:
    def __init__(self, X, n_components=None, fitWith='SVD', max_iters=500):
        self.n_samples, self.n_features = X.shape

        # Center data
        self.mean = np.mean(X, axis=0)
        X -= self.mean

        if n_components is None:
            self.n_components = min(self.n_features,self.n_samples)
        elif not 0 <= n_components <= self.n_features:
            raise ValueError("n_components=%r invalid for n_features=%d" % (n_components, self.n_features))
        else:
            self.n_components = n_components

        self.evals = None
        self.evecs = None
        self.components = None
        self.s2 = None
        self.fitWith = fitWith
        self.LL = None

        if fitWith == 'SVD':
            self.fitSVD(X)
        elif fitWith == 'EM':
            self.fitEM(X, max_iters)

    def fitSVD(self, X):
        U, S, V = la.svd(X, full_matrices=False)
        self.evals = (S ** 2) / self.n_samples
        self.evecs = V.T
        A=self.evecs[:,:self.n_components]
        B=np.diag(np.sqrt((self.evals[:self.n_components] - self.s2 * np.ones([1, self.n_components])).flatten()))
        self.components = np.dot(A,B)
        self.s2 = np.mean(self.evals[self.n_components:])

    def fitEM(self,X,max_iters):
        W = np.random.randn(self.n_features,self.n_components)
        cX = np.dot(X.T,X)/self.n_samples
        s2 = np.mean(np.diag(cX))

        const = -self.n_features*0.5*np.log(2*np.pi)
        LL=[]
        for i in range(0,max_iters):
            U,S,V = la.svd(W, full_matrices=True)
            #Minv2=la.inv(np.dot(W.T,W)+s2*np.identity(self.n_components))
            Minv = np.dot(V.T*(1./(S**2+s2)),V)
            SW=np.dot(X.T,np.dot(X,W))/self.n_samples

            #LLtr=np.trace(np.dot(la.inv(np.dot(W,W.T)+s2*np.identity(self.n_features)),cX))
            #Cinv1=la.inv(np.dot(W,W.T)+s2*np.identity(self.n_features))
            s=1./(np.append(S**2,np.zeros([1,self.n_features-self.n_components]))+s2)
            Cinv = np.dot(U*s,U.T)
            LLtr=np.trace(np.dot(Cinv,cX))
            LLdetC=self.n_features*np.log(s2)-np.linalg.slogdet(s2*Minv)[1]
            LLi = self.n_samples*const - self.n_samples*0.5*LLdetC - self.n_samples*0.5*LLtr
            LL.append(LLi)
            if i>0 and np.abs(LL[i]-LL[i-1])<1e-6:
                break

        self.components = np.dot(SW,la.inv(s2*np.identity(self.n_components)+np.dot(Minv,np.dot(W.T,SW))))
        self.s2 = np.mean(np.std(X,axis=0)**2-np.diag(np.dot(SW,np.dot(Minv,W.T))))
        self.LL = np.array(LL)


    def setComponents(self, n_components):
        if self.fitWith == 'EM':
            raise TypeError("Cannot change number of components for PPCA model fit with EM")
        if not 1 <= n_components <= self.n_features:
            raise ValueError("n_components=%r invalid for n_features=%d"
                             % (n_components, self.n_features))
        self.s2 = np.mean(self.evals[n_components:])
        self.components = np.dot(self.components, np.diag(np.sqrt((self.evals - self.s2 * np.ones([1, n_components])).flatten())))

        return self


    def logLikelihood(self, n_components):
        self.LL = 0


# Standalone function for fitting PCA with EM
def pcaEM(X, n_components, max_iters):
        n_samples,n_features = X.shape
        W = np.random.randn(n_features, n_components)

        err = []
        for i in range(0, max_iters):
            # constrained algorithm
            Minv = la.lapack.flapack.dtrtri(np.tril(np.dot(W.T,W)).T)[0].T
            XW = np.dot(X,W)
            Z = np.dot(Minv, XW.T)
            ZZt = np.triu(np.dot(Z,Z.T))
            SW = np.dot(X.T, XW)
            W = np.dot(SW, np.dot(Minv.T, la.lapack.flapack.dtrtri(ZZt)[0]))

            err.append(la.norm(X.T - np.dot(W, Z)))
            if i > 0 and np.abs(err[i] - err[i-1]) < 1e-6:
                break

        return W


def ppca_minka(X,n_components):
    pca = pca_model(X)
    N,d = X.shape

    pca_scores = []
    for n in n_components:
        evals=pca.evals

        if n == d:
            s2 = 1
            pv = 0
        else:
            s2 = np.mean(evals[n:])
            pv = -N*(d-n)/2.*np.log(s2)

        pl = -N/2.*np.sum(np.log(evals[0:n]))

        m = d*n-n*(n+1.)/2.
        pp = np.log(2.*np.pi)*(m+n+1.)/2.

        evals_=evals.copy()
        evals_[n:]=s2
        pa=0.
        for i in range(n):
            for j in range(i+1, d):
                pa += np.log(1./evals_[j]-1./evals_[i])+np.log(evals[i]-evals[j])+np.log(N)
        pa=-pa/2.

        pu = -n*np.log(2.)
        for i in range(1,n+1):
            pu += special.gammaln((d-i+1)/2.)-np.log(np.pi)*(d-i+1)/2.

        pca_scores.append(pu+pl+pv+pp+pa-n/2.*np.log(N))

    return np.array(pca_scores)
