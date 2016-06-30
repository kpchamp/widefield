import numpy as np
import scipy.linalg as la
from sklearn.decomposition import FactorAnalysis
import warnings
from memory_profiler import profile


class fa_model:
    def __init__(self, Xin, n_components=None, fit_type='sklearn'):
        self.n_samples, self.n_features = Xin.shape

        # Center data
        self.mean = np.mean(Xin, axis=0)
        X = Xin - self.mean

        if n_components is None:
            self.n_components = min(self.n_features,self.n_samples)
        elif not 0 <= n_components <= self.n_features:
            raise ValueError("n_components=%r invalid for n_features=%d" % (n_components, self.n_features))
        else:
            self.n_components = n_components

        self.components = None
        self.variance = None
        self.ll = None
        self.n_iters = None

        fa = FactorAnalysis(n_components=n_components)
        if fit_type=='sklearn':
            fa.fit(X)
            self.components = fa.components_
            self.variance = fa.noise_variance_
            self.ll = fa.loglike_
            self.n_iters = fa.n_iter_
        else:
            self.my_fit(X)

    @profile
    def my_fit(self, X):
        n_samples, n_features = X.shape
        n_components = self.n_components
        if n_components is None:
            n_components = n_features

        # some constant terms
        nsqrt = np.sqrt(n_samples)
        llconst = n_features * np.log(2. * np.pi) + n_components
        var = np.var(X, axis=0)

        if self.noise_variance_init is None:
            psi = np.ones(n_features, dtype=X.dtype)
        else:
            if len(self.noise_variance_init) != n_features:
                raise ValueError("noise_variance_init dimension does not "
                                 "with number of features : %d != %d" %
                                 (len(self.noise_variance_init), n_features))
            psi = np.array(self.noise_variance_init)

        loglike = []
        old_ll = -np.inf
        SMALL = 1e-12

        # we'll modify svd outputs to return unexplained variance
        # to allow for unified computation of loglikelihood
        if self.svd_method == 'lapack':
            def my_svd(X):
                _, s, V = la.svd(X, full_matrices=False)
                return (s[:n_components], V[:n_components],
                        np.dot(s[n_components:],s[n_components:]))
        # elif self.svd_method == 'randomized':
        #     random_state = check_random_state(self.random_state)
        #
        #     def my_svd(X):
        #         _, s, V = randomized_svd(X, n_components,
        #                                  random_state=random_state,
        #                                  n_iter=self.iterated_power)
        #         return s, V, squared_norm(X) - squared_norm(s)
        else:
            raise ValueError('SVD method %s is not supported. Please consider'
                             ' the documentation' % self.svd_method)

        for i in xrange(self.max_iter):
            # SMALL helps numerics
            sqrt_psi = np.sqrt(psi) + SMALL
            s, V, unexp_var = my_svd(X / (sqrt_psi * nsqrt))
            s **= 2
            # Use 'maximum' here to avoid sqrt problems.
            W = np.sqrt(np.maximum(s - 1., 0.))[:, np.newaxis] * V
            del V
            W *= sqrt_psi

            # loglikelihood
            ll = llconst + np.sum(np.log(s))
            ll += unexp_var + np.sum(np.log(psi))
            ll *= -n_samples / 2.
            loglike.append(ll)
            if (ll - old_ll) < self.tol:
                break
            old_ll = ll

            psi = np.maximum(var - np.sum(W ** 2, axis=0), SMALL)
        else:
            warnings.warn('FactorAnalysis did not converge.' +
                          ' You might want' +
                          ' to increase the number of iterations.')

        self.components = W
        self.variance = psi
        self.ll = loglike
        self.n_iters = i + 1
        return self
