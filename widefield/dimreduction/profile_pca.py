import numpy as np
import scipy.linalg as la
from widefield.dimreduction.pca import pcaEM
import cProfile
import pstats
import pickle
from memory_profiler import memory_usage

q = 8200
dims = np.arange(10000,60000,10000)
n_iter = 3

times = np.zeros((n_iter, dims.size))
mem = np.zeros((n_iter, dims.size))

for i_iter in range(n_iter):
    for i_dim, N in enumerate(dims):
        print N
        x = np.random.randn(N,q).astype(np.float32)
        
        tmp = memory_usage((la.svd, [x,False]))
        #tmp = memory_usage((pcaEM, [x,min(N,q),500]))
        mem[i_iter, i_dim] = np.max(tmp)

        fname = "svdStats%d" % (i_dim)
        cProfile.run('la.svd(x,full_matrices=False)',fname)
        #fname = "emStats%d" % i_dim
        #cProfile.run('pcaEM(x,min(N,q),500)',fname)
        times[i_iter, i_dim] = pstats.Stats(fname).total_tt


stats={'N': dims, 'mem': mem, 'time': times}
pickle.dump(stats, open("svd_stats.pkl","wb"))
