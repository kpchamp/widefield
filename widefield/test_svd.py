import numpy as np
import tables as tb
import scipy.linalg as la
import timeit
import cProfile, pstats, sys
import pickle

dims = np.arange(200000,350000,50000)

results = np.zeros((dims.size,))

fpath='/gscratch/riekesheabrown/kpchamp/data/m187201_150727_decitranspose_detrend.h5'
fout=open('/gscratch/riekesheabrown/kpchamp/output','w')
open_tb=tb.open_file(fpath,'r')

for i_dim, numFrames in enumerate(dims):
    fresult='/gscratch/riekesheabrown/kpchamp/data/svd_%06dframe.h5' % numFrames

    mov=open_tb.root.data[:,0:numFrames]
    print >> fout, 'movie shape is ', mov.shape
    fout.flush()

    #start_time = timeit.default_timer()
    fname = "svdStats"
    pr = cProfile.Profile()
    pr.enable()
    U,s,V=la.svd(mov.T, full_matrices=False)
    pr.disable()
    results[i_dim] = pstats.Stats(pr, stream=sys.stdout).total_tt
    #svd_time = timeit.default_timer() - start_time
    print >> fout, 'SVD time: ', results[i_dim]
    fout.flush()

    tbout=tb.open_file(fresult,'w')
    tbout.create_array(tbout.root,'V',V.T)
    tbout.create_array(tbout.root,'s',s)
    tbout.create_array(tbout.root,'U',U)
    tbout.close()


open_tb.close()
fout.close()
stats={'N': dims, 'time': results}
pickle.dump(stats, open("svd_stats.pkl","w"))
