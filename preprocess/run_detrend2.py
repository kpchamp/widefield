from movie_mask import *
import tables as tb
import timeit
from detrend import detrend

## GLOABL VARS
datapath = '/gscratch/riekesheabrown/kpchamp/data/'

mouseId = "m177931"
collectionDate = "150731"

infile = datapath + mouseId + "/" + collectionDate + "/data.h5"
outfile = datapath + mouseId + "/" + collectionDate + "/transpose_detrend.h5"
maskfile = datapath + mouseId + "/" + collectionDate + "/mask.h5"

# load data
open_tb = tb.open_file(infile, 'r')
mov = open_tb.root.data[:,:]

dff = True
start = 0 # first frame in movie to detrend
stop = mov.shape[0] # last frame to detrend
window = 60 # window in seconds
exposure = 10 # camera exposure in ms

frames = range(start, stop)
print str(len(frames)) + ' frames will be detrended'

mask = tb.open_file('/gscratch/riekesheabrown/kpchamp/data/mask.h5','r')
mask_idx = (mask.root.mask_idx[0], mask.root.mask_idx[1])
pullmask = mask.root.pullmask[:]
pushmask = mask.root.pushmask[:]

# detrend the movie
start_time = timeit.default_timer()
mov_detrend = detrend(mov, mask_idx, pushmask, frames, exposure, window, dff)
detrend_time = timeit.default_timer() - start_time
print 'detrending took ' + str(detrend_time) + ' seconds\n'

# Kathleen mods start here
f=tb.open_file(outfile,'w')
f.create_array(f.root,'data',cut_to_mask(mov_detrend,pushmask).T)
f.close()

open_tb.close()