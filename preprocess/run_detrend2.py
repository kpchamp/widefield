from widefield.preprocess.movie_mask import *
import tables as tb
import timeit
from widefield.preprocess.detrend import detrend, detrend_nomask

## GLOABL VARS
datapath = '/gscratch/riekesheabrown/kpchamp/data/'

mouseId = "m177931"
collectionDate = "150731"

infile = datapath + mouseId + "/" + collectionDate + "/data.h5"
outfile = datapath + mouseId + "/" + collectionDate + "/data_detrend.h5"
maskfile = datapath + mouseId + "/" + collectionDate + "/mask.h5"

# load data
open_tb = tb.open_file(infile, 'r')
mov = open_tb.root.data[:]
open_tb.close()

dff = True
start = 0 # first frame in movie to detrend
stop = mov.shape[0] # last frame to detrend
window = 60 # window in seconds
exposure = 10 # camera exposure in ms

frames = range(start, stop)
print str(len(frames)) + ' frames will be detrended'

# mask = get_mask(mov)
# masky = mask.shape[0]
# maskx = mask.shape[1]
# mask_idx, pullmask, pushmask = mask_to_index(mask)

# mask = tb.open_file('/gscratch/riekesheabrown/kpchamp/data/mask2.h5','r')
# mask_idx = mask.root.mask_idx
# pullmask = mask.root.pullmask
# pushmask = mask.root.pushmask

# detrend the movie
start_time = timeit.default_timer()
#mov_detrend = detrend(mov, mask_idx, pushmask, frames, exposure, window, dff)
mov_detrend = detrend_nomask(mov, frames, exposure, window, dff)
detrend_time = timeit.default_timer() - start_time
print 'detrending took ' + str(detrend_time) + ' seconds\n'
del mov

# Kathleen mods start here
f=tb.open_file(outfile,'w')
#f.create_array(f.root,'data',cut_to_mask(mov_detrend,pushmask).T)
f.create_array(f.root,'data',mov_detrend)
f.close()

# mask.close()
