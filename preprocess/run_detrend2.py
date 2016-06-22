from widefield.preprocess.movie_mask import *
import tables as tb
import timeit
from widefield.preprocess.detrend import detrend, detrend_nomask

## GLOABL VARS
datapath = '/suppscr/riekesheabrown/kpchamp/data/'

mouseId = "m177931"
collectionDate = "150731"

infile = datapath + mouseId + "/" + collectionDate + "/data.h5"

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

# create and save the mask
mask = get_mask(mov)
masky = mask.shape[0]
maskx = mask.shape[1]
mask_idx, pullmask, pushmask = mask_to_index(mask)
maskfile = datapath + mouseId + "/" + collectionDate + "/mask.h5"
f = tb.open_file(maskfile,'w')

# detrend the movie without masking
start_time = timeit.default_timer()
mov_detrend = detrend_nomask(mov, frames, exposure, window, dff)
detrend_time = timeit.default_timer() - start_time
print 'detrending took ' + str(detrend_time) + ' seconds\n'
del mov

# save the unmasked detrended movie
outfile1 = datapath + mouseId + "/" + collectionDate + "/data_detrend.h5"
f = tb.open_file(outfile1,'w')
f.create_array(f.root,'data',mov_detrend)
f.close()

# save the masked detrended movie
outfile2 = datapath + mouseId + "/" + collectionDate + "/data_detrend_mask.h5"
f = tb.open_file(outfile2,'w')
f.create_array(f.root,'data',cut_to_mask(mov_detrend,pushmask))
f.close()
