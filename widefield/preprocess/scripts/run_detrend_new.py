from widefield.preprocess.movie_mask import *
import tables as tb
import timeit
from widefield.preprocess.detrend import detrend
import sys
import numpy as np

## GLOABL VARS
basepath = '/suppscr/riekesheabrown/kpchamp/data/'

mouseId = sys.argv[1]
collectionDate = sys.argv[2]

infile = basepath + mouseId + "/" + collectionDate + "/data.npy"
outfile = basepath + mouseId + "/" + collectionDate + "/data_detrend_mask.h5"
maskfile = basepath + mouseId + "/" + collectionDate + "/mask.h5"

# load data
mov = np.load(infile)
# open_tb = tb.open_file(infile, 'r')
# mov = open_tb.root.data[:]

dff_type = 'ff0'
start = 0 # first frame in movie to detrend
stop = mov.shape[0] # last frame to detrend
window = 60 # window in seconds
exposure = 10 # camera exposure in ms

frames = range(start, stop)
print str(len(frames)) + ' frames will be detrended'

# setup masking variables
mask = get_mask(mov)
masky = mask.shape[0]
maskx = mask.shape[1]
mask_idx, pullmask, pushmask = mask_to_index(mask)

# detrend the movie
start_time = timeit.default_timer()
mov_detrend = detrend(mov, mask_idx, pushmask, frames, exposure, window, dff_type)
detrend_time = timeit.default_timer() - start_time
print 'detrending took ' + str(detrend_time) + ' seconds\n'

f=tb.open_file(outfile,'w')
f.create_array(f.root,'data',cut_to_mask(mov_detrend,pushmask).T)
f.close()

f=tb.open_file(maskfile,'w')
f.create_array(f.root,'mask_idx',mask_idx)
f.create_array(f.root,'pullmask',pullmask)
f.create_array(f.root,'pushmask',pushmask)
f.close()
