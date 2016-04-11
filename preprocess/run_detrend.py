from movie_mask import *
import tables as tb
import timeit
from detrend import detrend

## GLOABL VARS
dff = True
start = 0 # first frame in movie to detrend
stop = 3000 # last frame to detrend
window = 10 # window in seconds
exposure = 10 # camera exposure in ms
infile = '/Users/kpchamp/Dropbox (uwamath)/backup/research/python/data/20150727_subset.h5'
outfile = '/Users/kpchamp/Dropbox (uwamath)/backup/research/python/data/20150727_detrend.h5'

# load data
open_tb = tb.open_file(infile, 'r')
mov = open_tb.root.data

frames = range(start, stop)
print str(len(frames)) + ' frames will be detrended'

# setup masking variables
mask = get_mask(mov)
masky = mask.shape[0]
maskx = mask.shape[1]
mask_idx, pullmask, pushmask = mask_to_index(mask)

# detrend the movie
start_time = timeit.default_timer()
mov_detrend = detrend(mov, mask_idx, pushmask, frames, exposure, window, dff)
detrend_time = timeit.default_timer() - start_time
print 'detrending took ' + str(detrend_time) + ' seconds\n'

# Kathleen mods start here
f=tb.open_file(outfile,'w')
f.create_array(f.root,'data',cut_to_mask(mov_detrend,pushmask))
f.create_group(f.root,'mask')
f.create_array(f.root.mask,'mask_idx',mask_idx)
f.create_array(f.root.mask,'pullmask',pullmask)
f.create_array(f.root.mask,'pushmask',pushmask)
f.close()
open_tb.close()
