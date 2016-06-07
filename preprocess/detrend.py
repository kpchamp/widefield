from movie_mask import *
import scipy.signal as sig


def detrend(mov, mask_idx, pushmask, frames, exposure, window, dff):
    # define size of rolling average
    expose = np.float(exposure/1000.) #convert exposure from ms to s
    win = np.float(window) # window in seconds
    win = win/expose
    win = np.ceil(win)
    if win > len(frames)/2:
        print 'please choose a window smaller than half the length of time you are analyzing'
    # pre-allocate matrices
    kernal = sig.gaussian(win, win/8)
    padsize = len(frames) + (win*2) - 1
    mov_pad = np.zeros([padsize], dtype=('float32'))
    yidx = np.array(mask_idx[0])
    xidx = np.array(mask_idx[1])
    mov_detrend = np.zeros([len(frames), mov.shape[1], mov.shape[2]], dtype=('float32'))
    # baseline subtract by gaussian convolution along time (dim=0)
    #print 'per-pixel baseline subtraction using gaussian convolution ...\n'
    len_iter = pushmask.shape[0]
    #print str(len_iter) + ' pixels'
    for n in range(len_iter):
        print n
        # put data in padded frame
        mov_pad = pad_vector(mov[frames,yidx[n],xidx[n]], win)
        # moving average by convolution
        mov_ave = sig.fftconvolve(mov_pad, kernal/kernal.sum(), mode='valid')
        # cut off pad
        mov_ave = mov_ave[win/2:(-win/2)-1]
        mov_ave = mov_ave.astype('float32')
        # and now use moving average as f0 for df/f
        if dff:
            mov_detrend[:,yidx[n],xidx[n]] = (mov[frames,yidx[n],xidx[n]] - mov_ave)/mov_ave
            if mov_ave == 0:
                print >>open('output.txt','a'), n
        else:
            mov_detrend[:,yidx[n],xidx[n]] = (mov[frames,yidx[n],xidx[n]] - mov_ave)
    return mov_detrend

def pad_vector(dat, win):
    tlen = dat.shape[0]
    pad_start = dat[0:win]+(dat[0]-dat[win])
    pad_end = dat[tlen-win:]+(dat[-1]-dat[tlen-win])
    dat_pad = np.append(np.append(pad_start, dat), pad_end)
    return dat_pad

