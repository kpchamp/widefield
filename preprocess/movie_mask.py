import numpy as np


def generate_mask(img, percentage):
    img_dims = img.shape
    bitdepth = 2**16
    img_hist, img_bins = np.histogram(img, bitdepth/100, [0,bitdepth])
    background_proportion = (img_dims[0] * img_dims[1])/(100/percentage)
    cum = np.cumsum(img_hist)
    idx = cum[cum<background_proportion].shape[0]
    thresh = np.floor(img_bins[idx]).astype('uint16')
    mask = np.zeros(img_dims[0]*img_dims[1])
    img_flat = img.reshape(img_dims[0]*img_dims[1])
    mask[img_flat>thresh] = 1
    mask = mask.reshape(img_dims)
    return mask


def get_mask(mov):
    frame = mov[0,:,:] # mask using the first frame of the movie
    mask = generate_mask(frame, 50)
    #plt.imshow(frame * mask)
    return mask


def mask_to_index(mask):
    mask = mask.astype('uint16') #64 bit integer
    mask_idx = np.ndarray.nonzero(mask)
    pullmask = mask.reshape(mask.shape[0]*mask.shape[1]) #flatten image to vector
    pullmask = np.squeeze(np.array([pullmask==1])) #convert to boolean
    pushmask = np.ndarray.nonzero(pullmask) # indexes for within the mask
    pushmask = pushmask[0]
    return mask_idx, pullmask, pushmask


def cut_to_mask(mov,pushmask):
    frames,ny,nx = mov.shape
    npxls = ny*nx
    return mov.reshape([frames,npxls])[:,pushmask]


def unmask(mov_detrend,pushmask,npxls):
    if mov_detrend.ndim == 1:
        mov_full = np.zeros((npxls,1))
        mov_full[pushmask,0] = mov_detrend
    else:
        frames = mov_detrend.shape[1]
        mov_full = np.zeros((npxls,frames))
        mov_full[pushmask,:] = mov_detrend
    return mov_full
