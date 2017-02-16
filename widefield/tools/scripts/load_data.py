import numpy as np
import os, sys
import tables as tb
import pickle
import matplotlib.pyplot as plt
from widefield.dynamics.dmd import DynamicModeDecomposition
from widefield.preprocess.movie_mask import unmask, unmask_to_movie
from widefield.tools.alignment import *

basepath = "/suppscr/riekesheabrown/kpchamp/data/"
# mouse_id = "m187474"
# collection_date = "150804"
mouse_id = sys.argv[1]
collection_date = sys.argv[2]

# load actual recorded image data
if os.path.isfile(os.path.join(basepath, mouse_id, collection_date, "data_detrend_mask.h5")):
    f = tb.open_file(os.path.join(basepath, mouse_id, collection_date, "data_detrend_mask.h5"))
    X = f.root.data[:].T
    f.close()

    offset = np.min(X,axis=0)
    X_nonneg = X - offset

# load pushmask - for converting data to images
if os.path.isfile(os.path.join(basepath, mouse_id, collection_date, "mask.h5")):
    mask = tb.open_file(os.path.join(basepath, mouse_id, collection_date, "mask.h5"))
    pushmask = mask.root.pushmask[:]
    mask.close()

# load data of Doug's regions
if os.path.isfile(os.path.join(basepath, mouse_id, collection_date, "data_regions.pkl")):
    region_data = pickle.load(open(os.path.join(basepath, mouse_id, collection_date, "data_regions.pkl"),'r'))

# load regression input variables
if os.path.isfile(os.path.join(basepath, mouse_id, collection_date, "data_inputs.npy")):
    U = np.load(os.path.join(basepath, mouse_id, collection_date, "data_inputs.npy"))
