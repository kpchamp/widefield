import numpy as np
import os
import tables as tb
import pickle
import matplotlib.pyplot as plt
from widefield import *

basepath = "/suppscr/riekesheabrown/kpchamp/data/"
mouse_id = "m187474"
collection_date = "150804"

# load actual recorded image data
f = tb.open_file(os.path.join(basepath, mouse_id, collection_date, "data_detrend_mask.h5"))
X = f.root.data[:]
f.close()

# load pushmask - for converting data to images
mask = tb.open_file(os.path.join(basepath, mouse_id, collection_date, "mask.h5"))
pushmask = mask.root.pushmask
mask.close()

# load data of Doug's regions
region_data = pickle.load(open(os.path.join(basepath, mouse_id, collection_date, "data_regions.pkl"),'r'))

# load regression input variables
U = np.load(os.path.join(basepath, mouse_id, collection_date, "data_inputs.npy"))
