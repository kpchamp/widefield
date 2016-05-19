import numpy as np
import math
import scipy.linalg as la
import tables as tb
import pandas as pd
from manage_data import get_cutoff


def subspace_angle(A,B):
    A = A/np.sqrt(np.sum(np.abs(A)**2,axis=0))
    B = B/np.sqrt(np.sum(np.abs(B)**2,axis=0))
    u,s,v = la.svd(np.dot(A.T,B),full_matrices=False)
    return math.acos(s[-1])


def compare_components(A,B):
    A = A/np.sqrt(np.sum(np.abs(A)**2,axis=0))
    B = B/np.sqrt(np.sum(np.abs(B)**2,axis=0))
    return np.dot(A.T,B)


# f = tb.open_file("/gscratch/riekesheabrown/kpchamp/data/m187201_150727_decitranspose_detrend.h5",'r')
# df = pd.read_pickle("/gscratch/riekesheabrown/kpchamp/data/allData_df.pkl")
# X=f.root.data[:,:].T
#
# t_win = 86976
# df_subset = df[df['windowLength'] == t_win]
