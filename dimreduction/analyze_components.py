import numpy as np
import math
import scipy.linalg as la
import pickle
from widefield.dimreduction.pca import ppca_model


basepath = '/gscratch/riekesheabrown/kpchamp/data/m187201/150727/'
datapath = basepath + 'decitranspose_detrend.h5'


def subspace_angle(A,B):
    A = A/np.sqrt(np.sum(np.abs(A)**2,axis=0))
    B = B/np.sqrt(np.sum(np.abs(B)**2,axis=0))
    u,s,v = la.svd(np.dot(A.T,B),full_matrices=False)
    return math.acos(min(1,max(s[-1],-1)))


def compare_components(A,B):
    A = A/np.sqrt(np.sum(np.abs(A)**2,axis=0))
    B = B/np.sqrt(np.sum(np.abs(B)**2,axis=0))
    return np.abs(np.dot(A.T,B))


def get_component_comparison(dfrow1,dfrow2):
    fname1 = basepath + 'components/components_twin%d_nsamples%d_tstart%d.pkl' % (dfrow1['windowLength'],dfrow1['sampleSize'],dfrow1['startTime'])
    fname2 = basepath + 'components/components_twin%d_nsamples%d_tstart%d.pkl' % (dfrow2['windowLength'],dfrow2['sampleSize'],dfrow2['startTime'])
    A = pickle.load(open(fname1,'r'))
    B = pickle.load(open(fname2,'r'))
    return compare_components(A,B)


def get_subspace_angles(dfrow1,dfrow2,cutoff):
    fname1 = basepath + 'components/components_twin%d_nsamples%d_tstart%d.pkl' % (dfrow1['windowLength'],dfrow1['sampleSize'],dfrow1['startTime'])
    fname2 = basepath + 'components/components_twin%d_nsamples%d_tstart%d.pkl' % (dfrow2['windowLength'],dfrow2['sampleSize'],dfrow2['startTime'])
    A = pickle.load(open(fname1,'r'))
    B = pickle.load(open(fname2,'r'))
    angles = np.zeros((cutoff,))
    for i in range(cutoff):
        angles[i] = subspace_angle(A[:,0:i+1],B[:,0:i+1])
    return angles


def get_cutoff(data,type):
    if type == 'threshold':
        return data['p_threshold']
    elif type == 'aic':
        return np.argmin(data['aic'])+1
    elif type == 'bic':
        return np.argmin(data['bic'])+1
    elif type == 'xval':
        return data['ps'][np.argmax(data['lltest'])]
    elif type == '90percent':
        sv_totals=np.array([np.sum(data['svs'][0:k+1]) for k in range(len(data['svs']))])
        return np.argmax(sv_totals>(0.9*sv_totals[-1]))+1
    else:
        raise ValueError('must specify a type')


def get_residuals(X, n_components):
    ppca = ppca_model(X, n_components=n_components)
    Xnew = ppca.reconstruct(X)
    residual = X - Xnew
    return residual
