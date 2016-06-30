import numpy as np


def broadcast_left_matrix_multiply(A,X):
    return np.transpose(np.dot(np.transpose(X,(0,2,1)),A.T),(0,2,1))