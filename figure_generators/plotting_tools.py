import matplotlib.pyplot as plt
import pickle
import numpy as np


def plot_from_pickle(fname):
    f = pickle.load(open(fname,'r'))
    exec f['code']
