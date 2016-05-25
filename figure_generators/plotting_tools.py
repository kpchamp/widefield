import matplotlib.pyplot as plt
import pickle


def plot_from_pickle(fname):
    f = pickle.load(open(fname,'r'))
    exec f['code']
