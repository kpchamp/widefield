import matplotlib.pyplot as plt
import pickle
import numpy as np


def plot_dims(data,n_samples,legendLoc=2):
    plt.plot(data['n_samples'],data['p_threshold'],'o-',label='threshold')
    plt.plot(data['n_samples'],data['p_bic'],'o-',label='BIC')
    plt.plot(data['n_samples'],data['p_aic'],'o-',label='AIC')
    plt.plot(data['n_samples'],data['p_xval'],'o-',label='xval')
    plt.legend(loc=legendLoc)
    plt.xlabel('number of samples')
    plt.ylabel('p')
    plt.title('Samples vs p - Twin=%d'%n_samples)
