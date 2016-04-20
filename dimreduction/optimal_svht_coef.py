import scipy.integrate
import numpy as np
from __future__ import division

# Code modified from MATLAB. beta must be a number, not a vector.


def optimal_svht_coef(beta,sigma_known):
    if sigma_known:
        return optimal_svht_coef_sigma_known(beta)
    else:
        return optimal_svht_coef_sigma_unknown(beta)


def optimal_svht_coef_sigma_known(beta):
    w = (8.*beta)/(beta+1.+np.sqrt(beta**2+14.*beta+1.))
    return np.sqrt(2.*(beta+1.)+w)


def optimal_svht_coef_sigma_unknown(beta):
    coef = optimal_svht_coef_sigma_known(beta)

    MPmedian = medianMarcenkoPastur(beta)
    omega = coef / np.sqrt(MPmedian)
    return omega


def medianMarcenkoPastur(beta):
    MarPas = lambda x: 1-incMarPas(x,beta,0)
    lobnd = (1.-np.sqrt(beta))**2
    hibnd = (1.+np.sqrt(beta))**2
    change = 1

    while change and (hibnd - lobnd > 0.001):
        change = 0
        x = np.linspace(lobnd,hibnd,5)
        y = np.zeros(x.shape)
        for i in range(x.size):
            y[i] = MarPas(x[i])
        if np.any(y < 0.5):
            lobnd = np.max(x[y<0.5])
            change = 1
        if np.any(y > 0.5):
            hibnd = np.min(x[y>0.5])
            change = 1

    return (hibnd+lobnd)/2.


def incMarPas(x0,beta,gamma):
    if beta > 1:
        raise ValueError("beta=%f must be greater than 1",beta)
    topSpec = (1.+np.sqrt(beta))**2
    botSpec = (1.-np.sqrt(beta))**2
    MarPas = lambda x: IfElse((topSpec-x)*(x-botSpec) > 0,
                              np.sqrt((topSpec-x)*(x-botSpec))/(beta*x)/(2.*np.pi))

    if gamma != 0:
        f = lambda x: (x**gamma * MarPas(x))
    else:
        f = lambda x: MarPas(x)

    I = scipy.integrate.quad(f,x0,topSpec)[0]
    return I


def IfElse(Q,point):
    y = point
    notQ = np.where(np.logical_not(Q))[0]
    if notQ.size != 0:
        y = 0.
    return y
