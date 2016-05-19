import matplotlib.pyplot as plt
import pickle
import numpy as np
from widefield.preprocess.movie_mask import unmask
from widefield.dimreduction.analyze_components import *


basepath = '/gscratch/riekesheabrown/kpchamp/data/'
datapath = basepath + 'm187201_150727_decitranspose_detrend.h5'


def plot_components(W, components_to_plot, pushmask, n_rows, n_cols, fname=None, clim=None):
    n_plots = len(components_to_plot)
    for i in range(n_plots):
        plt.subplot(np.floor(np.sqrt(n_plots)),np.ceil(n_plots/np.floor(np.sqrt(n_plots))),i+1)
        img = np.reshape(unmask(W[:,components_to_plot[i]], pushmask, n_rows*n_cols), [n_rows, n_cols])
        plt.imshow(img)
        if clim is not None:
            plt.clim(clim)
        plt.title('component %d' % components_to_plot[i])
        plt.axis('off')
    if fname is not None:
        pickle.dump(plt, open(fname, 'w'))


def plot_component_comparison(dfrow1, dfrow2, component_limit=500, fname=None):
    fname1 = basepath + 'components_twin%d_nsamples%d_tstart%d.pkl' % (dfrow1['windowLength'],dfrow1['sampleSize'],dfrow1['startTime'])
    fname2 = basepath + 'components_twin%d_nsamples%d_tstart%d.pkl' % (dfrow2['windowLength'],dfrow2['sampleSize'],dfrow2['startTime'])
    A = pickle.load(open(fname1,'r'))
    B = pickle.load(open(fname2,'r'))
    C = compare_components(A,B)
    plt.imshow(C[0:component_limit,0:component_limit])
    plt.ylabel('t_win=%d, n_samples=%d, t_start=%d' % (dfrow1['windowLength'],dfrow1['sampleSize'],dfrow1['startTime']))
    plt.xlabel('t_win=%d, n_samples=%d, t_start=%d' % (dfrow2['windowLength'],dfrow2['sampleSize'],dfrow2['startTime']))
    if fname is not None:
        pickle.dump(plt, open(fname, 'w'))


def dim_vs_samples(df):
    for t_win in set(df['windowLength']):
        startTimes = sorted(set(df['startTime'][df['windowLength'] == t_win]))
        fig = plt.figure()
        for i, t_start in enumerate(startTimes):
            sampleSizes = sorted(set(df['sampleSize'][(df['windowLength'] == t_win) & (df['startTime'] == t_start)]))
            p_threshold = []
            p_aic = []
            p_bic = []
            p_xval = []
            p_90percent = []
            for n_samples in sampleSizes:
                data = df['data'][(df['windowLength'] == t_win) & (df['startTime'] == t_start) & (df['sampleSize'] == n_samples)].item()
                p_threshold.append(data['p_threshold'])
                p_aic.append(np.argmin(data['aic'])+1)
                p_bic.append(np.argmin(data['bic'])+1)
                p_xval.append(data['ps'][np.argmax(data['lltest'])])
                sv_totals=np.array([np.sum(data['svs'][0:k+1]) for k in range(len(data['svs']))])
                p_90percent.append(np.argmax(sv_totals>(0.9*sv_totals[-1]))+1)
            if len(startTimes)>8:
                plt.subplot(4,len(startTimes)/4,i+1)
            elif len(startTimes)>2:
                plt.subplot(2,len(startTimes)/2,i+1)
            else:
                plt.subplot(1, len(startTimes), i+1)
            plt.plot(sampleSizes, p_threshold, 'o-', label='threshold')
            plt.plot(sampleSizes, p_bic, 'o-', label='BIC')
            plt.plot(sampleSizes, p_aic, 'o-', label='AIC')
            plt.plot(sampleSizes, p_xval, 'o-', label='xval')
            plt.plot(sampleSizes, p_90percent, 'o-', label='90%')
            if i == 0 and len(startTimes) < 8:
                plt.legend(loc=2)
            plt.xlabel('number of samples')
            plt.ylabel('p')
            plt.title('T_win=%d, T_start=%d' % (t_win, t_start))
        if len(startTimes) < 8:
            plt.tight_layout()
        pickle.dump(fig, open('plot_Twin%d.pkl' % t_win, 'w'))


def plot_dims(data, n_samples, legendLoc=2):
    plt.plot(data['n_samples'], data['p_threshold'], 'o-', label='threshold')
    plt.plot(data['n_samples'], data['p_bic'], 'o-', label='BIC')
    plt.plot(data['n_samples'], data['p_aic'], 'o-', label='AIC')
    plt.plot(data['n_samples'], data['p_xval'], 'o-', label='xval')
    plt.legend(loc=legendLoc)
    plt.xlabel('number of samples')
    plt.ylabel('p')
    plt.title('Samples vs p - Twin=%d' % n_samples)


def generate_bic_plot():
    r1_1 = pickle.load(open('../p_twin347904_nsamples43488_1.pkl', 'r'))
    r1_2 = pickle.load(open('../p_twin347904_nsamples130464_1.pkl', 'r'))
    r1_3 = pickle.load(open('../p_twin347904_nsamples217440_1.pkl', 'r'))

    ps = np.arange(8252)+1.
    m = 8252.*ps+1.-0.5*ps*(ps-1.)

    plt.subplot(1,3,1)
    plt.plot(-2*r1_1['lltrain'])
    plt.plot(-2*r1_2['lltrain'])
    plt.plot(-2*r1_3['lltrain'])
    plt.title('-2*(neg log likelihood)')
    plt.subplot(1,3,2)
    plt.plot(m*np.log(43488.*3./4.))
    plt.plot(m*np.log(130464*3./4.))
    plt.plot(m*np.log(217440*3./4.))
    plt.title('BIC penalty')
    plt.subplot(1,3,3)
    plt.plot(-2*r1_1['lltrain']+m*np.log(43488.*3./4.))
    plt.plot(-2*r1_2['lltrain']+m*np.log(130464.*3./4.))
    plt.plot(-2*r1_3['lltrain']+m*np.log(217440.*3./4.))
    plt.title('BIC')
    plt.show()