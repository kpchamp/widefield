import matplotlib.pyplot as plt
from widefield.preprocess.movie_mask import unmask
from widefield.dimreduction.analyze_components import *
from matplotlib.mlab import normpdf


def plot_components(W, components_to_plot, pushmask, n_rows, n_cols, clim=None):
    n_plots = len(components_to_plot)
    for i in range(n_plots):
        plt.subplot(np.floor(np.sqrt(n_plots)),np.ceil(n_plots/np.floor(np.sqrt(n_plots))),i+1)
        img = np.reshape(unmask(W[:,components_to_plot[i]], pushmask, n_rows*n_cols), [n_rows, n_cols])
        plt.imshow(img)
        if clim is not None:
            plt.clim(clim)
        plt.title('component %d' % components_to_plot[i])
        plt.axis('off')


def plot_component_comparison(dfrow1, dfrow2, component_limit=500):
    C = get_component_comparison(dfrow1,dfrow2)
    plt.imshow(C[0:component_limit,0:component_limit],interpolation='nearest')
    plt.ylabel('t_win=%d, n_samples=%d, t_start=%d' % (dfrow1['windowLength'],dfrow1['sampleSize'],dfrow1['startTime']))
    plt.xlabel('t_win=%d, n_samples=%d, t_start=%d' % (dfrow2['windowLength'],dfrow2['sampleSize'],dfrow2['startTime']))


def plot_residual(residuals):
    if residuals.ndim == 1:
        residuals = np.reshape(residuals,(residuals.shape[0],1))
    n_plots = residuals.shape[1]
    for i in range(n_plots):
        mu = np.mean(residuals[:,i])
        sigma = np.std(residuals[:,i])
        x = np.linspace(np.min(residuals[:,i]), np.max(residuals[:,i]), 1000)
        plt.subplot(np.floor(np.sqrt(n_plots)),np.ceil(n_plots/np.floor(np.sqrt(n_plots))),i+1)
        plt.plot(x, normpdf(x, mu, sigma))
        plt.hist(residuals[:,i], bins=100, normed=True)


def dim_vs_samples(df):
    for t_win in set(df['windowLength']):
        f = {}
        startTimes = sorted(set(df['startTime'][df['windowLength'] == t_win]))
        f['code'] = 'fig = plt.figure()\n'
        for i, t_start in enumerate(startTimes):
            f['sampleSizes'] = sorted(set(df['sampleSize'][(df['windowLength'] == t_win) & (df['startTime'] == t_start)]))
            f['p_threshold'] = []
            f['p_aic'] = []
            f['p_bic'] = []
            f['p_xval'] = []
            f['p_90percent'] = []
            for n_samples in f['sampleSizes']:
                data = df['data'][(df['windowLength'] == t_win) & (df['startTime'] == t_start) & (df['sampleSize'] == n_samples)].item()
                f['p_threshold'].append(data['p_threshold'])
                f['p_aic'].append(np.argmin(data['aic'])+1)
                f['p_bic'].append(np.argmin(data['bic'])+1)
                f['p_xval'].append(data['ps'][np.argmax(data['lltest'])])
                sv_totals=np.array([np.sum(data['svs'][0:k+1]) for k in range(len(data['svs']))])
                f['p_90percent'].append(np.argmax(sv_totals>(0.9*sv_totals[-1]))+1)
            if len(startTimes)>8:
                f['code'] += 'plt.subplot(4,%d,%d)\n' % (np.int(len(startTimes)/4),i+1)
            elif len(startTimes)>2:
                f['code'] += 'plt.subplot(2,%d,%d)\n' % (np.int(len(startTimes)/2),i+1)
            else:
                f['code'] += 'plt.subplot(1, %d, %d)\n' % (len(startTimes),i+1)
            f['code'] += "plt.plot(f['sampleSizes'], f['p_threshold'], 'o-', label='threshold')\n"
            f['code'] += "plt.plot(f['sampleSizes'], f['p_bic'], 'o-', label='BIC')\n"
            f['code'] += "plt.plot(f['sampleSizes'], f['p_aic'], 'o-', label='AIC')\n"
            f['code'] += "plt.plot(f['sampleSizes'], f['p_xval'], 'o-', label='xval')\n"
            f['code'] += "plt.plot(f['sampleSizes'], f['p_90percent'], 'o-', label='90%')\n"
            if i == 0 and len(startTimes) < 8:
                f['code'] += "plt.legend(loc=2)\n"
            f['code'] += "plt.xlabel('number of samples')\n"
            f['code'] += "plt.ylabel('p')\n"
            f['code'] += "plt.title('T_start=%d')\n" % t_start
        f['code'] += "plt.suptitle('T_win=%d')\n" % t_win
        pickle.dump(f, open('plot_Twin%d.pkl' % t_win, 'w'))


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