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


def generate_bic_plot():
    r1_1=pickle.load(open('../p_twin347904_nsamples43488_1.pkl','r'))
    r1_2=pickle.load(open('../p_twin347904_nsamples130464_1.pkl','r'))
    r1_3=pickle.load(open('../p_twin347904_nsamples217440_1.pkl','r'))

    ps=np.arange(8252)+1.
    m=8252.*ps+1.-0.5*ps*(ps-1.)

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