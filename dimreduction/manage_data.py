import glob, re, pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def process_files(str):
    allData = DataCollection()
    files=glob.glob(str)
    for i,fname in enumerate(files):
        t_win, n_samples, startidx = map(int,re.findall('[0-9]+',fname))
        t_start = (startidx-1)*t_win
        data = pickle.load(open(fname,'r'))
        allData.add_data(Dataset(t_win, t_start, n_samples, data))
    pickle.dump(allData,open('allData','w'))


class Dataset:
    def __init__(self, t_win, t_start, n_samples, data):
        self.windowLength = t_win
        self.startTime = t_start
        self.sampleSize = n_samples
        self.data = data


class DataCollection:
    def __init__(self):
        self.windowLengths = []
        self.startTimes = []
        self.sampleSizes = []
        self.data = []

    def add_data(self, dataset):
        self.windowLengths.append(dataset.windowLength)
        self.startTimes.append(dataset.startTime)
        self.sampleSizes.append(dataset.sampleSize)
        self.data.append(dataset)

    def get_data(self, t_win=None, t_start=None, n_samples=None):
        idxs = set(range(len(self.data)))
        if t_win is not None:
            tmp = set([i for i, x in enumerate(self.windowLengths) if x==t_win])
            idxs.intersection_update(tmp)
        if t_start is not None:
            tmp = set([i for i, x in enumerate(self.startTimes) if x==t_start])
            idxs.intersection_update(tmp)
        if n_samples is not None:
            tmp = set([i for i, x in enumerate(self.sampleSizes) if x==n_samples])
            idxs.intersection_update(tmp)
        return idxs


def convert_collection_to_pandas(allData):
    dictList = []
    for data in allData.data:
        dictList.append(data.data)
    dict = {'windowLength': allData.windowLengths, 'startTime': allData.startTimes,
            'sampleSize': allData.sampleSizes, 'data': dictList}
    df = pd.DataFrame(dict)
    return df


def plot_data_collection(allData):
    for t_win in set(allData.windowLengths):
        idxs=allData.get_data(t_win=t_win)
        startTimes = sorted(set([allData.startTimes[i] for i in idxs]))
        fig = plt.figure()
        for i, t_start in enumerate(startTimes):
            idxs2 = allData.get_data(t_win=t_win, t_start=t_start)
            sampleSizes = np.array(sorted(set([allData.sampleSizes[j] for j in idxs2])))
            idxs2_ord = [x for (y,x) in sorted(zip([allData.sampleSizes[j] for j in idxs2],idxs2))]
            p_threshold=[]
            p_aic=[]
            p_bic=[]
            p_xval=[]
            p_90percent=[]
            for j in idxs2_ord:
                p_threshold.append(allData.data[j].data['p_threshold'])
                p_aic.append(np.argmin(allData.data[j].data['aic'])+1)
                p_bic.append(np.argmin(allData.data[j].data['bic'])+1)
                p_xval.append(allData.data[j].data['ps'][np.argmax(allData.data[j].data['lltest'])])
                sv_totals=np.array([np.sum(allData.data[j].data['svs'][0:k+1]) for k in range(len(allData.data[j].data['svs']))])
                p_90percent.append(np.argmax(sv_totals>(0.9*sv_totals[-1]))+1)
            if len(startTimes)>8:
                plt.subplot(4,len(startTimes)/4,i+1)
            elif len(startTimes)>2:
                plt.subplot(2,len(startTimes)/2,i+1)
            else:
                plt.subplot(1,len(startTimes),i+1)
            plt.plot(sampleSizes,p_threshold,'o-',label='threshold')
            plt.plot(sampleSizes,p_bic,'o-',label='BIC')
            plt.plot(sampleSizes,p_aic,'o-',label='AIC')
            plt.plot(sampleSizes,p_xval,'o-',label='xval')
            plt.plot(sampleSizes,p_90percent,'o-',label='90%')
            if i==0 and len(startTimes)<8:
                plt.legend(loc=2)
            plt.xlabel('number of samples')
            plt.ylabel('p')
            plt.title('T_win=%d, T_start=%d'%(t_win,t_start))
        if len(startTimes)<8:
            plt.tight_layout()
        pickle.dump(fig,open('plot_Twin%d.pkl'%t_win,'w'))



# class data_Twin:
#     def __init__(self, Twin):
#         self.values = []
#         self.objects = []
#         self.Twin = Twin
#
#     def add_Tstart(self, Tstart, obj):
#         if Tstart in self.values:
#             raise ValueError("already have entry for Tstart=%d", Tstart)
#         else:
#             i = bisect.bisect(self.values, Tstart)
#             self.values.insert(i, Tstart)
#             self.objects.insert(i, obj)
#
#     def get_Tstart(self, Tstart):
#         if Tstart not in self.values:
#             raise ValueError("no entry for Tstart=%d", Tstart)
#
#         return self.objects[self.values.index(Tstart)]
#
#     def get_samples(self, Tstart, n_samples):
#         return self.get_Tstart(Tstart).get_samples(n_samples)
#
#     def get_start_times(self):
#         return self.values
#
#
# class data_Tstart:
#     def __init__(self, Tstart):
#         self.values = []
#         self.objects = []
#         self.Tstart = Tstart
#
#     def add_samples(self, n_samples, dict):
#         if n_samples in self.values:
#             raise ValueError("already have entry for n_samples=%d", n_samples)
#         else:
#             i = bisect.bisect(self.values, n_samples)
#             self.values.insert(i, n_samples)
#             self.objects.insert(i, dict)
#
#     def get_samples(self, n_samples):
#         if n_samples not in self.values:
#             raise ValueError("no entry for n_samples=%d", n_samples)
#
#         return self.objects[self.values.index(n_samples)]
#
#     def get_sample_sizes(self):
#         return self.values
