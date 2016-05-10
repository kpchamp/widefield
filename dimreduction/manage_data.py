import bisect


class Dataset:
    def __init__(self, Twin, Tstart, n_samples, data):
        self.windowLength = Twin
        self.startTime = Tstart
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
        self.data.append(dataset.data)

    def get_data(self,t_win=None,t_start=None,n_samples=None):
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
