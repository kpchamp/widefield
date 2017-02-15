from sklearn.decomposition import FastICA
import numpy as np
import tables as tb

fname="/gscratch/riekesheabrown/kpchamp/data/m187201_150727_decitranspose_detrend.h5"
f=tb.open_file(fname,'r')
X=f.root.data[:,:].T

ica = FastICA()
ica.fit(X)
np.save('/gscratch/riekesheabrown/kpchamp/data/ica_components.py',ica.components_)
