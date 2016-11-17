from sklearn.decomposition import FactorAnalysis
import numpy as np
import tables as tb

# Import data
basepath = "/suppscr/riekesheabrown/kpchamp/data/m187474/150804/"
image_data_path = basepath + "data_detrend_mask.h5"
tb_open = tb.open_file(image_data_path, 'r')
image_data_train = tb_open.root.data[:,20000:183000].T
image_data_test = tb_open.root.data[:,183000:-20000].T
tb_open.close()

# Do model selection for Factor Analysis model
ll = np.load(basepath + 'fa_loglikelihoods.npy').tolist()
for n_components in np.arange(2010,3001,10):
    fa_model = FactorAnalysis(n_components=n_components+1)
    fa_model.fit(image_data_train)
    ll.append(fa_model.score(image_data_test))

np.save(basepath + 'fa_loglikelihoods', np.array(ll))
