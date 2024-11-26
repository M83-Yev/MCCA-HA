import numpy as np

from MA.Tutorial.functions.CV_Tool import CV_Tool
from MA.Tutorial.functions.Duncan_prep import Duncan_Prep
from MA.Tutorial.functions.config import CONFIG

prep = Duncan_Prep(sub_range=np.array([1, 2, 3, 4, 5, 6]), VT_atlas='HA')
prep.design_matrix(plot=False)
prep.masker(vt_idx=np.array([15, 16]))
X, Y = prep.prepare_data()

CONFIG["MCCA"]["n_components_mcca"] = 100
CV = CV_Tool(method=None, permute=False, seed=10000)
CV.inter_sub(X, Y)
