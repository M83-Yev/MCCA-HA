import nilearn
import numpy as np
import os
from nilearn import datasets, masking

data_dir = 'G:\Data\Schaefer'
nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=1, data_dir=data_dir, base_url=None, resume=True, verbose=1)