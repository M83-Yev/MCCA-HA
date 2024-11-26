import nilearn
import numpy as np
import os
import nibabel as nib
#from nilearn.plotting import plot_img
import matplotlib.pyplot as plt

data_dir = 'G:/Data/sub-01/func/sub-01_task-onebacktask_run-01_bold.nii'
#func_dir = os.path.join(data_dir)
func_img = nib.load('sub-01_task-onebacktask_run-01_bold.nii.gz')
# plot_img(func_img)
plt.imshow(func_img.dataobj[:,:,17,0])
plt.show()
