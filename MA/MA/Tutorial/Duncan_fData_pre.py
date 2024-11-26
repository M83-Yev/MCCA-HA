import nibabel as nib
import numpy as np
import os
# glm.first_level.make_firstleveldesgin nilearn

data_dir = 'G:\\Data\\word_obj\\sub-01\\func'

# label_path = os.path.join(data_dir, 'labels_Tutorial.npy')
data_path = os.path.join(data_dir, 'sub-01_task-onebacktask_run-01_bold.nii.gz')
output_dir = os.path.join(data_dir, 'sliced')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# data_path2 = 'G:\\Data\\word_obj\\sub-01\\func\\sub-01_task-onebacktask_run-01_bold.nii\\rsub-01_task-onebacktask_run-01_bold.nii'

img = nib.load(data_path)
data = img.get_fdata()
header = img.header
print("Data dimension:", data.shape)

affine = img.affine
for i in range(data.shape[3]):
    # extract 3D data (slice)
    img_3d = data[:, :, :, i]
    new_img = nib.Nifti1Image(img_3d, affine)

    slice_number = str(i + 1).zfill(3)
    # save
    output_path = os.path.join(output_dir, f'slice_{slice_number}.nii')
    nib.save(new_img, output_path)

# img2 =nib.load(output_path)
# data2 = img2.get_fdata()
# print("Data dimension:", data2.shape)

# sliced = 'G:\\Data\\word_obj\\sub-01\\func\\sliced\\swrslice_065.nii'
# img2 = nib.load(sliced)
# data2 = img2.get_fdata()
# print("Data dimension:", data2.shape)
