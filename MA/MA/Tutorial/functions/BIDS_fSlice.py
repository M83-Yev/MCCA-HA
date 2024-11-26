import nibabel as nib
import numpy as np
import os

def BIDS_fSlice(main_path, sub_range):
    """
    Extracts 3D slices from 4D fMRI NIfTI files for a range of subjects and saves each slice as a separate 3D NIfTI file.

    Parameters:
    main_path (str): Main directory path containing subject data.
    sub_range (list): List of subject numbers to process.
    output_folder_name (str): Name of the folder where the slices will be saved.
    """
    for sub_num in sub_range:
        sub_id = f'sub-{str(sub_num).zfill(2)}'
        data_dir = os.path.join(main_path, sub_id, 'func')
        data_filename = f'{sub_id}_task-onebacktask_run-01_bold.nii.gz'
        data_path = os.path.join(data_dir, data_filename)
        output_dir = os.path.join(data_dir, 'sliced')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Load the fMRI data
        img = nib.load(data_path)
        data = img.get_fdata()
        affine = img.affine

        print(f"Data dimension for {sub_id}:", data.shape)

        # Extract and save each 3D slice
        for i in range(data.shape[3]):
            # extract 3D data (slice)
            img_3d = data[:, :, :, i]
            new_img = nib.Nifti1Image(img_3d, affine)
            slice_number = str(i + 1).zfill(3)

            # save
            output_path = os.path.join(output_dir, f'slice_{slice_number}.nii')
            nib.save(new_img, output_path)


main_path = 'G:\\Data\\word_obj\\'
sub_range = [1, 2, 3, 4, 5, 6]
BIDS_fSlice(main_path, sub_range)
