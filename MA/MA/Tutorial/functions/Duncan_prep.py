import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from matplotlib.pyplot import show
import nibabel as nib
from nilearn import image, datasets, plotting, masking
from nilearn.image import resample_to_img
from nilearn.image.resampling import coord_transform
from nilearn.glm.first_level import make_first_level_design_matrix, compute_regressor, spm_hrf, FirstLevelModel
from nilearn.plotting import plot_design_matrix

from MA.Tutorial.functions.config import CONFIG

class Duncan_Prep:
    def __init__(self, sub_range=[1,2,3,4,5,6], VT_atlas='HA'):
        self.events_types = None
        self.gmvt_fmri_data = None
        self.data_prep_Nifti = []
        self.Design_Matrices = []
        self.sub_range = sub_range
        self.VT_atlas = VT_atlas

    def design_matrix(self, plot=True):
        main_path = CONFIG["Prep"]["main_path"]
        for sub in self.sub_range:

            folder_path = os.path.join(main_path, f'sub-{str(sub).zfill(2)}', 'func\\sliced')

            slice_files = sorted(
                [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                 f.endswith('.nii') and f.startswith('wrslice')])
            slices = [nib.load(f) for f in slice_files]

            # merge data into one '.nii' file and data
            dat_prep_Nifti = nib.concat_images(slice_files)
            # data_prep = data_prep_Nifti.get_fdata()

            output_file = os.path.join(main_path, 'prep', f'sub-{str(sub).zfill(2)}', '-run-01.nii.gz')
            output_dir = os.path.dirname(output_file)
            # TODO: check condition, there's some problem, if file deleted, but not folder
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                nib.save(dat_prep_Nifti, output_file)

            self.data_prep_Nifti.append(dat_prep_Nifti)

            ############ Step 2 ##############
            # Load events and create onsets array

            # events_dir = os.path.join(main_path, 'sub-02\\func\\sub-02_task-onebacktask_run-01_events.tsv')
            events_dir = os.path.join(main_path, f'sub-{str(sub).zfill(2)}', 'func',
                                      f'sub-{str(sub).zfill(2)}_task-onebacktask_run-01_events.tsv')
            events_tsv = pd.read_csv(events_dir, sep='\t')

            # events_tsv.head()
            #    onset  duration trial_type      0  1   2    3
            # 0    0.0      0.35      Words   mist  1  NR   NR
            # 1    1.0      0.35      Words  otter  1  NR   NR
            # 2    2.0      0.35      Words  otter  1  27  642
            # 3    3.1      0.35      Words  verse  1  NR   NR
            # 4    4.1      0.35      Words   pail  1  NR   NR

            self.events_types = ['Words', 'Objects', 'Scrambled objects', 'Consonant strings']
            onsets = [events_tsv[events_tsv['trial_type'] == ty]['onset'].values for ty in self.events_types]

            ############ Step 3 ##############
            # compute design matrix
            ## https://nilearn.github.io/dev/auto_examples/04_glm_first_level/plot_design_matrix.html#sphx-glr-auto-examples-04-glm-first-level-plot-design-matrix-py

            events = events_tsv[['onset', 'duration', 'trial_type']]
            tr = 3.0
            n_scans = len(slices)
            frame_times = np.arange(n_scans) * tr

            motion = np.loadtxt(os.path.join(folder_path, 'rp_slice_001.txt'))

            design_matrix = make_first_level_design_matrix(
                frame_times,
                events=events,
                add_regs=motion,
                hrf_model='spm')

            if plot:
                # print(design_matrix.head())
                plot_design_matrix(design_matrix)
                show()

            self.Design_Matrices.append(design_matrix)

        return self.Design_Matrices, self.data_prep_Nifti

    def masker(self, vt_idx=7):

        ############ Step 4 ##############
        # Mask Design

        ### Harvard-Oxford Altas VT-Mask
        # get Harvard-Oxford atlas
        ho_atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        ho_maps = ho_atlas.maps

        # read labels
        for idx, label in enumerate(ho_atlas.labels):
            print(f"{idx}: {label}")

        ### Backup
        # # define VT_area
        # vt_idx = [34, 35, 38, 39, 40]
        #
        # # load atlas image
        # atlas_img = ho_maps
        # atlas_data = atlas_img.get_fdata()
        #
        # # mask design
        # vt_mask_data = np.zeros(atlas_data.shape)
        # for idx in vt_idx:
        #     vt_mask_data[atlas_data == idx] = 1
        #
        # # Mask to Nifti file
        # HO_vt_mask_img = nib.Nifti1Image(vt_mask_data, affine=atlas_img.affine)
        # plotting.plot_roi(HO_vt_mask_img, title='VT Mask')
        # HO_vt_mask_img.to_filename('VT_mask.nii.gz')
        #
        # # get same resolution like data_prep
        # HO_vt_mask_resampled = resample_to_img(HO_vt_mask_img, data_prep_Nifti[0], interpolation='nearest')

        ### Testing VT-area
        # define VT_area
        # vt_idx = [15, 16, 23, 34, 35, 38, 39, 40]
        # vt_idx = [7]
        # load atlas image
        atlas_img = ho_maps
        atlas_data = atlas_img.get_fdata()

        # mask design
        vt_mask_data = np.zeros(atlas_data.shape)
        # HO_masks = []

        #TODO: check here
        if len(vt_idx) == 1:
            vt_mask_data[atlas_data == vt_idx] = 1
        else:
            for idx in vt_idx:
                vt_mask_data[atlas_data == idx] = 1

        # Mask to Nifti file
        HO_vt_mask_img = nib.Nifti1Image(vt_mask_data, affine=atlas_img.affine)
        plotting.plot_roi(HO_vt_mask_img, title='VT Mask')
        HO_vt_mask_img.to_filename('VT_mask.nii.gz')

        # get same resolution like data_prep
        HO_vt_mask_resampled = resample_to_img(HO_vt_mask_img, self.data_prep_Nifti[0], interpolation='nearest')

        # HO_masks.append(HO_vt_mask_resampled)

        ###
        ### Duncan VT Mask
        #TODO: build condition for duncun-vt-mask
        MNI_template = datasets.load_mni152_template()
        duncan_vt_mask = np.zeros(MNI_template.shape, dtype=int)

        # MNI range (Duncan, et al., 2009)
        x_min, x_max = -54, -30
        y_min, y_max = -70, -45
        z_min, z_max = -30, -4

        # inverse MNI 12-DOF affine to transform MNI locations back to matrix indexed
        i_min, j_min, k_min = coord_transform(x_min, y_min, z_min, np.linalg.inv(MNI_template.affine))
        i_max, j_max, k_max = coord_transform(x_max, y_max, z_max, np.linalg.inv(MNI_template.affine))

        # returned results from last step {float}
        i_min, i_max = int(i_min), int(i_max)
        j_min, j_max = int(j_min), int(j_max)
        k_min, k_max = int(k_min), int(k_max)

        # value the voxels in range to 1
        duncan_vt_mask[i_min:i_max + 1, j_min:j_max + 1, k_min:k_max + 1] = 1

        # transform numpy array to nifti
        duncan_vt_mask_img = nib.Nifti1Image(duncan_vt_mask, MNI_template.affine)
        duncan_vt_mask_resampled = image.resample_to_img(duncan_vt_mask_img, self.data_prep_Nifti[0],
                                                         interpolation='nearest')

        ### Grey Matter Mask Design

        # grey matter mask
        # gm_mask = datasets.fetch_icbm152_2009()['gm']
        gm_mask = datasets.load_mni152_gm_template()
        gm_mask_resampled = image.resample_to_img(gm_mask, self.data_prep_Nifti[0], interpolation='nearest')

        # As gm-mask values are not boolean, set a threshold, in oder to regenerate gm-mask into binary mask
        # TODO: function, nilearn.maskers.NiftiMasker, parameter mask_strategy{“background”, “epi”, “whole-brain-template”,
        #  ”gm-template”, “wm-template”}, might avoid using user defined threshold
        binary_gm_mask = image.math_img("img > 0.5", img=gm_mask_resampled)
        # gm_fmri_data = masking.apply_mask(data_prep_Nifti[0], binary_gm_mask)

        ### Combine GM mask and VT mask
        # gmvt_mask = [image.math_img("img1 * img2", img1=binary_gm_mask, img2=HO_mask) for HO_mask in HO_masks]
        gmvt_mask = image.math_img("img1 * img2", img1=binary_gm_mask, img2=HO_vt_mask_resampled)
        # gmvt_mask = image.math_img("img1 * img2", img1=binary_gm_mask, img2=duncan_vt_mask_resampled)
        # plotting.plot_roi(gmvt_mask, title='Combined GM and VT Mask')
        # show()

        ### apply mask on data
        # gmvt_fmri_data_list = []

        self.gmvt_fmri_data = [masking.apply_mask(Nifti, gmvt_mask) for Nifti in self.data_prep_Nifti]
        # gmvt_fmri_data_list.append(gmvt_fmri_data)
        # gmvt_fmri_data = [masking.apply_mask(Nifti, gmvt_mask) for Nifti in data_prep_Nifti]
        # gm_fmri_data = masking.apply_mask(data_prep_Nifti, binary_gm_mask)
        # vt_fmri_data = masking.apply_mask(data_prep_Nifti, HO_vt_mask_resampled)

        return self.gmvt_fmri_data

    def prepare_data(self):
        ############ Step 4 ##############
        # Build X and y

        # Define events:
        # 'Words'              1
        # 'Objects'            2
        # 'Scrambled objects'  3
        # 'Consonant strings'  4

        window_size = 5  # find peak in a certain window size
        peaks = []
        X = []
        Y = []

        for sub in range(len(self.gmvt_fmri_data)):
            x = []
            y = []
            for idx, eve_type in enumerate(self.events_types):
                dat = self.Design_Matrices[sub][eve_type].values
                # find_peaks with window size of 5: find local maximum
                # as window size is not fixed as 5 in design matrix, can be 6, or 7
                # just find the local maximum, and extract the two neighbours to form the window with len 5
                peak, _ = find_peaks(dat, distance=window_size)

                x.append([self.gmvt_fmri_data[sub][p - 2:p + 3, :] for p in peak])
                y.append([[idx + 1] * 5 for p in peak])
                peaks.append(peak)

            X.append(x)
            Y.append(y)

        # Nr_sub = len(X)
        # Nr_event = len(X[0])
        # Nr_trial = len(X[0][0])
        # Nr_rep = X[0][0][0].shape[0]
        # Nr_vox = X[0][0][0].shape[1]

        X_array = [np.array(sub) for sub in X]
        X_array = [sub.reshape(-1, sub.shape[-1]) for sub in X_array]
        Y_array = [np.array(label) for label in Y]
        Y_array = [label.reshape(-2) for label in Y_array]

        return np.array(X_array), np.array(Y_array)