import nilearn
import numpy as np
import os
from nilearn import datasets, masking
from nilearn.image.image import mean_img
from nilearn.plotting import plot_epi, show, plot_roi
import matplotlib
import nibabel as nib

# Loading the data
data_dir = 'G:\Data\Haxby_2001'

X = []  # X for all data, in capital, x for every sub
Y = []  # Y for all labels, in capital, y for every sub
fetch_data = False  # if False, not download data again
sub_range = range(1, 7)  # define range of working range of sub

# Labels initializing
label_mapping = {
    'rest': 0,
    'bottle': 1,
    'cat': 2,
    'chair': 3,
    'face': 4,
    'house': 5,
    'scissors': 6,
    'shoe': 7,
    'scrambledpix': 0
}

for sub in sub_range:
    # By default, 2nd subject will be fetched
    if fetch_data:
        haxby_dataset = nilearn.datasets.fetch_haxby(data_dir=data_dir, subjects=(sub,), fetch_stimuli=True, url=None,
                                                     resume=True, verbose=1)
        stimulus_information = haxby_dataset.stimuli

        # dataset directory
        anat_dir = haxby_dataset.anat[0]
        func_dir = haxby_dataset.func[0]
        maskvt_dir = haxby_dataset.mask_vt[0]

    else:
        # dataset directory
        anat_dir = os.path.join(data_dir, f'haxby2001\\subj{sub}\\anat.nii.gz')
        func_dir = os.path.join(data_dir, f'haxby2001\\subj{sub}\\bold.nii.gz')
        maskvt_dir = os.path.join(data_dir, f'haxby2001\\subj{sub}\\mask4_vt.nii.gz')

    # label directory
    label_dir = os.path.join(os.path.dirname(anat_dir), 'labels.txt')

    # Initialization
    labels = []
    labels_bool = []
    with open(label_dir, 'r') as label_file:
        lines = label_file.readlines()[1:]  # exclude first line
        for line in lines:
            label = line.strip().split(' ')[0]  # Get labels, ignore numbers
            labels.append(label_mapping[label])
            labels_bool.append(label != 'rest' and label != 'scrambledpix')

    labels_bool = np.array(labels_bool)
    labels_bool = labels_bool.ravel()
    labels = np.array(labels)
    labels = labels.ravel()




    ## data Loading
    func_img = nib.load(func_dir)
    mask_img = nib.load(maskvt_dir)

    # Compute the mean EPI: we do the mean along the axis 3, which is time
    mean_haxby = mean_img(func_dir)

    # Plotting
    plot_epi(mean_haxby, colorbar=True, cbar_tick_format="%i")
    show()
    plot_roi(mask_img, mean_haxby)

    # apply Mask on functional data
    masked_func_img = masking.apply_mask(func_img, mask_img)
    # masked_func_img = masking.unmask(masked_func_img, mask_img)

    # save masked functional image
    masked_func_dir = os.path.join(data_dir, f'sub2{sub}_masked_func.nii.gz')
    # masked_func_img.to_filename(masked_func_dir)

    # get masked func data
    # masked_func = masked_func_img.get_fdata()
    print(f"sub {sub} masked data shape: {masked_func_img.shape}")

    # reshape masked func data
    _, timepoints = masked_func_img.shape
    # reshaped_data = masked_func_img.reshape(-1, timepoints)
    # print(f"sub {2} masked data（reshaped）: {reshaped_data.shape}")

    # index masked_func
    # x = masked_func_img


    ## get data into correct shape of HA and MCCA
    trail_index_all = []
    current_label = None
    y = []
    x = []
    for index, label in enumerate(labels):
        if label != current_label:
            if current_label is not None:
                trail_index = range(trail_start,index)
                trail = masked_func_img[trail_index,:]
                x.append(trail)

            current_label = label
            trail_start = index
            y.append(label)

    if current_label is not None:
        trial = masked_func_img[trail_start:len(masked_func_img), :]
        x.append(trial)

    y_bool = np.array(list(map(lambda x: x != 0, y)))
    y = np.array(y)


    y = y[y_bool]
    x = [trial for i, trial in enumerate(x) if y_bool[i]]
    x = np.array(x)

    # print(f"sub {sub} labels shape: {len(y)}")
    # print(f"sub {sub} data shape: {len(x)}")
    # save data together
    X.append(x)
    Y.append(y)

X_array = np.array(X, dtype=object)
Y_array = np.array(Y, dtype=object)


# np.savez(os.path.join(data_dir, 'masked_func_data.npy'), X=X_array, y=Y_array)
np.savez(os.path.join(data_dir, 'single_trial_no_tSSS.npz'), X=X_array, y=Y_array)
# np.save(os.path.join(data_dir, 'masked_func_data.npy'), X_array)


# # https://nilearn.github.io/dev/auto_examples/01_plotting/plot_visualization.html#sphx-glr-auto-examples-01-plotting-plot-visualization-py
# from nilearn.image.image import mean_img
#
# # Compute the mean EPI: we do the mean along the axis 3, which is time
# func_filename = haxby_dataset.func[0]
# mean_haxby = mean_img(func_filename)
#
# from nilearn.plotting import plot_epi, show
#
# plot_epi(mean_haxby, colorbar=True, cbar_tick_format="%i")
# show()
#
#
# sub_01_dir = 'G:\Data\Haxby_2001\haxby2001\subj1\bold.nii'
# mask_01_dir = 'G:\Data\Haxby_2001\haxby2001\subj1\mask4_vt.nii'
#
# from nilearn.masking import compute_epi_mask
#
# mask_img = compute_epi_mask(func_filename)
#
# # Visualize it as an ROI
# from nilearn.plotting import plot_roi
#
# plot_roi(mask_img, mean_haxby)
# data = mask_img.get_fdata()
# print(f"Data array shape: {data.shape}")
