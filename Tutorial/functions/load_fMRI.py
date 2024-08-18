import os
import nilearn
import numpy as np
from nilearn import datasets, masking
import nibabel as nib


def load_Haxby(data_dir, sub_range=range(1, 7), fetch_data=False, individual_mask=True):
    """
    Load_Haxby: using data of Haxby(2001) 6 subjects data to an object stimuli fMRI recording. The stimuli were
    presented in different orders, for sub5 there was one block recording missed. Therefore, in further step taking the
    stimuli size of sub5. Masks for each subject are available, here we use VT mask.


    Parameter:
        data_dir(str): directory for downloading and saving
        sub_range(int/array): default sub1 to sub6
        fetch_data(bool): if data are not downloaded yet, switch to True (default: False)
        individual_mask(bool): Whether apply individual masks on each subject (default: True)

    return:
        X_array, Y_array

    """

    X = []  # X for all data, in capital, x for every sub
    Y = []  # Y for all labels, in capital, y for every sub

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
            haxby_dataset = nilearn.datasets.fetch_haxby(data_dir=data_dir, subjects=(sub,), fetch_stimuli=True,
                                                         url=None,
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
            if individual_mask:
                maskvt_dir = os.path.join(data_dir,
                                          f'haxby2001\\subj{sub}\\mask4_vt.nii.gz')
            else:
                # TODO: dynamically find and define the largest size of voxel
                maskvt_dir = os.path.join(data_dir,
                                          f'haxby2001\\subj{4}\\mask4_vt.nii.gz')  # Sub 4, bc largest voxel number

        print(f"sub: {sub}, anat_dir: {anat_dir}, func_dir: {func_dir}, maskvt_dir: {maskvt_dir}")

        # label directory
        label_dir = os.path.join(os.path.dirname(func_dir), 'labels.txt')

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
        print(f"\t\t sub {sub} labels shape: {len(labels)}")
        print(f"\t\t sub {sub} number of stimuli: {np.sum(labels_bool)}")

        # y for labels
        y = labels[labels_bool]
        Y.append(y)

        ## data Loading
        func_img = nib.load(func_dir)
        mask_img = nib.load(maskvt_dir)

        # apply Mask on functional data
        masked_func_img = masking.apply_mask(func_img, mask_img)
        # masked_func_img = masking.unmask(masked_func_img, mask_img)

        # save masked functional image
        masked_func_dir = os.path.join(data_dir, f'sub{sub}_masked_func.nii.gz')
        # masked_func_img.to_filename(masked_func_dir)

        # get masked func data
        # masked_func = masked_func_img.get_fdata()
        # print(f"sub {sub} masked data shape: {masked_func_img.shape}")

        # reshape masked func data
        _, timepoints = masked_func_img.shape
        # reshaped_data = masked_func_img.reshape(-1, timepoints)
        # print(f"sub {2} masked data（reshaped）: {reshaped_data.shape}")

        # index masked_func
        x = masked_func_img[labels_bool, :]
        # print(f"sub {sub} x shape: {x.shape}")
        # save data together
        X.append(x)

    X_array = np.array(X, dtype=object)
    Y_array = np.array(Y, dtype=object)

    # Cutting all subjects into the same length of minimum size of blocks
    min_size = min(arr.shape[0] for arr in Y)

    for i in range(6):
        X_array[i] = X_array[i][0:min_size, :]
        Y_array[i] = Y_array[i][0:min_size]

    print(f"\t\t X shape: {X_array.shape}")
    print(f"\t\t Y shape: {Y_array.shape}")

    print("\t\t Saving...")
    if individual_mask:
        np.save(os.path.join(data_dir, 'masked_func_data_diffmask.npy'), X_array)
    else:
        np.save(os.path.join(data_dir, 'masked_func_data_onemask.npy'), X_array)
    np.save(os.path.join(data_dir, 'labels_Tutorial.npy'), Y_array)
    print("\t\t Saved")


# def reorder(ref_label, subject_label, subject_data):
def reorder(data_dir, X_array, Y_array, nr_blocks_selected=None):
    """
    Reorder the subject data into same stimuli order. Because each subject had individual random order stimuli within
    one Block, we arrange them into a single subject data (default: the first subject sequence)

    parameter:
        ref_label(y-array): reference stimuli order
        subject_label(Y-array): all stimuli
        subject_data(X-array): all data
        nr_blocks_selected(Int): determine how many times of iteration will be executed, default None for all blocks
            As accuracy is estimated around 80%, which is high, explore if this still work with fewer training data,
            for example, with 2 blocks of data instead of 11 blocks

    return:
        reordered_data, reordered_label
    """
    nr_blocks = 11
    size_blocks = 63
    nr_type = 7
    size_type = 9
    num_blocks = nr_blocks_selected if nr_blocks_selected is not None else nr_blocks

    X_reordered = np.empty_like(X_array, dtype=object)
    Y_reordered = np.empty_like(Y_array, dtype=object)

    # set first subject data label as reference
    ref_label = Y_array[0]
    # get trail label order (no-repeat)
    ref_order = ref_label[0:len(ref_label):size_type]

    for i, (sub, label) in enumerate(zip(X_array, Y_array)):
        sub_order = label[0:len(ref_label):size_type]
        subject_data = sub.reshape(nr_blocks, nr_type, size_type, -1)  # (11,7,9,Voxel)

        sub_label_reordered = []
        sub_data_reordered = []

        for block in range(num_blocks):
            ref_order_block = ref_order[(0 + block * nr_type): ((1 + block) * nr_type)]
            sub_order_block = sub_order[(0 + block * nr_type): ((1 + block) * nr_type)]

            sub_data_block = subject_data[block, :, :, :]

            # ref index mapping
            ref_map = {label: idx for idx, label in enumerate(ref_order_block)}

            # reorder subject label within block
            sort_idx = np.argsort([ref_map[label] for label in sub_order_block])

            sub_label_block_reordered = [sub_order_block[idx] for idx in sort_idx]
            sub_data_block_reordered = sub_data_block[sort_idx, :, :]

            # reconstruct the sublabel within the single block (repeat labels)
            sub_label_block_reordered = np.repeat(sub_label_block_reordered, size_type)
            sub_label_reordered.append(sub_label_block_reordered)

            # Reshape the reordered data to (size_blocks, -1) before appending
            sub_data_block_reordered = sub_data_block_reordered.reshape(size_blocks, -1)
            sub_data_reordered.append(sub_data_block_reordered)

        Y_reordered[i] = np.vstack(sub_label_reordered).reshape(-1)
        X_reordered[i] = np.vstack(sub_data_reordered)

    print("\t\t Saving reordered data...")
    np.save(os.path.join(data_dir, 'func_data_reordered.npy'), X_reordered)
    np.save(os.path.join(data_dir, 'labels_reordered.npy'), Y_reordered)
    print("\t\t Saved")

    return X_reordered, Y_reordered


def pad_arrays(arrays, fill_value=0):
    """
    Pad arrays to make them the same shape.
    """

    # Find the maximum
    max_shape_1 = max(arr.shape[1] for arr in arrays)

    # Pad each array to the maximum shape in the second dimension
    padded_arrays = []
    for arr in arrays:
        pad_width = [(0, 0), (0, max_shape_1 - arr.shape[1])]
        padded_array = np.pad(arr, pad_width, mode='constant', constant_values=fill_value)
        padded_arrays.append(padded_array)

    return np.array(padded_arrays)
