import os
import numpy as np
from matplotlib import pyplot as plt

from Tutorial.functions.LOOCVtool import LOOCVtool
from Tutorial.functions.MCCAtool import whiten, center, zscore
from Tutorial.functions.load_fMRI import load_Haxby, reorder, pad_arrays

# Settings
data_dir = 'G:\\testing\\diff_mask' # diff_mask, one_mask
diff_mask = True  # True = diff masks, False = one mask

# Step 1: Loading fMRI data of Haxby(2001)
print("Step1： Loading Haxby(2001) data and reorder into same length of blocks ")

# Define file paths
label_path = os.path.join(data_dir, 'labels_Tutorial.npy')
if diff_mask:
    data_path = os.path.join(data_dir, 'masked_func_data_diffmask.npy')
else:
    data_path = os.path.join(data_dir, 'masked_func_data_onemask.npy')

# Loading data or downloading data
if os.path.exists(data_path) and os.path.exists(label_path):
    print("\t Data files already exist. Loading data...")
    X_array = np.load(data_path, allow_pickle=True)
    Y_array = np.load(label_path, allow_pickle=True)
else:
    print("\t Data files do not exist. Downloading and processing data...")
    load_Haxby(data_dir, sub_range=range(1, 7), fetch_data=False, individual_mask=diff_mask)
    # Load data
    X_array = np.load(data_path, allow_pickle=True)
    Y_array = np.load(label_path, allow_pickle=True)

# (693, 577)
# (693, 464)
# (693, 307)
# (693, 675)
# (693, 422)
# (693, 348)

# (693,)
# (693,)
# (693,)
# (693,)
# (693,)
# (693,)

# TODO: if mask_boolean changed, data need to be deleted. Then here in data preparation step need to be merged together.

# Loading data or reforming data
if (os.path.exists(os.path.join(data_dir, 'func_data_reordered.npy')) and
        os.path.exists(os.path.join(data_dir, 'labels_reordered.npy'))):
    print("\t Reordered data already exist. Loading data...")
    X_reordered = np.load(os.path.join(data_dir, 'func_data_reordered.npy'), allow_pickle=True)
    Y_reordered = np.load(os.path.join(data_dir, 'labels_reordered.npy'), allow_pickle=True)
    print("\t Done with loading")
else:
    print("\t Data not reordered yet, reordering...")
    X_reordered, Y_reordered = reorder(data_dir, X_array, Y_array)
    print("\t Done with reordering")
# Testing
# sub4 label [43675...] sub1 label [64275]
# (X_array[3][0:9, :] == X_reordered[3][9:18, :]).all()
# (X_array[3][18:27, :] == X_reordered[3][0:9, :]).all()
# padding with 0
# X_reordered = pad_arrays(X_reordered, fill_value=0) # get data into same shape

# Testing
# sub4 label [43675...] sub1 label [64275]
# (X_array[3][0:9, :] == X_reordered[3][9:18, :]).all()


# Step 2: Intra-subject performance

# Step 3: Without alignment technics
# LOOCVtool(X_reordered, Y_reordered)
# chance level ~0.14

# Step 4: MCCA
# X_reordered = np.array([whiten(sub) for sub in X_reordered])    #3d Array
X_reordered = [whiten(sub) for sub in X_reordered]    #list
# X_reordered[np.isnan(X_reordered)] = 0
LOOCVtool(X_reordered, Y_reordered, type='MCCA', n_mcca_components=300)

# test_range = np.arange(10, 1501, 20)
# accuracies = []
# for idx, n_cc in enumerate(test_range):
#     BA,_,_ = LOOCVtool(X_reordered, Y_reordered, type='MCCA', n_mcca_components=n_cc)
#     accuracies.append(np.mean(BA))
#
# plt.plot(test_range, accuracies)
# plt.show()

# Step 5: Hyperalignment
LOOCVtool(X_reordered, Y_reordered, type='HA')
# Processing with subject 1 left out...
# 0 0.8816738816738816
# Processing with subject 2 left out...
# 1 0.922077922077922
# Processing with subject 3 left out...
# 2 0.8903318903318904
# Processing with subject 4 left out...
# 3 0.8600288600288601
# Processing with subject 5 left out...
# 4 0.9191919191919193
# Processing with subject 6 left out...
# 5 0.8989898989898991
# Running time: 35.68 seconds

# a = np.mean(np.array([np.abs(clf.estimators_[i].coef_) for i in range(7)]), axis=1)
# a = np.vstack(np.array([np.abs(clf.estimators_[i].coef_) for i in range(7)]))
# b = np.mean(a)
# b = np.mean(a, axis=0)
# plt.plot(b)
# [<matplotlib.lines.Line2D object at 0x000001F352B6D2D0>]
# plt.show()
# c = np.std(X_test)
# c = np.std(X_test, axis=0)