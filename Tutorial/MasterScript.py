import os
from itertools import combinations
import random

import numpy as np
from math import ceil

# from matplotlib import pyplot as plt
from MA.Tutorial.functions.LOOCVtool import LOOCVtool
from MA.Tutorial.functions.MCCAtool import whiten, center, zscore, PCA_60
from MA.Tutorial.functions.load_fMRI import load_Haxby, reorder, pad_arrays, data_selector

# Settings
data_dir = 'G:\\testing\\diff_mask'  # diff_mask, one_mask
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
    X_array = np.load(data_path, allow_pickle=True)  # may need downgraded numpy
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
# if (os.path.exists(os.path.join(data_dir, 'func_data_reordered_list.npy')) and
#         os.path.exists(os.path.join(data_dir, 'labels_reordered_list.npy'))):
#     print("\t Reordered data already exist. Loading data...")
#     X_reordered = np.load(os.path.join(data_dir, 'func_data_reordered_list.npy'), allow_pickle=True)
#     Y_reordered = np.load(os.path.join(data_dir, 'labels_reordered_list.npy'), allow_pickle=True)
#     print("\t Done with loading")
# else:
#     print("\t Data not reordered yet, reordering...")
#     X_reordered, Y_reordered = reorder(data_dir, X_array, Y_array)
#     print("\t Done with reordering")

# X_selected, Y_selected = data_selector(X_array, Y_array, 7)
# X_reordered, Y_reordered = reorder(data_dir, X_selected, Y_selected, nr_blocks_selected=7)

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

# Testing sub blocks accuracy

##################
# CC_range = np.array(range(50, 1500, 50))
# training_range = np.array(range(5, 11))
# testing_range = np.array(range(1,11))
#
# BAs_mean = np.zeros((len(training_range), len(testing_range), len(CC_range)))
# BAs_std = np.zeros((len(training_range), len(testing_range), len(CC_range)))
#
# for i, nr_CC in enumerate(CC_range):  # i:nr_CC
#     BAs = []
#
#     for j, tr_blocks in enumerate(training_range):  # j:nr_training blocks
#         # training_combi = list(combinations(range(11), tr_blocks))
#         x_tr, y_tr = reorder(X_array, Y_array, nr_blocks_selected=tr_blocks)
#         x_tr[j] = [whiten(sub) for sub in x_tr[j]]
#
#         for k, te_blocks in enumerate(testing_range):  # k:nr_testing blocks
#             # testing_range = list(combinations(range(11), te_blocks))
#             BA = []
#
#             x_te, y_te = reorder(X_array, Y_array, nr_blocks_selected=te_blocks)
#             x_te[k] = [whiten(sub) for sub in x_te[k]]
#
#             for tr in range(len(x_tr)):
#                 for te in range(len(x_te)):
#                     BA, _, _ = LOOCVtool(x_tr[tr], x_te[te], y_tr[tr], y_te[te], type='MCCA', n_mcca_components=nr_CC)
#
#             # try:
#             #     BA, _, _ = LOOCVtool(x_tr, x_te, y_tr, y_te, type='MCCA', n_mcca_components=nr_CC)
#             # except Exception as e:
#             #     print(f"Error at index cc{nr_CC}, training{tr_blocks}, testing{te_blocks}: {e}")
#             #     BA = 0
#             #     continue
#
#             BAs.append(np.mean(BA))
#
#             BAs_mean[i][j][k] = np.mean(BAs)
#             BAs_std[i][j][k] = np.std(BAs)
#################
CC_range = np.array([50, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500])
block_range = np.array(range(5, 11))


BAs_mean = np.zeros((len(block_range), len(block_range), len(CC_range)))
BAs_std = np.zeros((len(block_range), len(block_range), len(CC_range)))

BA_all = []

for i, nr_CC in enumerate(CC_range):  # i:nr_CC
    print(f"CC: {nr_CC}")
    BAs = []

    for j, nr_blocks in enumerate(block_range):  # j:nr_training blocks
        print(f"Blocks: {nr_blocks}")
        # training_combi = list(combinations(range(11), tr_blocks))
        BA = []
        x_tr, y_tr = reorder(X_array, Y_array, nr_blocks_selected=nr_blocks)
        x_tr[j] = [whiten(sub) for sub in x_tr[j]]

        x_te, y_te = reorder(X_array, Y_array, nr_blocks_selected=nr_blocks)
        x_te[j] = [whiten(sub) for sub in x_te[j]]

        # for tr in range(len(x_tr)):
        #     for te in range(len(x_te)):
        #         BA, _, _ = LOOCVtool(x_tr[tr], x_te[te], y_tr[tr], y_te[te], type='MCCA', n_mcca_components=nr_CC)

        for run in range(ceil(len(x_te) * 0.3)):
            print(f"run: {run}")
            idx_tr = random.sample(range(len(x_te)),1)
            idx_te = random.sample(range(len(x_te)),1)

            # ba, _, _ = LOOCVtool(x_tr[idx_tr[0]], x_te[idx_te[0]], y_tr[idx_tr[0]], y_te[idx_te[0]], type='MCCA',
            #                      n_mcca_components=nr_CC)

            try:
                ba, _, _ = LOOCVtool(x_tr[idx_tr[0]], x_te[idx_te[0]], y_tr[idx_tr[0]], y_te[idx_te[0]], type='MCCA', n_mcca_components=nr_CC)
            except Exception as e:
                print(f"Error at index nr_cc{nr_CC}, nr_blocks{nr_blocks}: {e}")
                ba = 0
                continue

            BA.append(np.mean(ba))
            print(f"BA {len(BA)}, BA shape {BA[0].shape}")

        BAs.append(BA)
        print(f"BAs {len(BAs)}, BAs shape {len(BAs[0])}")

    BA_all.append(BAs)
np.save('BA_all.npy', np.array(BA_all))





# for idx in range(len(X_reordered)):
#     X_selected, Y_selected = data_selector(X_array, Y_array, 7)
#     X_reordered, Y_reordered = reorder(data_dir, X_selected, Y_selected, nr_blocks_selected=7)
#     X_reordered[idx] = [whiten(sub) for sub in X_reordered[idx]]  # list
#     try:
#         BA, _, _ = LOOCVtool(X_reordered[idx], Y_reordered[idx], type='MCCA', n_mcca_components=300)
#         idxes.append(idx)
#     except Exception as e:
#         print(f"Error at index {idx}: {e}")
#         BA, idxes = 0, 0
#         continue
#     BAs.append(np.mean(BA))
#
# plt.plot(range(len(BAs)), BAs, marker='o')
# plt.xlabel('Index')
# plt.ylabel('Mean BA')
# plt.title('Mean BA vs Index')
# plt.savefig('mean_ba_vs_index.png')
# np.save(os.path.join(data_dir, 'results', 'MCCA_with7Blocks_idx.npy'), idxes)
# np.save(os.path.join(data_dir, 'results', 'MCCA_with7Blocks_BAs.npy'), BAs)

# for idx, sub in enumerate(X_reordered[0]):
#     X_reordered[0][idx] = whiten(sub)     #list

# test_range = np.arange(10, 1501, 20)
# accuracies = []
# for idx, n_cc in enumerate(test_range):
#     BA,_,_ = LOOCVtool(X_reordered, Y_reordered, type='MCCA', n_mcca_components=n_cc)
#     accuracies.append(np.mean(BA))
#
# plt.plot(test_range, accuracies)
# plt.show()

# Step 5: Hyperalignment
# LOOCVtool(X_reordered, Y_reordered, type='HA')

# idx = 0
# LOOCVtool(X_reordered[idx], Y_reordered[idx], type='HA')

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
