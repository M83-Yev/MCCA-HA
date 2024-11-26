import os
import time
from itertools import combinations
import random
import numpy as np
from math import ceil

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# from matplotlib import pyplot as plt
from MA.Tutorial.functions.LOOCVtool import LOOCVtool
from MA.Tutorial.functions.MCCAtool import whiten, center, zscore, PCA_60
from MA.Tutorial.functions.config import CONFIG
from MA.Tutorial.functions.load_fMRI import load_Haxby, reorder, pad_arrays, data_selector
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.manifold import TSNE
from MA.Tutorial.functions.CV_Tool import CV_Tool

main_path = 'G:\\Data\\word_obj'

X_array = np.load(os.path.join(main_path, 'X_array.npy'), allow_pickle=True)
Y_array = np.load(os.path.join(main_path, 'Y_array.npy'), allow_pickle=True)
# X_array = [x for x in X_array]
# Y_array = [y for y in Y_array]


####


# Define events:
# 'Words'              1
# 'Objects'            2
# 'Scrambled objects'  3
# 'Consonant strings'  4

## CV within Sub
# print('Result: 5-fold CV for single subject')
# Labels = ['1', '2', '3', '4']
# kfold = KFold(n_splits=5, shuffle=True)
# plt.figure(figsize=(8, 8.5), dpi=120)
# for i, data in enumerate(X_array):  # for each subject
#     y_true_all = []
#     y_pred_all = []
#     BAs = []
#     accuracy = 0
#     label = Y_array[i]
#     for train_idx, test_idx in kfold.split(data):  # for each fold
#         X_train, X_test = data[train_idx, :], data[test_idx, :]
#         y_train, y_test = label[train_idx], label[test_idx]
#
#         # build within-subject classification LR
#         clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', penalty='l2', random_state=0))
#         # clf = LogisticRegression(multi_class='multinomial', penalty='l2', random_state=0)
#         clf.fit(X_train, y_train)
#         y_pred = clf.predict(X_test)
#
#         y_true_all.append(y_test)
#         y_pred_all.append(y_pred)
#
#     score = balanced_accuracy_score(np.array(y_true_all).reshape(-1), np.array(y_pred_all).reshape(-1))
#     print(i, score)
#     BAs.append(score)
#
#     # Visualize confusion matrix
#     # X-axis means Predicted labels
#     # Y-axis means Actual labels
#     plt.subplot(2, 3, i + 1)
#     plt.imshow(confusion_matrix(np.array(y_true_all).reshape(-1), np.array(y_pred_all).reshape(-1)), cmap='Blues')
#     plt.clim(0, 108)
#     plt.title('Subject ' + str(i + 1) + ' ({:0.3f})'.format(score))
#     plt.xticks(np.arange(len(Labels)), Labels, rotation=45)
#     plt.yticks(np.arange(len(Labels)), Labels)
#     plt.ylim(-0.5, len(Labels) - 0.5)
#
#     for j in range(len(Labels)):
#         for k in range(len(Labels)):
#             c = confusion_matrix(np.array(y_true_all).reshape(-1), np.array(y_pred_all).reshape(-1))[k, j]
#             plt.text(j, k, str(c), va='center', ha='center')
# plt.subplots_adjust(wspace=0.3, hspace=0.4)
# plt.show()

# print('Result: MCCA, without whitening')
#
# BAs = []
# for i in range(1):
#     ba, _, _ = LOOCVtool(X_array, Y_array, method='MCCA', n_mcca_components=300)
#     BAs.append(ba)
#
# print('Result: MCCA, with whitening')
# X_array_whiten = [zscore(sub) for sub in X_array]
# BAs = []
# for i in range(1):
#     ba, _, _ = LOOCVtool(X_array_whiten, Y_array, method='MCCA', n_mcca_components=300)
#     BAs.append(ba)

# print('Result: HA, without whitening')
#
# BAs = []
# for i in range(1):
#     ba, _, _ = LOOCVtool(X_array, Y_array, method='HA', n_mcca_components=300)
#     BAs.append(ba)

# print('Result: HA, with whitening')
# X_array_whiten = [zscore(sub) for sub in X_array]
# BAs = []
# for i in range(1):
#     ba, _, _ = LOOCVtool(X_array_whiten, Y_array, method='HA', n_mcca_components=300)
#     BAs.append(ba)


# TODO
### Test: using shorter record
# Nr_run = 6
# Nr_rep = 5
# Nr_type = 4
# Nr_Vox = X_array[0].shape[1]
#
# Array_idx = np.zeros(120, dtype=int)
# values = np.arange(1, 121)
#
# for run in range(Nr_run):
#     for i in range(4):
#         start_idx = run * Nr_rep + i * (Nr_run * Nr_rep)
#         Array_idx[start_idx:start_idx + Nr_rep] = values[run * Nr_rep * Nr_type + i * Nr_rep: run * Nr_rep * Nr_type + (
#                 i + 1) * Nr_rep]
#
# X_reshaped = [x[Array_idx, :] for x in X_array]


############################################
main_path = 'G:\\Data\\word_obj'

X_array = np.load(os.path.join(main_path, 'X_array.npy'), allow_pickle=True)
Y_array = np.load(os.path.join(main_path, 'Y_array.npy'), allow_pickle=True)

idx = np.array([np.arange(10,15), np.arange(45,50), np.arange(65,70), np.arange(90,95)]).reshape(-1)
X_array = [x[idx,:] for x in X_array]
X_array = np.array(X_array)
Y_array = np.array([y[idx] for y in Y_array])
CV = CV_Tool(method='MCCA', permute=False, seed=10000)
CV.inter_sub(np.array(X_array), np.array(Y_array))
CV = CV_Tool(method='HA')
CV.inter_sub(np.array(X_array), np.array(Y_array))
#### Testing best CC
# CC_range = array = np.arange(10, 101, 10)
#
# print('check data with label permutation')
# CV = CV_Tool(method='MCCA', permute=True, seed=10000)
# CV.inter_sub(X_array, Y_array)
#
# ba_within = []
# ba_inter = []
# for nr_cc in CC_range:
#     CONFIG["MCCA"]["n_components_mcca"] = nr_cc
#
#     print('within subject cross validation')
#     CV = CV_Tool(method='MCCA', permute=False, seed=10000)
#     ba_within = CV.within_sub(X_array, Y_array)
#
#     print(f'inter subject cross validation with CC number: {nr_cc}')
#     ba = CV.inter_sub(X_array, Y_array)
#     ba_inter.append(ba)


##### Some Check
# CONFIG["MCCA"]["n_components_mcca"] = 100
# CV = CV_Tool(method='MCCA', permute=False, seed=10000)
# CV.inter_sub(X_array, Y_array)
