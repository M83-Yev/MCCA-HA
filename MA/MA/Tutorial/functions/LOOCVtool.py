import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
import time
import random
from itertools import combinations
from sklearn.multiclass import OneVsRestClassifier
from MA.Tutorial.functions.Hypertool import HyperAlign, procrustes
from MA.Tutorial.functions.MCCAtool import MCCA


# def LOOCVtool(X_array, Y_array, method=None, n_mcca_components=675):
#     nr_sub = len(X_array)
#     nr_block = 11
#     nr_rep = 63
#     nr_block = 6
#     nr_rep = 20
#     # reshape X_array (list, [sub, block*rep, voxel]) into (list, [sub, block, rep, voxel])
#     X_array = [x.reshape(nr_block, nr_rep, x.shape[1]) for x in X_array]    # list:6 (11,63,577)
#     Y_array = [y.reshape(nr_block, nr_rep) for y in Y_array]                # list:6 (11,63)
#
#     y_true_all = []
#     y_pred_all = []
#     BAs = np.zeros((nr_sub, nr_block, nr_block))
#
#     # define classifier
#     clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', penalty='l2', random_state=0))
#
#     start_time = time.time()
#
#     for i in range(nr_sub):  # Leave one subject out
#         print(f"Processing with subject {i + 1} left out...")
#
#         X_train = [X_array[j] for j in range(nr_sub) if j != i]
#         X_test = X_array[i]       # list
#         y_train = np.array([Y_array[j] for j in range(nr_sub) if j != i])
#         y_test = Y_array[i]
#
#
#
#         # X_train = np.array([X[j] for j in range(nr_sub) if j != i])     # 3d Array
#         if method is None:
#             X_test = X_array[i]
#         elif method == 'HA':
#
#             X_train = [x.reshape(x.shape[0] * x.shape[1], x.shape[2]) for x in X_train]
#             X_train, Template = HyperAlign(X_train)
#             X_test = procrustes((X_array[i].reshape(X_array[i].shape[0] * X_array[i].shape[1], X_array[i].shape[2])),
#                                 Template)
#             y_train = [y.reshape(y.shape[0] * y.shape[1]) for y in y_train]
#
#             X_train = np.vstack(X_train)
#             y_train = np.hstack(y_train)
#             y_test = np.hstack(y_test)
#
#
#             clf.fit(X_train, y_train)
#             y_pred = clf.predict(X_test)
#
#             y_true_all.append(y_test)
#             y_pred_all.append(y_pred)
#
#             score = balanced_accuracy_score(y_test, y_pred)
#             print(i, score)
#             BAs = score
#
#
#
#
#         elif method == 'MCCA':
#
#             # Select combination of Blocks for forming the CCA space
#             for nr_CCAspace in range(nr_block):
#                 tr_combi = list(combinations(range(nr_block), nr_CCAspace + 1))
#                 tr_idx = random.sample(tr_combi, 1)[0]
#
#
#                 # MCCA
#                 # X_test = X_array[i][tr_idx[0], :, :].reshape(nr_block * nr_rep, X_array[i].shape[2])
#                 mcca = MCCA(n_components_mcca=n_mcca_components, r=0)
#                 # reform X_train back to list, with array(nr_CCAblock * nr_rep, Voxels)
#                 X_train_CCASpace = [x[tr_idx, :, :].reshape(len(tr_idx) * nr_rep, x.shape[2]) for x in X_train]
#                 X_test_CCASpace = X_test[tr_idx, :, :].reshape(len(tr_idx) * nr_rep, X_test.shape[2])
#
#                 mcca.fit(X_train_CCASpace)
#                 mcca.transform(X_train_CCASpace) # mcca.mcca_space()
#                 mcca.fit_new_data(X_test_CCASpace)
#
#                 # X_train_trans = np.vstack(X_train_trans)
#                 # y_train = np.hstack(y_train[:, tr_idx, :].reshape(y_train.shape[0], -1))
#
#             # Select combination of Blocks for classifier
#                 for nr_classi in range(nr_block):
#                     te_combi = list(combinations(range(nr_block), nr_classi + 1))
#                     te_idx = random.sample(te_combi, 1)[0]
#
#                     X_train_classi = [x[te_idx, :, :].reshape(len(te_idx) * nr_rep, x.shape[2]) for x in X_train]
#                     # X_test_classi = X_test[te_idx, :, :].reshape(len(te_idx) * nr_rep, X_test.shape[2])
#                     X_test_classi = X_test[:, :, :].reshape(11 * nr_rep, X_test.shape[2])
#                     X_train_trans = mcca.transform(X_train_classi)
#                     X_train_trans = np.vstack(X_train_trans)
#                     X_test_trans = mcca.transform_new_data(X_test_classi)
#
#                     y_train_selected = np.hstack(y_train[:, te_idx, :].reshape(y_train.shape[0], -1))
#                     # y_test_selected = y_test[te_idx, :].reshape(-1)
#                     y_test_selected = y_test[:, :].reshape(-1)
#
#                     # logistic regression model
#                     clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', penalty='l2', random_state=0))
#                     # clf = LogisticRegression(multi_class='multinomial', penalty='l2', random_state=0)
#                     clf.fit(X_train_trans, y_train_selected)
#                     y_pred = clf.predict(X_test_trans)
#
#                     y_true_all.append(y_test_selected)
#                     y_pred_all.append(y_pred)
#
#                     score = balanced_accuracy_score(y_test_selected, y_pred)
#                     print(i, score)
#                     BAs[i, nr_CCAspace, nr_classi] = score
#
#     end_time = time.time()
#     running_time = end_time - start_time
#     print(f"Running time: {running_time:.2f} seconds")
#
#     return BAs, y_true_all, y_pred_all





def LOOCVtool(X, Y, method=None, n_mcca_components=675):
    nr_sub = len(X)

    y_true_all = []
    y_pred_all = []
    BAs = []

    start_time = time.time()
    for i in range(nr_sub):  # Leave one subject out
        print(f"Processing with subject {i + 1} left out...")

        # X_train = np.array([X[j] for j in range(nr_sub) if j != i])     # 3d Array
        X_train = [X[j] for j in range(nr_sub) if j != i]     # list
        y_train = np.array([Y[j] for j in range(nr_sub) if j != i])
        # X_test = X[i]
        y_test = Y[i]

        if method is None:
            X_test = X[i]
        elif method == 'HA':
            X_train, Template = HyperAlign(X_train)
            X_test = procrustes(X[i], Template)
        elif method == 'MCCA':
            # MCCA
            mcca = MCCA(n_components_mcca=n_mcca_components, r=0)
            mcca.fit(X_train)
            X_train = mcca.transform(X_train)
            mcca.mcca_space()
            X_test = X[i]
            mcca.fit_new_data(X_test)
            X_test = mcca.transform_new_data(X_test)

        # logistic regression model
        clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', penalty='l2', random_state=0))
        # clf = LogisticRegression(multi_class='multinomial', penalty='l2', random_state=0)
        X_train = np.vstack(X_train)
        y_train = np.hstack(y_train)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)

        score = balanced_accuracy_score(y_test, y_pred)
        print(i, score)
        BAs.append(score)

    end_time = time.time()
    running_time = end_time - start_time
    print(f"Running time: {running_time:.2f} seconds")

    return BAs, y_true_all, y_pred_all