import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier

from MA.Tutorial.functions.Hypertool import HyperAlign, procrustes
from MA.Tutorial.functions.MCCAtool import MCCA
from MA.Tutorial.functions.config import CONFIG


class CV_Tool:
    def __init__(self, visualize=False, method=None, permute=False, seed=0):
        self.seed = seed
        self.permute = permute
        self.BAs = []
        self.visualize = visualize
        self.method = method

    def within_sub(self, X_array, Y_array, n_splits=5):
        """

        Parameters:
            X_array: (array/list): features for subjects with number of len(X_array)
            Y_array: (array/list): labels for subjects with number of len(Y_array)

        Return:
            BAs: (array)：balanced accuracy for each subject data
            plot: optional output
        """

        print("Processing within subject cross-validation...")

        kfold = KFold(n_splits, shuffle=True)
        for i, data in enumerate(X_array):  # for each subject
            y_true_all = []
            y_pred_all = []
            label = Y_array[i]

            for train_idx, test_idx in kfold.split(data):  # for each fold
                X_train, X_test = data[train_idx, :], data[test_idx, :]
                y_train, y_test = label[train_idx], label[test_idx]

                # build within-subject classification LR
                clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', penalty='l2', random_state=0))
                # clf = LogisticRegression(multi_class='multinomial', penalty='l2', random_state=0)
                clf.estimators_[0].coef_
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                y_true_all.append(y_test)
                y_pred_all.append(y_pred)

            _score = balanced_accuracy_score(np.array(y_true_all).reshape(-1), np.array(y_pred_all).reshape(-1))
            print(i, _score)
            self.BAs.append(_score)

        # TODO
        if self.visualize:
            pass

        return self.BAs

    def inter_sub(self, X_array, Y_array):
        """
        Parameters:
            X_array: (array/list): features for subjects with number of len(X_array)
            Y_array: (array/list): labels for subjects with number of len(Y_array)

        Return:
            BAs: (array)：balanced accuracy for leave one out cross validation
            plot: optional output
        """

        print("Processing inter subject cross-validation...")

        nr_sub = len(X_array)
        y_true_all = []
        y_pred_all = []

        start_time = time.time()
        for i in range(nr_sub):
            print(f"Processing with subject {i + 1} left out...")

            X_train, X_test, y_train, y_test = self.transform_data(X_array, Y_array, i)
            X_train = np.vstack(X_train)

            if self.permute:
                y_train = np.hstack(self._random_permutation(y_train, self.seed))
            else:
                y_train = np.hstack(y_train)

            # logistic regression model
            clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', penalty='l2', random_state=0))
            clf.fit(X_train, y_train)
            clf.estimators_[0].coef_
            y_pred = clf.predict(X_test)

            y_true_all.append(y_test)
            y_pred_all.append(y_pred)

            score = balanced_accuracy_score(y_test, y_pred)
            print(i, score)
            self.BAs.append(score)

    def transform_data(self, X_array, Y_array, leave_out_sub):

        train_idx = np.setdiff1d(np.arange(len(X_array)), leave_out_sub)
        X_train = X_array[train_idx]
        y_train = Y_array[train_idx]
        y_test = Y_array[leave_out_sub]

        if self.method is None:
            X_test = X_array[leave_out_sub]

        elif self.method == 'HA':
            X_train, Template = HyperAlign(X_train)
            X_test = procrustes(X_array[leave_out_sub], Template)

        elif self.method == 'MCCA':
            mcca = MCCA(
                n_components_mcca=CONFIG["MCCA"]["n_components_mcca"],
                r=CONFIG["MCCA"]["r"]
            )
            mcca.fit(X_train)
            X_train = mcca.transform(X_train)
            mcca.mcca_space()
            X_test = X_array[leave_out_sub]
            mcca.fit_new_data(X_test)
            X_test = mcca.transform_new_data(X_test)


        return X_train, X_test, y_train, y_test

    def _nested_cv(self, X_train, X_test, Y_train, Y_test):
        pass

    def _random_permutation(self, y, seed):
        """ Randomly permute labels for each subject using the provided seed. """
        tmp = []
        for i in range(len(y)):
            tmp.append(np.random.RandomState(seed=seed).permutation(y[i]))
        return np.array(tmp)


def block_selector(X_array, Y_array, nr_block, nr_rep, keep_block):
    # One block data should have each class shown up once
    # TODO: Works for anatomical and MCCA, maybe not for HA, need reconsider
    # TODO: make variables to be global

    # reshape X_array (list, [sub, block*rep, voxel]) into (list, [sub, block, rep, voxel])
    X_array = [x.reshape(nr_block, nr_rep, x.shape[1]) for x in X_array]  # list:6 (11,63,577)
    Y_array = [y.reshape(nr_block, nr_rep) for y in Y_array]  # list:6 (11,63)

    # X_selected = [x[keep_block, :, :] for x in X_array]
    # Y_selected = [y[keep_block, :] for y in Y_array]

    X_selected = [x[keep_block, :, :].reshape(len(keep_block) * x.shape[1], x.shape[2]) for x in X_array]
    Y_selected = [y[keep_block, :].reshape(-1) for y in Y_array]

    return X_selected, Y_selected

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
#     plt.imshow(confusion_matrix(np.array(y_true_all).reshape(-1), np.array(y_pred_all).reshape(-1)),
#                cmap='Blues')
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
