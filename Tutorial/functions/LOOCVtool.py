import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
import time
from sklearn.multiclass import OneVsRestClassifier
from MA.Tutorial.functions.Hypertool import HyperAlign, procrustes
from MA.Tutorial.functions.MCCAtool import MCCA


def LOOCVtool(X_tr, X_te, y_tr, y_te, type=None, n_mcca_components=675):
    nr_sub = len(X_tr)

    y_true_all = []
    y_pred_all = []
    BAs = []

    start_time = time.time()
    for i in range(nr_sub):  # Leave one subject out
        print(f"Processing with subject {i + 1} left out...")

        # X_train = np.array([X[j] for j in range(nr_sub) if j != i])     # 3d Array
        X_train = [X_tr[j] for j in range(nr_sub) if j != i]     # list
        y_train = np.array([y_tr[j] for j in range(nr_sub) if j != i])
        y_test = y_te[i]

        if type is None:
            X_test = X_te[i]
        elif type == 'HA':
            X_train, Template = HyperAlign(X_train)
            X_test = procrustes(X_te[i], Template)
        elif type == 'MCCA':
            # MCCA
            mcca = MCCA(n_components_mcca=n_mcca_components, r=0)
            mcca.fit(X_train)
            X_train = mcca.transform(X_train)
            X_test = X_te[i]
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

# def LOOCVtool(x_tr, y_tr, x_te, y_te, type_=None, n_mcca_components=675):
#     y_true_all = []
#     y_pred_all = []
#     BAs = []
#
#     if type_ == 'HA':
#         x_tr, Template = HyperAlign(x_tr)
#         x_te = procrustes(x_tr, Template)
#     elif type_ == 'MCCA':
#         mcca = MCCA(n_components_mcca=n_mcca_components, r=0)
#         mcca.fit(x_tr)
#         x_tr = mcca.transform(x_tr)
#         mcca.fit_new_data(x_te)
#         x_te = mcca.transform_new_data(x_te)
#
#     # logistic regression model
#     clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', penalty='l2', random_state=0))
#     # clf = LogisticRegression(multi_class='multinomial', penalty='l2', random_state=0)
#     x_tr = np.vstack(x_tr)
#     y_tr = np.hstack(y_tr)
#     clf.fit(x_tr, y_tr)
#     y_pred = clf.predict(x_te)
#
#     y_true_all.append(y_te)
#     y_pred_all.append(y_pred)
#
#     score = balanced_accuracy_score(y_te, y_pred)
#     BAs.append(score)
#
#     return BAs, y_true_all, y_pred_all
