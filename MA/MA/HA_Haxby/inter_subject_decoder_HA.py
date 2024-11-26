import os

import numpy as np
import sklearn.cross_decomposition
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
# import xgboost as xgb

import prepare_data
from MCCA_transformer import MCCATransformer
from HyperalignmentTransformer import HyperalignmentTransformer as HATrans
from Hyperalign import Hyperalignment
from config import CONFIG
from sklearn.svm import SVC


def inter_subject_decoder():
    """ Inter-subject decoder using leave-one-subject-out cross-validation

    MCCA is computed based on averaged data from all but one subjects. MCCA projection weights
    for the left-out subject are estimated via linear regression from the trial-averaged
    timeseries of the left-out subject in PCA space to the average of other subjects'
    trial-averaged timeseries in MCCA space. All single-trial data is transformed
    into MCCA space with the computed weights.

    Hyperalignment: similar to MCCA, using the PCs instead of sensors.
    We want to minimize the ||X1 - X2 * TransformMatrix ||_{2}, first of all, multiply the first two subject data,
    get the dot production of X1 and X2, then using SVD get transform matrix for X2, to get aligned with X1.
    Afterwards, update the aligned template on other data-sets. For testing the model, LOOCV is applied.
    By using OLS fit pseudoinverse from the common space (template from training data-sets), test every left-out
    data-set, get the accuracy for the LOOCV.


    Inter-subject classifiers are trained on all but one subject and tested on the
    left-out subject.

    """
    save = CONFIG.results_folder + CONFIG.save_fn + '.npz'
    print(save)

    X, y = prepare_data.load_single_trial_data(tsss_realignment=CONFIG.tsss_realignment)
    for i, x in enumerate(X):
        mean_x, std_x = np.mean(x, axis=(0, 1)), np.std(x, axis=(0, 1))
        X[i] -= mean_x
        X[i] /= std_x

    n_subjects = y.shape[0]
    y_true_all = []
    y_pred_all = []
    BAs = []

    # folder_path = '/data2/xwu/PP_2023/Hyper_MEG/Trans_data_AllSensors'
    for i in range(n_subjects):
        print(f"Processing with subject {i + 1} left out...")
        X_train, X_test, y_train, y_test = _transform_data(X, y, i)

        # # save transformed data for each loop, for further compairson with MCCA transformed data
        # sub_dir = os.path.join(folder_path, f'sub{i+1}_out')
        # os.makedirs(sub_dir, exist_ok=True)
        # np.savez_compressed(os.path.join(sub_dir, f'X_data_sub{i+1}_out.npz'), X_train=X_train, X_test=X_test)
        # print(f"Data for Sub_{i + 1} left out saved")


        # X_train_HA, X_test_HA, _, _ = _transform_data(X, y, i, mode="Hyperalignment")
        # X_train_MCCA, X_test_MCCA, _, _ = _transform_data(X, y, i, mode="MCCA")
        # X_train_HA = [x.reshape(x.shape[0]*751,10) for x in X_train_HA]
        # X_train_MCCA = [x.reshape(x.shape[0] * 751, 10) for x in X_train_MCCA]
        # #cca_train = [np.linalg.svd(x_HA.T @ x_MCCA) for x_HA, x_MCCA in zip(X_train_HA,X_train_MCCA)]
        # cca_train = [sklearn.cross_decomposition.CCA(n_components=10).fit_transform(x_HA, x_MCCA) for x_HA, x_MCCA in zip(X_train_HA,X_train_MCCA)]
        # c = np.corrcoef(cca_train[0][0][:, :], cca_train[0][1][:, :])
        # print(c)
        # u_test,s_test,v_test = np.linalg.svd(X_test_HA.T @ X_test_MCCA)

        print(f"x_train shape is now: {X_train.shape}")
        print(f"x_train {X_train}")
        print(f"y_train shape is now: {y_train.shape}")

        print("Logistic regression classifier fitting is going on")
        clf = LogisticRegression(multi_class='ovr', solver='liblinear', penalty='l2', random_state=0)
        clf.fit(X_train, y_train)
        print("Fitting in logistic regression classifier is done")
        y_pred = clf.predict(X_test)
        #
        # # clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', random_state=0)
        # # model = xgb.XGBClassifier(n_jobs=1)
        # # clf = GridSearchCV(model, {"max_depth": [2, 4], "n_estimators": [50, 100]}, n_jobs=32)
        # # clf = xgb.XGBClassifier(n_jobs=32)


        print(f"Training labels for subject {i + 1} left out: ")
        print(y_train)

        # SVM
        # print("SVM is going on")
        # linear_SVM = SVC(kernel='linear')
        # linear_SVM.fit(X_train, y_train)
        # print("SVM fitting is done")
        # y_pred = linear_SVM.predict(X_test)

        print(f"Predictions for subject {i + 1}: {y_pred}")

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        score = balanced_accuracy_score(y_test, y_pred)
        print(i, score)
        BAs.append(score)
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    np.savez(save, y_true=y_true_all, y_pred=y_pred_all, scores=BAs)


def _transform_data(X, y, left_out_subject, permute=False, seed=0):
    """
    Apply MCCA to averaged data from all but one subjects. MCCA projection
    weights are estimated via linear regression from the trial-averaged
    timeseries of the left out subject in PCA space and the average of other
    subjects' trial-averaged timeseries in MCCA space. All single-trial data
    is transformed into MCCA space.


    Hyperalignment: similar to MCCA, using the PCs instead of sensors.
    We want to minimize the ||X1 - X2 * TransformMatrix ||_{2}, first of all, multiply the first two subject data,
    get the dot production of X1 and X2, then using SVD get transform matrix for X2, to get aligned with X1.
    Afterwards, update the aligned template on other data-sets. For testing the model, LOOCV is applied.
    By using OLS fit pseudoinverse from the common space (template from training data-sets), test every left-out
    data-set, get the accuracy for the LOOCV.

    Parameters:
        X (ndarray): The training data (subjects x trials x samples x channels)

        y (ndarray): Labels corresponding to trials in X (subjects x trials)

        left_out_subject (int): Index of the left-out subject

        permute (bool): Whether to shuffle labels for permutation testing

        seed (int): Seed for shuffling labels when permute is set to True

    Returns:
        X_train (ndarray): The transformed data from all but the left out
            subject in MCCA space, flattened across MCCA and time dimensions for
            the classifier. Trials are concatenated across subjects.
            (trials x (samples x MCCAs))

        X_test (ndarray): The transformed data from the left out subject in MCCA
            space, flattened across MCCA and time dimensions for the classifier.
            (trials x (samples x MCCAs))

        y_train (ndarray): Labels corresponding to trials in X_train.

        y_test (ndarray): Labels corresponding to trials in X_test.

    """
    if permute:
        y = _random_permutation(y, seed)
    n_subjects = y.shape[0]
    n_classes = len(np.unique(np.concatenate(y)))
    leave_one_out = np.setdiff1d(np.arange(n_subjects), left_out_subject)
    transformer = MCCATransformer(CONFIG.n_pcs, CONFIG.n_ccs, CONFIG.r, CONFIG.new_subject_trials == 'nested_cv')
    # HA_transformer = HATrans(CONFIG.n_pcs, CONFIG.new_subject_trials == 'nested_cv')
    if CONFIG.mode == 'MCCA':
        X_train = transformer.fit_transform(X[leave_one_out], y[leave_one_out], )
        y_train = np.concatenate(y[leave_one_out], axis=0)
        X_left_out = X[left_out_subject]
        y_left_out = y[left_out_subject]

        if CONFIG.new_subject_trials in ['all', 'nested_cv']:
            X_test, y_test = transformer.fit_transform_online(X_left_out, y_left_out)
        else:
            if CONFIG.new_subject_trials == 'split':
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
                train, test = next(sss.split(X_left_out, y_left_out))
            else:
                sss = StratifiedShuffleSplit(n_splits=1, train_size=CONFIG.new_subject_trials * n_classes,
                                             test_size=None, random_state=0)
                train, test = next(sss.split(X_left_out, y_left_out))
            transformer.fit_online(X_left_out[train], y_left_out[train])
            X_test = transformer.transform_online(X_left_out[test])
            y_test = y_left_out[test]
    elif CONFIG.mode == 'PCA':
        transformer.fit(X[leave_one_out], y[leave_one_out])
        X_train = transformer.transform_pca_only(X[leave_one_out])
        y_train = np.concatenate(y[leave_one_out], axis=0)
        X_left_out = X[left_out_subject]
        y_left_out = y[left_out_subject]
        transformer.fit_online(X_left_out, y_left_out)
        X_test = transformer.transform_online_pca_only(X_left_out)
        y_test = y_left_out
    elif CONFIG.mode == 'sensorspace':
        X_train = np.concatenate([X[i] for i in leave_one_out])
        X_train = X_train.reshape((X_train.shape[0], -1))
        X_train = _standardize(X_train, axes=(1,)).astype(np.float64)
        y_train = np.concatenate(y[leave_one_out], axis=0)
        X_test = X[left_out_subject].reshape((X[left_out_subject].shape[0], -1))
        X_test = _standardize(X_test, axes=(1,)).astype(np.float64)
        y_test = y[left_out_subject]
    # TODO: Check if code is correct (before christmas break)
    elif CONFIG.mode == 'Hyperalignment':

        # data_averaged = _compute_prototypes(X, y)
        # data_averaged = np.squeeze(data_averaged)

        ha = Hyperalignment()
        X_train, _, _ = ha.fit_transform(X[leave_one_out], y[leave_one_out])
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y[leave_one_out], axis=0)
        X_left_out = X[left_out_subject]
        y_left_out = y[left_out_subject]
        print(f"X_train shape is {X_train.shape}")
        print(f"y_train shape is {y_train.shape}")

        X_test = ha.apply_to_new_data(X_left_out, y_left_out)
        y_test = y_left_out

        # print(f"X_test shape is {X_test.shape}")
        # print(f"y_test shape is {y_test.shape}")

        # X_train = HA_transformer.fit_transform(X[leave_one_out], y[leave_one_out], )
        # y_train = np.concatenate(y[leave_one_out], axis=0)
        # X_left_out = X[left_out_subject]
        # y_left_out = y[left_out_subject]
        #
        # if CONFIG.new_subject_trials in ['all', 'nested_cv']:
        #     X_test, y_test = HA_transformer.fit_transform_online(X_left_out, y_left_out)
        # else:
        #     if CONFIG.new_subject_trials == 'split':
        #         sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
        #         train, test = next(sss.split(X_left_out, y_left_out))
        #     else:
        #         sss = StratifiedShuffleSplit(n_splits=1, train_size=CONFIG.new_subject_trials * n_classes,
        #                                      test_size=None, random_state=0)
        #         train, test = next(sss.split(X_left_out, y_left_out))
        #     HA_transformer.fit_online(X_left_out[train], y_left_out[train])
        #     X_test = HA_transformer.transform_online(X_left_out[test])
        #     y_test = y_left_out[test]
    # # TODO: modify (before christmas break)
    # elif CONFIG.mode == 'Hyperalignment':
    #     X_train = HA_transformer.fit_transform(X[leave_one_out], y[leave_one_out], )
    #     y_train = np.concatenate(y[leave_one_out], axis=0)
    #     X_left_out = X[left_out_subject]
    #     y_left_out = y[left_out_subject]
    #
    #     if CONFIG.new_subject_trials in ['all', 'nested_cv']:
    #         X_test, y_test = HA_transformer.fit_transform(X_left_out, y_left_out)
    #     else:
    #         if CONFIG.new_subject_trials == 'split':
    #             sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
    #             train, test = next(sss.split(X_left_out, y_left_out))
    #         else:
    #             sss = StratifiedShuffleSplit(n_splits=1, train_size=CONFIG.new_subject_trials * n_classes,
    #                                          test_size=None, random_state=0)
    #             train, test = next(sss.split(X_left_out, y_left_out))
    #         HA_transformer.fit_online(X_left_out[train], y_left_out[train])
    #         X_test = HA_transformer.transform(X_left_out[test])
    #         y_test = y_left_out[test]
    # TODO: christmas break modify
    # elif CONFIG.mode == 'Hyperalignment':

    # NOTE: fixed line (new update during christmas in 2023)
    # TODO: changed, if not correct, maybe here
    # 290, as the shortest data has only 290 trails

    # hyperaligner = Hyperalignment()
    #
    # truncated_X = [subject_data[:290, :, :] for subject_data in X[leave_one_out]]
    # truncated_y = [subject_labels[:290] for subject_labels in y[leave_one_out]]
    #
    # aligned_data = hyperaligner.hyperalign(truncated_X)
    # # X_train = np.concatenate([d.reshape(d.shape[0], -1) for d in aligned_data], axis=0)
    # X_train = np.concatenate(aligned_data, axis=0)
    # y_train = np.concatenate(truncated_y, axis=0)
    #
    # X_left_out_truncated = X[left_out_subject][:290, :, :]
    # X_left_out = np.mean(X_left_out_truncated, axis=2)          # Average over the third dimension (sensors)
    # y_left_out = y[left_out_subject][:290]
    #
    # print(f"X_left_out shape: {X_left_out.shape}, y_left_out shape: {y_left_out.shape}")
    #
    # X_test = hyperaligner.procrustes(X_left_out, np.mean(aligned_data, axis=0))
    # y_test = y_left_out

    #### another version
    # hyperaligner = Hyperalignment()
    # aligned_data = hyperaligner.hyperalign(X[leave_one_out])
    # X_train = np.concatenate(aligned_data, axis=0)
    # y_train = np.concatenate(y[leave_one_out], axis=0)

    # X_left_out= np.mean(X[left_out_subject], axis=2) # Average over the third dimension(sensors)
    # X_left_out = X[left_out_subject]  # original line

    # X_left_out= np.mean(X[left_out_subject], axis=2)  # Average over the third dimension(sensors)
    # X_left_out = X[X_left_out][:290]
    # y_left_out = y[left_out_subject][:290]

    # print(f"X_left_out shape: {X_left_out.shape}, y_left_out shape: {y_left_out.shape}")

    # Align the left-out subject's data with the common template
    # X_test = hyperaligner.procrustes(X_left_out, np.mean(aligned_data, axis=0))
    # y_test = y_left_out
    # # TODO: christmas break modify_ PCA Version
    # elif CONFIG.mode == 'Hyperalignment':
    #
    #     # NOTE: fixed line (new update during christmas in 2023)
    #     # TODO: changed, if not correct, maybe here
    #     # 290, as the shortest data has only 290 trails
    #
    #     hyperaligner = Hyperalignment()
    #
    #     truncated_X = [subject_data[:290, :, :] for subject_data in X[leave_one_out]]
    #     truncated_y = [subject_labels[:290] for subject_labels in y[leave_one_out]]
    #
    #     aligned_data = hyperaligner.hyperalign(truncated_X)
    #     # X_train = np.concatenate([d.reshape(d.shape[0], -1) for d in aligned_data], axis=0)
    #     X_train = np.concatenate(aligned_data, axis=0)
    #     y_train = np.concatenate(truncated_y, axis=0)
    #
    #     X_left_out_truncated = X[left_out_subject][:290, :, :]
    #     X_left_out = np.mean(X_left_out_truncated, axis=2)          # Average over the third dimension (sensors)
    #     y_left_out = y[left_out_subject][:290]
    #
    #     print(f"X_left_out shape: {X_left_out.shape}, y_left_out shape: {y_left_out.shape}")
    #
    #     X_test = hyperaligner.procrustes(X_left_out, np.mean(aligned_data, axis=0))
    #     y_test = y_left_out

    # # TODO: new update 08.01.2024
    # elif CONFIG.mode == 'Hyperalignment':
    #     hyperaligner = Hyperalignment()
    #
    #     # Truncate data for all subjects
    #     truncated_X = [subject_data[:290, :, :] for subject_data in X]
    #     truncated_y = [subject_labels[:290] for subject_labels in y]
    #
    #     # Apply hyperalignment
    #     ### Average approach
    #     averaged_X = [np.mean(d, axis=0) for d in truncated_X]
    #     # TODO: new modify at 17.01.2024 (saved in new script)
    #     averaged_X=[]
    #     y_labels = []
    #     for sub_data, sub_label in zip(X,y):
    #         means = None
    #         y_label = None
    #         for class_ in np.unique(sub_label):
    #             label_trials = sub_data[sub_label == class_]
    #             label_mean = np.mean(label_trials, axis=0)
    #             print(f"shape of label_mean is {label_mean.shape}")
    #             if means is None:
    #                 means = label_mean
    #             else:
    #                 means = np.vstack((means, label_mean))
    #             print(f"shape of means is {means.shape}")
    #
    #             if y_label is None:
    #                 y_label = np.full(751, class_)
    #             else:
    #                 next_label = np.full(751, class_)
    #                 y_label = np.concatenate([y_label, next_label])
    #             print(f"shape of y_label is {y_label.shape}")
    #         averaged_X.append(means)
    #         y_labels.append(y_label)
    #
    #         for idx, subject_array in enumerate(averaged_X):
    #             print(f"Shape of array for subject {idx}: {subject_array.shape}")
    #     #
    #     #     # 38-40 line MCCCATransformer
    #     reshaped_X = [subject_data.reshape(290, -1) for subject_data in truncated_X]
    #     aligned_data = hyperaligner.hyperalign(averaged_X)
    #     ### Average approach end
    #
    #     ### PCA approach
    #     pca_transformed_data = []
    #     pca = PCA(n_components=10)
    #     sub_pca = None
    #     for sub in aligned_data:
    #         trans = pca.fit_transform(sub)
    #         print(f"shape of pcatrans {trans.shape}")
    #         if sub_pca is None:
    #             sub_pca = trans
    #         else:
    #             sub_pca = np.concatenate([sub_pca, trans])
    #     sub_pca = sub_pca.reshape(15,2253,10)
    #     print(f"shape of sub_pca{sub_pca.shape}")
    #
    #     # for subject_data in truncated_X:
    #     #     n_trials, n_samples, n_sensors = subject_data.shape
    #     #     # Reshape data for PCA: combining trials and samples into one dimension
    #     #     reshaped_data = subject_data.reshape(n_trials * n_samples, n_sensors)
    #     #
    #     #     # Perform PCA
    #     #     pca = PCA(n_components=50)
    #     #     subject_data_pca = pca.fit_transform(reshaped_data)
    #     #
    #     #     subject_data_pca_2d = subject_data_pca.reshape(n_trials, n_samples, 50).reshape(n_trials, -1)
    #     #     pca_transformed_data.append(subject_data_pca_2d)
    #     #
    #     # aligned_data = hyperaligner.hyperalign(pca_transformed_data)
    #     ### PCA approach end
    #
    #     # Select aligned data for training using leave_one_out indices
    #     X_train = np.concatenate([sub_pca[i] for i in leave_one_out], axis=0)    # mean approach line
    #     # X_train = np.concatenate([d.reshape(d.shape[0], -1) for d in aligned_data], axis=0) # PCA approach line
    #     y_train = np.concatenate([y_labels[i] for i in leave_one_out], axis=0)
    #     # y_train = np.concatenate([y[i] for i in leave_one_out], axis=0)
    #     print(f"y_train shape is now {y_train.shape}")
    #     print(f"x_train shape is now {X_train.shape}")
    #
    #
    #     # Select data for the left-out subject
    #     X_test = sub_pca[left_out_subject]
    #     y_test = y_labels[left_out_subject]

    else:
        raise Exception('Mode must be one of [MCCA, PCA, sensorspace or Hyperalignment].')

    return X_train, X_test, y_train, y_test


# def _compute_prototypes(X, y):
#     return np.concatenate([np.mean(X[np.where(y == class_)], axis=0) for class_ in np.unique(y)])

# def _compute_prototypes(X, y):
#     unique_classes = [0, 1, 2]
#     all_prototypes = []
#
#     for i in range(len(X)):
#         y_ = y[i]
#         X_ = X[i]
#         prototypes = []
#
#         for class_ in unique_classes:
#             class_indices = np.where(y_ == class_)[0]
#             class_data = [X_[index] for index in class_indices]
#
#             class_average = np.mean(np.stack(class_data), axis=0)
#             prototypes.append(class_average)
#
#         sub_prototypes = np.concatenate(prototypes, axis=0)
#         print(f"prototypes {i+1} shape is {sub_prototypes.shape}")
#         all_prototypes.append(np.stack(sub_prototypes))
#         print(f"all_prototypes length is {len(all_prototypes)}")
#         print(f"all_prototypes {i+1} shape is {all_prototypes[i].shape}")
#
#     return all_prototypes


def _random_permutation(y, seed):
    """ Randomly permute labels for each subject using the provided seed. """
    tmp = []
    for i in range(len(y)):
        tmp.append(np.random.RandomState(seed=seed).permutation(y[i]))
    return np.array(tmp)


def _standardize(x, axes=(2,)):
    """ Standardize x over specified axes. """
    mean_ = np.apply_over_axes(np.mean, x, axes)
    std_ = np.apply_over_axes(np.std, x, axes)
    return (x - mean_) / std_


def permutation_test(n_permutations, start_id=0, n_jobs=-1):
    """ Parallelized permutation test of inter-subject decoder

    Parameters:
        n_permutations: Number of permutations to compute

        start_id: Start seed for shuffling labels (default 0)

        n_jobs (int): Number of jobs to use for parallel execution

    """
    save_folder = CONFIG.results_folder + 'permutation_test/temp/' + CONFIG.save_fn + '/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    X, y = prepare_data.load_single_trial_data(tsss_realignment=CONFIG.tsss_realignment)
    save_path_observed = save_folder + CONFIG.save_fn + '_observed.npz'
    _leave_one_out_cv(X, y, False, None, save_path_observed)

    from joblib import dump, load
    mmap_fn = CONFIG.results_folder + 'permutation_test/temp/X.mmap'
    if not os.path.exists(mmap_fn):
        dump(X, mmap_fn)
    X = load(mmap_fn, mmap_mode='r')
    import gc
    gc.collect()
    save_path = save_folder + CONFIG.save_fn + '_perm%d.npz'
    Parallel(n_jobs=n_jobs, backend='loky', verbose=100, mmap_mode='r') \
        (delayed(_leave_one_out_cv)(X, y, True, i + start_id, save_path) for i in range(n_permutations))


def _leave_one_out_cv(X, y, perm, seed, save):
    """ Wraps leave-one-subject-out cross-validation loop for parallelization in permutation test. """
    if perm:
        save = save % seed
    if os.path.exists(save):
        print('File already exists: ' + save)
        return
    n_subjects = y.shape[0]
    y_true_all = []
    y_pred_all = []
    BAs = []
    for i in range(n_subjects):
        # TODO: to be fixed line
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        X_train, X_test, y_train, y_test = _transform_data(X, y, i, permute=perm, seed=seed)

        clf = LogisticRegression(multi_class='ovr', solver='liblinear', penalty='l2', random_state=0)
        # clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        BAs.append(balanced_accuracy_score(y_test, y_pred))
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    np.savez(save, y_true=y_true_all, y_pred=y_pred_all, scores=BAs)


def averaged_inter_subject_correlation(n_folds=5):
    save = CONFIG.results_folder + CONFIG.save_fn + ".npz"
    save_png = CONFIG.results_folder + CONFIG.save_fn + ".png"
    print(save)

    X, y = prepare_data.load_single_trial_data(tsss_realignment=CONFIG.tsss_realignment)
    X_averaged_train = [[] for _ in range(n_folds)]
    X_averaged_test = [[] for _ in range(n_folds)]
    n_subjects = len(y)
    for i in range(n_subjects):
        data_st, labels = X[i], y[i]
        sss = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
        j = 0
        for train, test in sss.split(data_st, labels):
            ave_train = np.concatenate([np.mean(data_st[train][np.where(labels[train] == class_)], axis=0)
                                        for class_ in np.unique(labels)])
            X_averaged_train[j].append(ave_train)
            ave_test = np.concatenate([np.mean(data_st[test][np.where(labels[test] == class_)], axis=0)
                                       for class_ in np.unique(labels)])
            X_averaged_test[j].append(ave_test)
            j += 1
    for j in range(n_folds):
        X_averaged_train[j] = np.stack(X_averaged_train[j], axis=0)
        X_averaged_test[j] = np.stack(X_averaged_test[j], axis=0)

    K = CONFIG.cc_low - CONFIG.cc_high
    corrs = []
    for j in range(n_folds):
        pipeline = make_pipeline(WhiteningPCA(CONFIG.n_pcs), MCCA(CONFIG.n_ccs, CONFIG.r))
        X_mcca_train = pipeline.fit_transform(X_averaged_train[j])
        X_mcca_train = np.stack(X_mcca_train, axis=0)
        X_mcca_test = pipeline.transform(X_averaged_test[j])
        X_mcca_test = np.stack(X_mcca_test, axis=0)

        correlations = np.zeros((K, 2))
        for i in range(K):
            inter_subject_corr_train = np.corrcoef(np.squeeze(X_mcca_train[:, :, i]))
            inter_subject_corr_test = np.corrcoef(np.squeeze(X_mcca_test[:, :, i]))
            # TODO: These means include diagonal values (always 1) -> overestimation!
            # averaged inter-subject correlations over training data
            correlations[i, 0] = np.mean(inter_subject_corr_train)
            # averaged inter-subject correlations over testing data
            correlations[i, 1] = np.mean(inter_subject_corr_test)

        print(j, correlations)
        corrs.append(correlations)
    corrs = np.mean(corrs, axis=0)

    # visualize the CCA performance (inter-subject correlations over testing data)
    plt.plot(corrs)
    plt.legend(('Training', 'Testing'))
    plt.xticks(range(K), range(CONFIG.cc_high, CONFIG.cc_low))
    plt.xlabel('CCA components')
    plt.ylabel('Averaged inter-subject correlations')
    plt.tight_layout()
    plt.savefig(save_png, format='png')
    plt.close()
    np.savez(save, corrs=corrs)
