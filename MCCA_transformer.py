import numpy as np
from sklearn.pipeline import make_pipeline

from transformations import MCCA, WhiteningPCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import StratifiedKFold


class MCCATransformer(BaseEstimator, TransformerMixin):
    """ Implements MCCA transformation for use in sklearn pipelines.

    Parameters:
        n_components_pca (int): Number of PCA components to retain for each subject (default 50)

        n_components_mcca (int): Number of MCCA components to retain (default 10)

        r (int/float): Regularization strength. (default 0)

        nested_cv (bool): If true, use nested cross-validation for new subjects fitted with fit_transform_online.
                          (default True)
    """

    def __init__(self, n_components_pca=50, n_components_mcca=10, r=0, nested_cv=True):
        """ Init. """
        self.pipeline = make_pipeline(WhiteningPCA(n_components_pca), MCCA(n_components_mcca, r))
        self.pipeline_new_subject = make_pipeline(WhiteningPCA(n_components_pca), MCCA(n_components_mcca, r))
        self.nested_cv = nested_cv
        self.cca_averaged = None

    def fit(self, X, y):
        """ Fit the MCCA weights to the training data X with labels y.
        
        Parameters:
            X (ndarray): The training data (subjects x trials x samples x channels)
                
            y (ndarray): Labels corresponding to trials in X. (subjects x trials)
        """
        data_averaged = [_compute_prototypes(X[i], y[i]) for i in range(len(X))]
        data_averaged = np.stack(data_averaged, axis=0)
        print(data_averaged.shape) # 14 2253 305
        # apply M-CCA to averaged data
        cca_averaged = self.pipeline.fit_transform(data_averaged)
        self.cca_averaged = np.mean(cca_averaged, axis=0)
        print(self.cca_averaged.shape) # 2253 10
        return self

    def transform(self, X):
        """ 
        Transform single-trial data from sensor dimensions to CCA dimensions,
        concatenate trials across subjects, and flatten time and CCA dimensions
        for the classifier.
        """
        # X_ = [self.MCCA.transform_trials(X[i], subject=i) for i in range(len(X))]
        X_ = self.pipeline.transform(X)
        X_ = np.concatenate(X_)
        #print(f"X_ before reshape, after CCA transform, shape is {X_.shape}")
        #return X_.reshape((X_.shape[0], -1))
        return X_

    def fit_transform(self, X, y=None, **kwargs):
        return self.fit(X, y).transform(X)

    def fit_online(self, X, y):
        """ 
        Fit MCCA weights to data from a new subject.
        """
        if self.cca_averaged is None:
            raise NotFittedError('MCCA transformer needs to be fitted to ' +
                                 'training data before calling fit_online')
        data_averaged = _compute_prototypes(X, y)
        # Calculate PCA for new subject
        pca_averaged = self.pipeline_new_subject[0].fit_transform(data_averaged[np.newaxis])
        pca_averaged = np.squeeze(pca_averaged)
        # Fit PCA of the new subject to average CCA from training data
        self.pipeline_new_subject[1].mcca_weights = np.dot(np.linalg.pinv(pca_averaged), self.cca_averaged)[np.newaxis]
        return self

    def transform_online(self, X):
        """ 
        Transform data from a new subject from sensor to CCA dimensions, and 
        flatten time and CCA dimensions for the classifier.
        """
        X = self.pipeline_new_subject[0].transform_online(X)
        X = self.pipeline_new_subject[1].transform_online(X)
        return X.reshape((X.shape[0], -1))

    def fit_transform_online(self, X, y):
        if self.nested_cv:
            # Nested cross-validation: 5-fold stratified shuffle split of the new (/left-out) subject's trials
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            Xs = []
            ys = []
            for train, test in cv.split(X, y):
                Xs.append(self.fit_online(X[train], y[train]).transform_online(X[test]))
                ys.append(y[test])
            return np.concatenate(Xs), np.concatenate(ys)
        else:
            return self.fit_online(X, y).transform_online(X), y

    def transform_pca_only(self, X):
        """ 
        Transform single-trial data from sensor dimensions to PCA dimensions,
        concatenate trials across subjects, and flatten time and PCA dimensions
        for the classifier.
        """
        # X_ = [self.MCCA.transform_trials_pca(X[i], subject=i) for i in range(len(X))]
        # X_ = np.concatenate(X_)
        X_ = self.pipeline[0].transform(X)
        return X_.reshape((X_.shape[0], -1))

    def transform_online_pca_only(self, X):
        """ 
        Transform data from a new subject from sensor to PCA dimensions, and 
        flatten time and PCA dimensions for the classifier.
        """
        # X = self.MCCA_new_subject.transform_trials_pca(X)
        X = self.pipeline_new_subject[0].transform(X)
        return X.reshape((X.shape[0], -1))


def _compute_prototypes(X, y):
    return np.concatenate([np.mean(X[np.where(y == class_)], axis=0) for class_ in np.unique(y)])
