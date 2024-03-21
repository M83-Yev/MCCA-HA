import numpy as np
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from scipy.linalg import eigh, norm
from config import CONFIG


class WhiteningPCA(TransformerMixin):
    """ Apply individual-subject whitening PCA. Stores PCA weights, means and sigmas.

    Parameters:
        n_components_pca (int): Number of PCA components to retain for each subject (default 50)

    Attributes:
        mu (ndarray): Mean subtracted before PCA (subjects, sensors)

        sigma (ndarray): PCA standard deviation (subjects, PCs)

        pca_weights (ndarray): PCA weights that transform sensors to PCs for each subject (subjects, sensors, PCs)

        pca_weights_inverse_ (ndarray): Inverted PCA weights that transform PCs to sensors for each
                                        subject (subjects, PCs, sensors)
    """
    def __init__(self, n_components_pca=50):
        self.n_pcs = n_components_pca
        self.pca_weights, self.mu, self.sigma, self.pca_weights_inverse_ = None, None, None, None

    def fit(self, X):
        """ Apply individual-subject PCA. Stores PCA weights, means and sigmas, and returns PCA scores.

        Parameters:
            X (ndarray): Input data in sensor space (subjects, samples, sensors)

        Returns:
            scores (ndarray): PCA scores
        """
        n_subjects, n_samples, n_sensors = X.shape
        X_pca = np.zeros((n_subjects, n_samples, self.n_pcs))
        self.pca_weights = np.zeros((n_subjects, n_sensors, self.n_pcs))
        self.mu = np.zeros((n_subjects, n_sensors))
        self.sigma = np.zeros((n_subjects, self.n_pcs))

        # obtain subject-specific PCAs
        for i in range(n_subjects):
            pca = PCA(n_components=self.n_pcs, svd_solver='full')
            x_i = np.squeeze(X[i]).copy()  # time x sensors
            score = pca.fit_transform(x_i)
            self.pca_weights[i] = pca.components_.T
            self.mu[i] = pca.mean_
            self.sigma[i] = np.sqrt(pca.explained_variance_)
            score /= self.sigma[i]
            X_pca[i] = score

        return self

    def transform(self, X):
        """ Transform single trial data from sensor space to PCA space

        Parameters:
            X (ndarray): Single trial data in sensor space (subjects, trials, samples, sensors)

        Returns:
            X_pca (ndarray): Transformed single trial data in PCA space (subjects, trials, samples, PCs)
        """
        if self.pca_weights is None:
            raise NotFittedError('PCATransformer needs to be fitted before calling transform')
        if type(X) is not list and X.ndim == 3:  # Handle averaged trial case
            X = X[:, np.newaxis]

        X_pca = []
        for i in range(len(X)):
            X[i] -= self.mu[i, np.newaxis, np.newaxis]
            # (subjects, trials, samples, sensors) * (subjects, sensors, PCs) -> (subjects, trials, samples, PCs)
            X_ = np.matmul(X[i], self.pca_weights[i, np.newaxis])
            X_ /= self.sigma[i, np.newaxis, np.newaxis]
            X_pca.append(X_)
        print(f"X_pca[1] shape is {X_pca[0].shape} and {len(X_pca)}")

        if len(X_pca) == 1 or X_pca[0].shape[0] == 1:
            return np.concatenate(X_pca)
        else:
            return X_pca

    def transform_online(self, X):
        if self.pca_weights is None:
            raise NotFittedError('PCATransformer needs to be fitted before calling transform')

        X -= self.mu[0, np.newaxis, np.newaxis]
        # (subjects, trials, samples, sensors) * (subjects, sensors, PCs) -> (subjects, trials, samples, PCs)
        X_ = np.matmul(X, self.pca_weights[0, np.newaxis])
        X_ /= self.sigma[0, np.newaxis, np.newaxis]
        return X_

    def fit_transform(self, X, y=None, **kwargs):
        return self.fit(X).transform(X)

    def transform_inverse(self, X_pca):
        """ Transform single trial data from PCA space back to sensor space.

        Parameters:
            X_pca (ndarray): Single trial data of one subject in PCA space (subjects, trials, samples, PCs)

        Returns:
            X (ndarray): Single trial data transformed back into sensor space
                         (subjects, trials, samples, sensors)
        """
        if self.pca_weights is None:
            raise NotFittedError('PCATransformer needs to be fitted before calling transform_inverse')
        if self.pca_weights_inverse_ is None:
            self.pca_weights_inverse_ = np.linalg.pinv(self.pca_weights)
        X_pca *= self.sigma[np.newaxis, np.newaxis]
        X = np.matmul(X_pca, self.pca_weights_inverse_)
        X += self.mu[:, np.newaxis, np.newaxis]
        return X


class MCCA(TransformerMixin):
    """ Performs multiset canonical correlation analysis with an optional
        regularization term based on spatial similarity of weight maps. The
        stronger the regularization, the more similar weight maps are forced to
        be across subjects. Note that the term 'weights' is used interchangeably
        with PCA / MCCA eigenvectors here.

    Parameters:
        n_components_mcca (int): Number of MCCA components to retain (default 10)

        r (int or float): Regularization strength (default 0)

    Attributes:
        mcca_weights (ndarray): MCCA weights that transform PCAs to MCCAs for each subject (subjects, PCs, CCs)

        mcca_weights_inverse_ (ndarray): Inverted MCCA weights that transform CCs to PCs for each
                                         subject (subjects, sensors, PCs)
    """

    def __init__(self, n_components_mcca=10, r=0):
        self.n_ccs = n_components_mcca
        self.r = r
        self.mcca_weights, self.mcca_weights_inverse_ = None, None

    def fit(self, X, pca_weights=None):
        """ Performs multiset canonical correlation analysis with an optional
            regularization term based on spatial similarity of weight maps. The
            stronger the regularization, the more similar weight maps are forced to
            be across subjects.

        Parameters:
            X (ndarray): Input data in PCA space (subjects, samples, PCs)

            pca_weights (ndarray): PCA weights that transform sensors to PCAs for each subject (subjects, sensors, PCs)
                                    (required if r!=0, ignored otherwise)

        Returns:
            mcca_scores (ndarray): Input data in MCCA space (subjects, samples, CCs).
        """
        n_subjects, _, n_pcs = X.shape
        # R_kl is a block matrix containing all cross-covariances R_kl = X_k^T X_l between subjects k, l, k != l
        # where X is the data in the subject-specific PCA space (PCA scores)
        # R_kk is a block diagonal matrix containing auto-correlations R_kk = X_k^T X_k in its diagonal blocks
        R_kl, R_kk = _compute_cross_covariance(X)
        # Regularization
        if self.r != 0:
            if pca_weights is None:
                raise ValueError("pca_weights was not given, but is required when r!=0")
            # The regularization terms W_kl and W_kk are calculated the same way as R_kl and R_kk above, but using
            # cross-covariance of PCA weights instead of PCA scores
            W_kl, W_kk = _compute_cross_covariance(pca_weights)
            # Add regularization term to R_kl and R_kk
            R_kl += self.r * W_kl
            R_kk += self.r * W_kk
        # Obtain MCCA solution by solving the generalized eigenvalue problem
        #                   R_kl h = p R_kk h
        # where h are the concatenated eigenvectors of all subjects and
        # p are the generalized eigenvalues (canonical correlations).
        # If PCA scores are whitened and no regularisation is used, R_kk is an identity matrix and the generalized
        # eigenvalue problem is reduced to a regular eigenvalue problem
        p, h = eigh(R_kl, R_kk, subset_by_index=(n_subjects * n_pcs - self.n_ccs, n_subjects * n_pcs - 1))
        # eigh returns eigenvalues in ascending order. To pick the k largest from a total of n eigenvalues,
        # we use subset_by_index=(n - k, n - 1).
        # Flip eigenvectors so that they are in descending order
        h = np.flip(h, axis=1)
        # Reshape h from (subjects * PCs, CCs) to (subjects, PCs, CCs)
        h = h.reshape((n_subjects, n_pcs, self.n_ccs))
        # Normalize eigenvectors per subject
        self.mcca_weights = h / norm(h, ord=2, axis=(1, 2), keepdims=True)
        return self

    def transform(self, X_pca):
        """ Use MCCA weights (obtained from averaged data) to transform single
            trial data from PCA space to MCCA space.

        Parameters:
            X_pca (ndarray): Single trial data of one subject in PCA space
                         (trials, samples, PCs)

        Returns:
            X_mcca (ndarray): Transformed single trial data in MCCA space
                            (trials, samples, CCs)
        """
        if self.mcca_weights is None:
            raise NotFittedError('MCCA needs to be fitted before calling transform_trials')
        # if type(X_pca) is not list and X_pca.ndim == 3:  # Handle online / single subject case
        #     X_pca = X_pca[np.newaxis]

        X_mcca = [np.matmul(X_pca[i], self.mcca_weights[i]) for i in range(len(X_pca))]
        return X_mcca

    def transform_online(self, X_pca):
        if self.mcca_weights is None:
            raise NotFittedError('MCCA needs to be fitted before calling transform_trials')
        X_mcca = np.matmul(X_pca, self.mcca_weights[0])
        return X_mcca

    def fit_transform(self, X, y=None, pca_weights=None):
        return self.fit(X, pca_weights).transform(X)

    def transform_inverse(self, X_mcca):
        """ Transform single trial data from MCCA space back to PCA space.

        Parameters:
            X_mcca (ndarray): Single trial data of one subject in MCCA space
                            (trials, samples, CCs)

        Returns:
            X_pca (ndarray): Single trial data transformed back into PCA space
                         (trials, samples, PCs)
        """
        if self.mcca_weights is None:
            raise NotFittedError('MCCA needs to be fitted before calling inverse_transform_trials')
        if self.mcca_weights_inverse_ is None:
            self.mcca_weights_inverse_ = np.linalg.pinv(self.mcca_weights)
        X_pca = np.matmul(X_mcca, self.mcca_weights_inverse_)
        return X_pca


def _compute_cross_covariance(X):
    """ Computes cross-covariance of PCA scores or components between subjects.

    Parameters:
        X (ndarray): PCA scores (subjects, samples, PCs) or weights (subjects, sensors, PCs)

    Returns:
        R_kl (ndarray): Block matrix containing all cross-covariances R_kl = X_k^T X_l between subjects k, l, k != l
                        with shape (subjects * PCs, subjects * PCs)
        R_kk (ndarray): Block diagonal matrix containing auto-correlations R_kk = X_k^T X_k in its diagonal blocks
                        with shape (subjects * PCs, subjects * PCs)
    """
    n_subjects, n_samples, n_pcs = X.shape
    R = np.cov(X.swapaxes(1, 2).reshape(n_subjects * n_pcs, n_samples))
    R_kk = R * np.kron(np.eye(n_subjects), np.ones((n_pcs, n_pcs)))
    R_kl = R - R_kk
    return R_kl, R_kk
