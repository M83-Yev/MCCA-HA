import numpy as np
import torch
from sklearn.base import TransformerMixin
from scipy.linalg import eigh, norm
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class MCCA(TransformerMixin):
    def __init__(self, n_components_mcca=10, r=0):
        self.n_ccs = n_components_mcca
        self.r = r
        self.mcca_weights = None
        self.mcca_weights_inverse_ = None
        self.X_mcca_avg = None

    # Assuming X is an 3D array
    # def fit(self, X):
    #     n_subjects, _, n_pcs = X.shape
    #     R_kl, R_kk = self._compute_cross_covariance(X)
    #
    #     # TODO: regularization to be added in
    #
    #     p, h = eigh(R_kl, R_kk, subset_by_index=(n_subjects * n_pcs - self.n_ccs, n_subjects * n_pcs - 1))
    #     h = np.flip(h, axis=1).reshape((n_subjects, n_pcs, self.n_ccs))
    #     self.mcca_weights = h / norm(h, ord=2, axis=(1, 2), keepdims=True)
    #     return self

    # Assuming X with subs in diff. length (voxels/channels)
    def fit(self, X):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        R_kl, R_kk = self._compute_cross_covariance(X)

        # TODO: regularization to be added in
        # solving non-positive definite problem in scipy linalg\_decomp (Solving generalized eigenvalue problem)
        R_kk = R_kk + 1e-9 * np.eye(R_kk.shape[0])

        p, h_all = eigh(R_kl, R_kk, subset_by_index=(self.size - self.n_ccs, self.size - 1))
        # h = np.flip(h, axis=1).reshape((n_subjects, n_pcs, self.n_ccs))
        h, self.mcca_weights = [], []
        idx_start = 0
        for sub, n_pcs in enumerate(self.n_pcs_list):
            idx_end = idx_start + n_pcs
            h.append(np.flip(h_all[idx_start:idx_end, :self.n_ccs], axis=0))
            idx_start = idx_end
            self.mcca_weights.append(h[sub] / norm(h[sub], ord=2, keepdims=True))
            # self.mcca_weights.append(h[sub] / norm(X[sub] @ h[sub], ord=2, keepdims=True))

        # self.mcca_weights = h / norm(h, ord=2, axis=(1, 2), keepdims=True)
        return self

    def transform(self, X):
        if self.mcca_weights is None:
            raise NotFittedError('MCCA needs to be fitted before calling transform')
        self.X_mcca_transformed = [np.matmul(X[i], self.mcca_weights[i]) for i in range(len(X))]
        self.X_mcca_avg = np.mean(self.X_mcca_transformed, axis=0)
        return self.X_mcca_transformed

    def fit_new_data(self, X_new):
        if self.X_mcca_avg is None:
            raise NotFittedError('MCCA average needs to be computed before calling fit_new_data')
        # print(f"X_new {X_new.shape}, self.X_mcca_avg {self.X_mcca_avg.shape}")
        self.new_mcca_weights = np.dot(np.linalg.pinv(X_new), self.X_mcca_avg)
        return self

    def transform_new_data(self, X_new):
        if self.new_mcca_weights is None:
            raise NotFittedError('New MCCA weights need to be fitted before calling transform_new_data')
        # print(f"X_new {X_new.shape}, self.new_mcca_weights {self.new_mcca_weights.shape}")
        X_new_mcca_transformed = np.dot(X_new, self.new_mcca_weights)
        return X_new_mcca_transformed

    # Assuming X is an 3D array
    # def _compute_cross_covariance(self, X):
    #     n_subjects, n_samples, n_pcs = X.shape
    #     R = np.cov(X.swapaxes(1, 2).reshape(n_subjects * n_pcs, n_samples))
    #     R_kk = R * np.kron(np.eye(n_subjects), np.ones((n_pcs, n_pcs)))
    #     R_kl = R - R_kk
    #
    #     return R_kl, R_kk

    # Assuming X with subs in diff. length (voxels/channels)

    def _compute_cross_covariance(self, X):

        self.n_subjects = len(X)
        self.n_pcs_list = [x.shape[1] for x in X]  #
        self.size = np.sum(self.n_pcs_list)
        # self.r_kk_list = []

        R_kk = np.zeros((self.size, self.size))
        R_kl = np.zeros((self.size, self.size))
        outer_idx_start = 0  # row index

        for idx, x in enumerate(X):
            # r_kk is the covariance matrix of each subject
            r_kk = np.cov(x.T)
            n_pcs = self.n_pcs_list[idx]
            outer_idx_end = outer_idx_start + n_pcs

            # r_kk locates on diagonal position
            R_kk[outer_idx_start:outer_idx_end, outer_idx_start:outer_idx_end] = r_kk
            # self.r_kk_list.append(r_kk)
            # r_kl is a list of the covariance matrix of two diff subjects, will locate on non-diagonal positions
            # initializing column index in outer loop
            inner_idx_start = 0
            for j in range(self.n_subjects):

                n_pcs_j = self.n_pcs_list[j]
                inner_idx_end = inner_idx_start + n_pcs_j

                if j != idx:
                    r_kl = np.cov(x.T, X[j].T)
                    # n_pcs_j = self.n_pcs_list[j]
                    # inner_idx_end = inner_idx_start + n_pcs_j

                    # locate r_kl on corresponding potions
                    R_kl[outer_idx_start:outer_idx_end, inner_idx_start:inner_idx_end] = r_kl[:n_pcs, n_pcs:]
                    R_kl[inner_idx_start:inner_idx_end, outer_idx_start:outer_idx_end] = r_kl[:n_pcs, n_pcs:].T

                inner_idx_start = inner_idx_end

            # update row index from outer loop
            outer_idx_start = outer_idx_end

        return R_kl, R_kk


def center(data):
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    return centered_data


def zscore(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    zscored_data = (data - mean) / std
    return zscored_data


# def whiten(data):
#     data_centered = center(data)
#     cov = np.cov(data_centered, rowvar=False)
#     U, S, V = np.linalg.svd(cov)
#     whitening_matrix = np.dot(U, np.diag(1.0 / np.sqrt(S)))
#     whitened_data = np.dot(data_centered, whitening_matrix)
#     return whitened_data

# def whiten(data):
#     # Center the data
#     whitened_data = np.empty_like(data, dtype=object)
#     for sub in data:
#         data_centered = data[sub] - np.mean(data[sub], axis=0)
#
#     # in case there is nan calculated
#     # std_dev = np.nanstd(data_centered, axis=0)
#     # std_dev[std_dev == 0] = 1e-10
#     # Apply PCA
#         pca = PCA(whiten=True)
#         whitened_data[sub] = pca.fit_transform(data_centered)
#
#     return whitened_data
def whiten(data):
    # Center the data
    data_centered = data - np.mean(data, axis=0)

    # in case there is nan calculated
    # std_dev = np.nanstd(data_centered, axis=0)
    # std_dev[std_dev == 0] = 1e-10
    # Apply PCA
    pca = PCA(n_components=60,whiten=True)
    whitened_data = pca.fit_transform(data_centered)

    return whitened_data

def PCA_60(data):
    # Center the data
    data_centered = data - np.mean(data, axis=0)

    # in case there is nan calculated
    # std_dev = np.nanstd(data_centered, axis=0)
    # std_dev[std_dev == 0] = 1e-10
    # Apply PCA
    pca = PCA(whiten=True)
    whitened_data = pca.fit_transform(data_centered)

    return whitened_data