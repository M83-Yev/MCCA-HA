import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA


class Hyperalignment(BaseEstimator, TransformerMixin):
    """
    Simplified Hyperalignment class for computing transformation matrices and
    applying transformations to new data.
    """

    def __init__(self):
        self._TransMat = None
        self.updated_TransMat = None
        self.updated_template = None
        self.aligned_data = None
        self.pca = PCA(n_components=10)
        # self.methods = '10_template_PCA'  # ('All_sensors','10_TransMat','10_Template_PCA','10_PCA')
        self.methods = '10_template_PCA'  # ('All_sensors','10_TransMat','10_Template_PCA','10_PCA')
    def fit_transform(self, X, y):
        """
        Fit the model to X and then transform X. Computes transformation matrices,
        updated template, and aligned data.

        Parameters:
            X: List of numpy arrays, each representing data for one subject.

        Returns:
            aligned_data: List of transformed numpy arrays.
        """

        # data_averaged = _compute_prototypes(X, y)
        #
        # self.updated_template, self._TransMat = self.compute_update_template(data_averaged)
        # self.aligned_data = self.transform(X)
        # return self.aligned_data, self.updated_template, self._TransMat

        print("Computing prototypes for all subjects...")
        data_averaged = _compute_prototypes(X, y)

        print("Computing the updated template and transformation matrices...")
        self.updated_template, self._TransMat = self.compute_update_template(data_averaged)
        print(f"now the template shape is {self.updated_template.shape}")
        print(f"now the _TransMat shape is {len(self._TransMat)}")
        print(f"now the _TransMat[0] shape is {self._TransMat[0].shape}")

        print("Applying the transformation to the original data...")
        print(f"now the X shape is {X.shape}")
        print(f"now the X[0] shape is {X[0].shape}")
        self.aligned_data = self.transform(X)
        print(f"now the aligned_data[0] shape is {self.aligned_data[0].shape}")

        print("Transformation complete.")

        return self.aligned_data, self.updated_template, self._TransMat

    def compute_update_template(self, X):
        """
        Compute the updated Hyperalignment template.

        Parameters:
            X: List of numpy arrays, each representing data for one subject.

        Returns:
            updated_template: The average template after alignment.
            _TransMat: A list of transformation matrices for each subject.
        """
        template = np.copy(X[0])
        updated_TransMat = []

        for x in range(1, len(X)):
            aligned_source, trans_mat = self.procrustes(X[x], template / x)
            template += aligned_source
            # aligned_source, trans_mat = self.procrustes(X[x], template)
            # template = (template + aligned_source) / 2

            # updated_TransMat.append(trans_mat)

        if self.methods == '10_template_PCA':
            template /= len(X)
            template = self.pca.fit_transform(template)
        else:
            template /= len(X)


        updated_template = np.zeros(template.shape)

        for x in range(len(X)):
            aligned_source, trans_mat = self.procrustes(X[x], template)
            updated_template += aligned_source
            updated_TransMat.append(trans_mat)

        if self.methods == '10_template_PCA':
            pass
            # updated_template /= len(X)
            # updated_template = self.pca.fit_transform(updated_template)
        else:
            updated_template /= len(X)

        return updated_template, updated_TransMat

    def procrustes(self, source, target):
        """
        Procrustes analysis to align two datasets.

        Parameters:
            source: The source data.
            target: The target data.

        Returns:
            source_aligned: The aligned source data.
            transmat: The transformation matrix.
        """

        datas = (source, target)
        # print(f"datas {datas}")

        ssqs = [np.sum(d ** 2, axis=0) for d in datas]
        print(f"shape of source {source.shape}")
        print(f"shape of target {target.shape}")

        norms = [np.sqrt(np.sum(ssq)) for ssq in ssqs]
        # normed = [data/norms for data, norm in zip(datas, norms)]
        # normed_source, normed_target = normed

        normed_source = source / norms[0]
        normed_target = target / norms[1]

        normalization_source = False
        if normalization_source:
            U, s, Vh = np.linalg.svd(np.dot(normed_target.T, normed_source), full_matrices=False)
            T = np.dot(Vh.T, U.T)
            scale = sum(s) * norms[1] / norms[0]  # scale = sum(s) * norms[1] / norms[0]
            # print(f"scale {scale}")
            transmat = scale * T
            source_aligned = np.dot(normed_source, transmat)
        else:
            U, s, Vh = np.linalg.svd(np.dot(normed_target.T, normed_source), full_matrices=False)
            T = np.dot(Vh.T, U.T)
            scale = sum(s) * norms[1] / norms[0]  # scale = sum(s) * norms[1] / norms[0]
            # print(f"scale {scale}")
            transmat = scale * T
            source_aligned = np.dot(source, transmat)

        if self.methods == '10_TransMat':
            transmat = transmat[:, :10]

        return source_aligned, transmat


    def transform(self, X):
        """
        Apply the Hyperalignment transformation to the data.

        Parameters:
            X: List of numpy arrays, each representing data for one subject.

        Returns:
            aligned: List of aligned numpy arrays.
        """
        if self._TransMat is None:
            raise ValueError("The transformation matrix has not been fitted yet.")

        aligned = []
        for x, trans_mat in zip(X, self._TransMat):
            reshaped_x = x.reshape(-1, x.shape[-1])  # Reshape the X into (Trial*Sample x Sensor)
            # normed_x = reshaped_x / np.sqrt(np.sum(x ** 2, axis=0))
            aligned_x = np.dot(reshaped_x, trans_mat)

            if self.methods == '10_PCA':
                aligned_x = np.dot(reshaped_x, trans_mat)
                pca_transformed_x = self.pca.fit_transform(aligned_x)
                reshaped_back_x = pca_transformed_x.reshape(x.shape[0], -1)
                print(f"PCA_transformed aligned data shape is {reshaped_back_x.shape}")
                aligned.append(reshaped_back_x)
            elif self.methods == '10_TransMat':
                trans_mat_10 = trans_mat[:, :10]  # take the first 10 columns of trans_mat
                aligned_x = np.dot(reshaped_x, trans_mat_10)
                reshaped_data = aligned_x.reshape(x.shape[0], -1)
                aligned.append(reshaped_data)
            else:
                aligned_x = np.dot(reshaped_x, trans_mat)
                reshaped_data = aligned_x.reshape(x.shape[0], -1)
                aligned.append(reshaped_data)

            # whether_PCA = False
            # if whether_PCA:
            #     pca_transformed_x = self.pca.fit_transform(aligned_x)
            #     reshaped_back_x = pca_transformed_x.reshape(x.shape[0], -1)
            #     print(f"PCA_transformed aligned data shape is {reshaped_back_x.shape}")
            #     aligned.append(reshaped_back_x)
            # else:
            #     reshaped_data = aligned_x.reshape(x.shape[0],-1)
            #     aligned.append(reshaped_data)

            # aligned.append(aligned_x.reshape(x.shape[0], x.shape[1] , x.shape[2]))  # Reshape back to (Trial x Sample*Sensor)

        self.aligned_data = aligned
        return aligned

    def apply_to_new_data(self, new_data, new_y):
        """
        Apply the Hyperalignment transformation to new data using the updated template.

        Parameters:
            new_data: Numpy array representing new subject data to be aligned.

        Returns:
            aligned_new_data: The aligned new data as a numpy array.
        """
        if self.updated_template is None:
            raise ValueError("The model has not been fitted with initial data yet.")

        # normed_new_data = new_data / np.sqrt(np.sum(new_data ** 2, axis=0))
        # aligned_new_data = self.procrustes(normed_new_data, self.updated_template)

        # averaged_X_new = _compute_prototypes([new_data], [new_y])[0]
        # reshaped_X_new = averaged_X_new.reshape(-1, averaged_X_new.shape[-1])
        # print(f"New data after type averaging shape: {averaged_X_new.shape}")
        # normed_new_data = reshaped_X_new / np.sqrt(np.sum(reshaped_X_new ** 2, axis=0))
        # aligned_new_data, _ = self.procrustes(normed_new_data, self.updated_template)
        #

        averaged_X_new = _compute_prototypes([new_data], [new_y])[0]
        reshaped_X_new = averaged_X_new.reshape(-1, averaged_X_new.shape[-1])
        print(f"New data after type averaging shape: {averaged_X_new.shape}")
        normed_new_data = reshaped_X_new / np.sqrt(np.sum(reshaped_X_new ** 2, axis=0))

        _, trans_mat = self.procrustes(normed_new_data, self.updated_template)

        reshaped_new = new_data.reshape(-1, new_data.shape[-1])
        # aligned_new_data = np.dot(reshaped_new, trans_mat)
        #
        # aligned_new_data = aligned_new_data.reshape(-1, new_data.shape[2])
        # print(f"New data after transforming shape: {aligned_new_data.shape}")

        if self.methods == '10_PCA':
            aligned_new_data = np.dot(reshaped_new, trans_mat)

            aligned_new_data = aligned_new_data.reshape(-1, new_data.shape[2])
            print(f"New data after transforming shape: {aligned_new_data.shape}")

            print("Applying PCA to new aligned data...")
            PCA_new_data = self.pca.transform(aligned_new_data)
            PCA_new_data = PCA_new_data.reshape(new_data.shape[0], -1)
            print(f"New data after PCA shape: {PCA_new_data.shape}")
            return PCA_new_data
        elif self.methods == 'All_sensors':
            aligned_new_data = np.dot(reshaped_new, trans_mat)

            aligned_new_data = aligned_new_data.reshape(-1, new_data.shape[2])
            print(f"New data after transforming shape: {aligned_new_data.shape}")

            reshaped_new = aligned_new_data.reshape(new_data.shape[0], -1)
            return reshaped_new
        elif self.methods == '10_TransMat':
            trans_mat_10 = trans_mat[:, :10]  # take the first 10 columns of trans_mat
            aligned_new_data = np.dot(reshaped_new, trans_mat_10)

            aligned_new_data = aligned_new_data.reshape(-1, 10)
            print(f"New data after transforming shape: {aligned_new_data.shape}")

            reshaped_new = aligned_new_data.reshape(new_data.shape[0], -1)
            return reshaped_new
        elif self.methods == '10_template_PCA':
            aligned_new_data = np.dot(reshaped_new, trans_mat)

            aligned_new_data = aligned_new_data.reshape(-1, 10)
            print(f"New data after transforming shape: {aligned_new_data.shape}")

            reshaped_new = aligned_new_data.reshape(new_data.shape[0], -1)
            return reshaped_new


def _compute_prototypes(X, y):
    unique_classes = [0, 1, 2]
    all_prototypes = []

    # go through subjects
    for i, (X_, y_) in enumerate(zip(X, y)):
        # print(f"Processing subject {i + 1}/{len(X)}")
        prototypes = []

        # go through classes
        for class_ in unique_classes:
            # print(f"  Computing prototypes for class {class_}")
            class_data = X_[y_ == class_]

            class_average = np.mean(class_data, axis=0)
            prototypes.append(class_average)

        sub_prototypes = np.concatenate(prototypes, axis=0)
        all_prototypes.append(sub_prototypes)
        # print(f"  Finished subject {i + 1}: Prototype shape {sub_prototypes.shape}")

    return all_prototypes
