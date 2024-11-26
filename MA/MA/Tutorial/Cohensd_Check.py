from itertools import combinations
import os
import numpy as np
from MA.Tutorial.functions.CV_Tool import CV_Tool

main_path = 'G:\\Data\\word_obj'

X_array = np.load(os.path.join(main_path, 'X_array.npy'), allow_pickle=True)
Y_array = np.load(os.path.join(main_path, 'Y_array.npy'), allow_pickle=True)

CV = CV_Tool(method='MCCA')
X_train, X_test, y_train, y_test = CV.transform_data(X_array, Y_array, 1)
X_train = np.vstack(X_train)
y_train = np.hstack(y_train)


def Cohensd_check(x, y):
    com_pairs = list(combinations(np.unique(Y_array), 2))
    for pair in com_pairs:
        data_class_1 = x[y == pair[0], :]
        data_class_2 = x[y == pair[1], :]

        mean1, mean2 = np.mean(data_class_1), np.mean(data_class_2)
        std1, std2 = np.std(data_class_1, ddof=1), np.std(data_class_2, ddof=1)

        n1, n2 = len(data_class_1), len(data_class_2)
        pooled_std = np.sqrt(((n1 - 1) * std1 + (n2 - 1) * std2) / (n1 + n2 - 2))
        # pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
        d = (mean1 - mean2) / pooled_std

        print(f'Class {pair[0]} and Class {pair[1]}: Cohens d: {d}')

X = X_array.reshape(X_array.shape[0]*X_array.shape[1], X_array.shape[2])
Y = Y_array.reshape(-1)
Cohensd_check(X, Y)

# data_class_1 = X_train[y_train == 3]
# data_class_2 = X_train[y_train == 2]
# mean1, mean2 = np.mean(data_class_1), np.mean(data_class_2)
# std1, std2 = np.std(data_class_1, ddof=1), np.std(data_class_2, ddof=1)
#
# n1, n2 = len(data_class_1), len(data_class_2)
# pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
# d = (mean1 - mean2) / pooled_std
#
# X = np.vstack(X_array.reshape(X_array.shape[0] * X_array.shape[1], X_array.shape[2]))
# Y = np.hstack(Y_array)
# data_class_1 = X_test[y_test == 3]
# data_class_2 = X_test[y_test == 2]
# mean1, mean2 = np.mean(data_class_1), np.mean(data_class_2)
# std1, std2 = np.std(data_class_1, ddof=1), np.std(data_class_2, ddof=1)
#
# n1, n2 = len(data_class_1), len(data_class_2)
# pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
# d = (mean1 - mean2) / pooled_std
