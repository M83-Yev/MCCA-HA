import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
from sklearn.cross_decomposition import CCA

# Determine the shape of each subject (how many trails for each subject)
file = "/data2/lmichalke/github/ANCP/MCCA/data/single_trial_no_tSSS.npz"
data = np.load(file, allow_pickle=True)
X = data['X']
y = data['y']
print(y.shape)

n_trials = []
for x in X:
    n_trials.append(len(x))
print(n_trials)  # [324, 319, 290, 330, 327, 329, 292, 305, 328, 323, 314, 327, 326, 325, 331]

# Load transformed data
base_path = '/data2/xwu/PP_2023/'
n_sub = 15
# Check the data
with np.load('/data2/xwu/PP_2023/Hyper_MEG/Trans_data/sub1_out/X_data_sub1_out.npz') as data:
    data_hymeg = np.concatenate((data['X_train'], data['X_test']), axis=0)
print(data_hymeg.shape)  # (4790, 7510)
with np.load('/data2/xwu/PP_2023/MCCA/Trans_data/sub1_out/X_data_sub1_out.npz') as data:
    data_mcmeg = np.concatenate((data['X_train'], data['X_test']), axis=0)
print(data_mcmeg.shape)  # (4790, 7510)


# Reform the shape of the data as (3_types x 751_samples * 10_CCs/PCs)
# function for calculating the mean of each type
def _compute_prototypes(X, y):
    unique_classes = [0, 1, 2]
    prototypes = []

    # go through classes
    for class_ in unique_classes:
        class_data = X[y == class_]

        class_average = np.mean(class_data, axis=0)
        prototypes.append(class_average)

    sub_prototypes = np.concatenate(prototypes, axis=0)

    return sub_prototypes


# function for loading the data
def load_subject_data(method, sub_id):
    file_path = f'{base_path}{method}/Trans_data/sub{sub_id + 1}_out/X_data_sub{sub_id + 1}_out.npz'
    with np.load(file_path) as data:
        data_combined = np.concatenate((data['X_train'], data['X_test']), axis=0)
    return data_combined


# Correct each LOOCV data into the correct shape (33795, 10)
# 33795 = 15 subjects * (3 types * 751 samples)
# 10 = 10 CCs/PCs
def shape_correction(subject_data, n_trials, y):
    # Divide the subject's data into trials based on n_trials
    # and compute prototypes for each trial
    start_idx = 0
    subject_prototypes = []
    for sub, trials in enumerate(n_trials):
        end_idx = start_idx + trials
        trial_data = subject_data[start_idx:end_idx]

        # Compute prototypes for this trial
        trial_prototypes = _compute_prototypes(trial_data, y[sub])
        subject_prototypes.append(trial_prototypes)

        start_idx = end_idx

    # Concatenate all trials' prototypes and reshape
    concatenated_prototypes = np.concatenate(subject_prototypes, axis=0)
    reshaped_prototypes = concatenated_prototypes.reshape(-1, 10)

    return reshaped_prototypes


# combine the functions
def correction(method, sub_id):
    corrected = shape_correction(load_subject_data(method, sub_id), n_trials, y)
    return corrected


# Check the data: if the data loaded correctly
hy1 = correction('Hyper_MEG', 1)
mc1 = correction('MCCA', 1)

print(f'shape for leaving sub_1 out hymeg data: {hy1.shape}')
print(f'shape for leaving sub_1 out mcmeg data: {mc1.shape}')

# Initialization of the CC parameters storages
cca_corr_result = []
cc_corr = [[] for _ in range(10)]

for i in range(n_sub):
    X_hymeg = correction('Hyper_MEG', i)
    X_mcca = correction('MCCA', i)

    print(f"Subject {i + 1} out CV:")
    print("Hyperaligned data shape is: ", X_hymeg.shape)
    print("MCCA transformed data shape is: ", X_mcca.shape)

    cca = CCA(n_components=10)  # 10
    cca.fit(X_hymeg, X_mcca)
    print(f'Now subject {i + 1} cc.fit is: {cca.score(X_hymeg, X_mcca)}')
    X_hyper_c, X_mcca_c = cca.transform(X_hymeg, X_mcca)

    # Compute correlation for each component
    for cc in range(10):
        corr_coeff = np.corrcoef(X_hyper_c[:, cc], X_mcca_c[:, cc])[0, 1]
        cc_corr[cc].append(corr_coeff)

    # Plot CCs
    plt.figure(i)

    # CC1
    plt.subplot(1, 2, 1)
    plt.scatter(X_hyper_c[:, 0], X_mcca_c[:, 0], c='blue', label='CC1')
    plt.xlabel('HY CC1')
    plt.ylabel('MCCA CC1')
    plt.title(f'Correlation between HY and MCCA (CC1) for Sub{i}')
    plt.legend()
    # CC10
    plt.subplot(1, 2, 2)
    plt.scatter(X_hyper_c[:, 9], X_mcca_c[:, 9], c='red', label='CC10')
    plt.xlabel('HY CC10')
    plt.ylabel('MCCA CC10')
    plt.title('Correlation between HY and MCCA (CC10)')
    plt.legend()
    # plt.tight_layout()
    plt.show()

    # Plot CC space HA and MCCA data, first cc against second cc
    # plt.figure(i)
    # plt.scatter(X_hyper_c[:, 0], X_hyper_c[:, 1], label='Hyperalignment')
    # plt.scatter(X_mcca_c[:, 0], X_mcca_c[:, 1], label='MCCA')
    # plt.xlabel('Canonical Component 1')
    # plt.ylabel('Canonical Component 2')
    # plt.title(f'CCA for Subject {i}')
    # plt.legend()
    # plt.show()

# Compute average correlations for each component across subjects
cca_corr_avg = [np.mean(cc_corr[cc]) for cc in range(10)]
cca_corr_result.append(cca_corr_avg)

print("CCA Correlation Results: ", cca_corr_result)
# [[0.9443588590930587, 0.9239623208208966, 0.8891808949563492, 0.8673930038245614, 0.8182260959731,
# 0.7856852713410232, 0.7385813031781467, 0.6797190743514065, 0.5313191741375984, 0.22085018057728853]]
