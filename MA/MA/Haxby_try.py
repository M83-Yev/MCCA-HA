import numpy as np


label_path = 'G:/Data/Haxby_2001/labels_Tutorial.npy'
data_path = 'G:/Data/Haxby_2001/masked_func_data_Tutorial_675.npy'
# data labeled with 675: 675 Voxels mask for sub4, keep all sub in same voxel number

X = np.load(data_path, allow_pickle=True)
Y = np.load(label_path, allow_pickle=True)

for i in range(6):
    exec(f'sub{i+1} = X[{i}]')
    exec(f'label_{i+1} = Y[{i}]')

for i in range(6):
    exec(f'print("sub{i+1}:", sub{i+1}.shape)')
    exec(f'print("label_{i+1}:", label_{i+1}.shape)')


# sub5 has 693 trails, means 1 blocks less than others
# cut other subjects into the same shape

sub = []
labels = []

for i in range(6):
    subject_data = X[i][0:693,:]
    subject_label = Y[i][0:693]
    sub.append(subject_data)
    labels.append(subject_label)

subs_data = np.array(sub)
subs_labels = np.array(labels)

print('All sub data shape:', subs_data.shape)
print('All sub labels shape:', subs_labels.shape)

nr_blocks = 11
size_blocks = 63
nr_type = 7
size_type = 9


def reorder(ref_label, subject_label, subject_data):
    sub_label_reordered = []
    sub_data_reordered = []

    # get trail label order (no-repeat)
    ref_order = ref_label[0:len(ref_label):size_type]
    sub_order = subject_label[0:len(ref_label):size_type]
    subject_data = subject_data.reshape(nr_blocks, nr_type, size_type, -1)  # (11,7,9,675)

    for block in range(nr_blocks):
        ref_order_block = ref_order[(0 + block * nr_type): ((1 + block) * nr_type)]
        sub_order_block = sub_order[(0 + block * nr_type): ((1 + block) * nr_type)]

        sub_data_block = subject_data[block, :, :, :]

        # ref index mapping
        ref_map = {label: idx for idx, label in enumerate(ref_order_block)}

        # reorder subject label within block
        sort_idx = np.argsort([ref_map[label] for label in sub_order_block])

        sub_label_block_reordered = [sub_order_block[idx] for idx in sort_idx]
        sub_data_block_reordered = sub_data_block[sort_idx, :, :]

        # reconstruct the sublabel within the single block (repeat labels)
        sub_label_block_reordered = np.repeat(sub_label_block_reordered, size_type)
        sub_label_reordered.append(sub_label_block_reordered)

        # Reshape the reordered data to (size_blocks, -1) before appending
        sub_data_block_reordered = sub_data_block_reordered.reshape(size_blocks, -1)
        sub_data_reordered.append(sub_data_block_reordered)

    sub_label_reordered = np.concatenate(sub_label_reordered)
    sub_data_reordered = np.vstack(sub_data_reordered)

    return sub_label_reordered, sub_data_reordered


ref_label = subs_labels[0]
reordered_data = []
reordered_labels = []

for i in range(6):
    sub_label = subs_labels[i]
    sub_data = subs_data[i]
    re_label, re_data = reorder(ref_label, sub_label, sub_data)

    reordered_labels.append(re_label)
    reordered_data.append(re_data)

reordered_data = np.array(reordered_data)
reordered_labels = np.array(reordered_labels)

print("Reordered data shape:", reordered_data.shape)
print("Reordered labels shape:", reordered_labels.shape)

# test whether label same
test = all(reordered_labels[0] == reordered_labels[1])



# Between-subject Classification without Hyperalignment
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

kfold = KFold(n_splits=5, shuffle=True)
pred, test = [], []
accuracy = 0

for train_idx, test_idx in kfold.split(reordered_data):  # for each fold
    X_train, X_test = np.vstack(reordered_data[train_idx, :]), np.vstack(reordered_data[test_idx, :])
    y_train, y_test = np.hstack(reordered_labels[train_idx]), np.hstack(reordered_labels[test_idx])

    # Build single classification model
    linear_SVM = SVC(kernel='linear')
    linear_SVM.fit(X_train, y_train)
    y_pred = linear_SVM.predict(X_test)
    pred += y_pred.tolist()
    test += y_test.tolist()
    accuracy += linear_SVM.score(X_test, y_test)

# Print mean test accuracy of Between-subject model
print('BSC mean test accuracy: {:0.3f}'.format(accuracy / 5))

# Visualize confusion matrix
# X-axis means Predicted labels
# Y-axis means Actual labels
plt.figure(figsize=(5, 5), dpi=120)
conf_mat = confusion_matrix(test, pred)
plt.imshow(conf_mat, cmap='Blues')
plt.clim(0, np.max(conf_mat))
plt.title('Between-subject without Hyperalignment ({:0.3f})'.format(accuracy / 5))
Labels = sorted(np.unique(reordered_labels))  # Assuming Labels are integers or sorted class names
plt.xticks(np.arange(len(Labels)), Labels, rotation=45)
plt.yticks(np.arange(len(Labels)), Labels)
plt.ylim(-0.5, len(Labels) - 0.5)

for j in range(len(Labels)):
    for k in range(len(Labels)):
        c = conf_mat[k, j]
        plt.text(j, k, str(c), va='center', ha='center')

plt.show()

