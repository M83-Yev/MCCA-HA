import numpy as np
import os
import matplotlib.pyplot as plt

data_dir = 'G:\\testing\\diff_mask\\results'
BAs = np.load(os.path.join(data_dir, f'MCCA_fMRI_Block_test.npy'))

mean_BAs = np.mean(BAs, axis=0)
mean_BAs = np.mean(mean_BAs, axis=0)

std_BAs = np.std(BAs, axis=0)

# X and Y coordinates for the grid
x = np.arange(1, 12)
y = np.arange(1, 12)
X, Y = np.meshgrid(x, y)

# Plotting the 3D plot after averaging both dimensions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Using the final averaged data
Z = mean_BAs
ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_xlabel('Nr. of Blocks for MCCA space')
ax.set_ylabel('Nr. of Blocks for training Classifier')
ax.set_zlabel('Mean Value')

plt.show()




# BA_test = BA_test[:4, :, :]
# CC_range = np.array([50, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500])
# block_range = np.array(range(5, 11))
#
# mean_BA_test = np.mean(BA_test, axis=0)
# X, Y = np.meshgrid(CC_range, block_range)
# Z = mean_BA_test
#
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X, Y, Z, cmap='viridis')
#
# ax.set_xticks(CC_range)
# ax.set_xlabel('CC Range')
# ax.set_ylabel('Block Range')
# ax.set_zlabel('Mean Value')
#
# fig.colorbar(surf)
# plt.show()
