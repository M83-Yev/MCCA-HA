import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon, kruskal, ks_2samp
import matplotlib.pyplot as plt

# Result with 10 PCs afterwards
result_with_10_pcs = np.array([
    0.3264693264693264, 0.330188679245283, 0.3263888888888889, 0.3271908271908272,
    0.35032568677428494, 0.35307428640761973, 0.32155797101449274, 0.31487711877983915,
    0.30707070707070705, 0.33174603174603173, 0.2980769230769231, 0.30707070707070705,
    0.33007856341189673, 0.3243572318495866, 0.34066176268011134
])

# Without PCA
without_pca = np.array([
    0.6206611373278039, 0.6436842414624111, 0.6370464852607709, 0.7375570733368898,
    0.6036816765788728, 0.6962068128734794, 0.7275060386473431, 0.6089909961869974,
    0.7419191919191919, 0.6982683982683983, 0.6488268021286889, 0.5971612145924072,
    0.7071829405162738, 0.7348510590100804, 0.5910956828388021
])

# With first 10 columns TransMat
with_10col_transmat = np.array([
    0.4682968682968683, 0.3962264150943396, 0.48611111111111116, 0.4556920556920557,
    0.3627867459643161, 0.3865532198865533, 0.3783212560386473, 0.39972443846326383,
    0.35499438832772173, 0.42727272727272725, 0.4070512820512821, 0.37485970819304154,
    0.3515151515151515, 0.37659984143164565, 0.3914924098410337
])

# With 10 template PCs
with_10pcs_template = np.array([
    0.6050765050765051, 0.6517662963616059, 0.64937641723356, 0.7001770744890011,
    0.5781648258283772, 0.6543134043134042, 0.6154194537346711, 0.5902143612419494,
    0.6219416386083053, 0.674939874939875, 0.5858732462506048, 0.5933360104613927,
    0.5958473625140291, 0.6583984596217012, 0.5534066176268011
])

# MCCA:
mcca = np.array([
    0.6469111969111969, 0.6362193616646094, 0.6380385487528345, 0.6941427666198309,
    0.580940243557066, 0.748007098007098, 0.6424540133779264, 0.6039924380787594,
    0.6313692480359148, 0.681208914542248, 0.5379172714078374, 0.6792445350549325,
    0.6872615039281706, 0.6832313965341488, 0.6009587569220597
])


# cca for mcca and HA with 10 pc template
cca_corr_result = [[0.9443588590930587, 0.9239623208208966, 0.8891808949563492, 0.8673930038245614, 0.8182260959731,
                    0.7856852713410232, 0.7385813031781467, 0.6797190743514065, 0.5313191741375984,
                    0.22085018057728853]]


# Perform Mann-Whitney U test
statistic, p_value = mannwhitneyu(with_10pcs_template, mcca)
#(with_10pcs_template, without_pca)
#(with_10pcs_template, mcca)

# Define significance level
alpha = 0.05

# Print the results
print("Mann-Whitney U Test:")
print("Statistic:", statistic)
print("p-value:", p_value)

# Check for significance
if p_value < alpha:
    print("Result is statistically significant; reject the null hypothesis.")
else:
    print("Result is not statistically significant; fail to reject the null hypothesis.")

# KS test
ks_statistic, ks_p_value = ks_2samp(with_10pcs_template, mcca)
print("K-S Test:")
print("Statistic:", ks_statistic)
print("p-value:", ks_p_value)


wilcoxon_statistic, wilcoxon_p_value = wilcoxon(with_10pcs_template, mcca)
print("\nWilcoxon Signed-Rank Test:")
print("Statistic:", wilcoxon_statistic)
print("p-value:", wilcoxon_p_value)






# Visualization

# left figure
# Data preparation
data_to_plot = [with_10pcs_template, mcca]
labels = ['HA', 'MCCA']
means = [np.mean(d) for d in data_to_plot]
stds = [np.std(d) for d in data_to_plot]

# Plotting for left figure
plt.rcParams.update({'font.size': 28, 'font.family': 'sans-serif'})
fig, axs = plt.subplots(1, 2, figsize=(24, 10))

# Plotting left figure
violin_parts = axs[0].violinplot(data_to_plot, vert=False, showmeans=False, showmedians=True)
colors = ['lightgreen', 'pink']
for pc, color in zip(violin_parts['bodies'], colors):
    pc.set_facecolor(color)
    pc.set_edgecolor('black')
    pc.set_alpha(0.6)

for i, (mean, std) in enumerate(zip(means, stds)):
    axs[0].text(0.35, i + 1, f'{mean:.2f} +/- {std:.2f}', verticalalignment='center')

axs[0].set_xlim(0.3, 0.8)
axs[0].set_yticks(np.arange(1, len(labels) + 1))
axs[0].set_yticklabels(labels)
axs[0].set_xlabel('BA')
axs[0].set_title('Accuracy Across Methods (mean +/- SD)')

for spine in ['top', 'right', 'bottom', 'left']:
    axs[0].spines[spine].set_linewidth(3)


# figure right
x_values = range(1, len(cca_corr_result[0]) + 1)
y_values = cca_corr_result[0]

axs[1].plot(x_values, y_values, marker='o', color='b', linestyle='-', linewidth=3.4, markersize=8)
axs[1].set_ylabel('Correlation')
axs[1].set_title('CCA for MCCA and HA Transformed data')
axs[1].set_xticks(x_values)
axs[1].set_xticklabels([f'CC{i}' for i in range(1, len(x_values) + 1)], rotation=45)

for spine in ['top', 'right']:
    axs[1].spines[spine].set_visible(False)
axs[1].grid(True, linestyle='--', alpha=0.7)
axs[1].tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.show()
