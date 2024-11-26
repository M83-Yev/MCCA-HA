import numpy as np

Array_idx = np.zeros(120, dtype=int)
values = np.arange(1, 121)

Nr_run = 6
Nr_rep = 5
Nr_type = 4

for group in range(Nr_run):
    for i in range(4):
        start_idx = group * Nr_rep + i * (Nr_run * Nr_rep)
        Array_idx[start_idx:start_idx + Nr_rep] = values[group * Nr_rep * Nr_type + i * Nr_rep: group * Nr_rep * Nr_type + (
                    i + 1) * Nr_rep]
