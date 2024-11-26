import numpy as np
from config import CONFIG


def load_single_trial_data(tsss_realignment=False):
    # fn = CONFIG.data_folder + ("single_trial_tSSS.npz" if tsss_realignment else "single_trial_no_tSSS.npz")
    fn = CONFIG.data_folder + "masked_func_data.npy"
    data = np.load(fn, allow_pickle=True)
    return data["X"], data["y"]
