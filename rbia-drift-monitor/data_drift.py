import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def population_stability_index(expected, actual, bins=10):
    expected_perc, _ = np.histogram(expected, bins=bins)
    actual_perc, _ = np.histogram(actual, bins=bins)
    expected_perc = expected_perc / len(expected)
    actual_perc = actual_perc / len(actual)
    psi = np.sum((expected_perc - actual_perc) * np.log((expected_perc + 1e-8) / (actual_perc + 1e-8)))
    return round(psi, 4)

def ks_drift(expected, actual):
    ks_stat, _ = ks_2samp(expected, actual)
    return round(ks_stat, 4)
