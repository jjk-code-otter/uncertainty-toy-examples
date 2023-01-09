"""
In this simulation, we look at the effect of rounding on measurement error where the true distribution of
temperatures varies within the rounding window. A number of different examples are run and these are
specified in the "experiments" dictionary.
"""
from pathlib import Path
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats as st

# dict of "experiments".
# The 0 element in each entry is the mean of the true temperature distribution
# the 1 element is each entry is the standard deviation of the true temperature distribution
# the 2 element is a list of the measurements to be processed.
# the 3 element is a boolean which, if set to True will plot a reflected version of the distribution
experiments = {
    'super_jaws': [15.0, 2.0, [11, 11], True],
    'jaws': [15.0, 2.0, [12, 12], True],
    'narrow_jaws': [15.0, 6.0, [12, 12], True],
    'rose_thorn': [15.0, 1.0, [12, 18], False],
    'skewed_mix': [15.0, 1.0, [11, 11, 15], True],
    'natural_mix': [15.0, 1.0, [13, 15, 16, 15, 16, 14, 15, 15, 17, 15, 14, 14], True]
}

# number of samples to draw for each distribution. more samples means more accurate distributions but slower running
n_trials = 10000000

for experiment in experiments:

    mean_temperature = experiments[experiment][0]
    sdv_temperature = experiments[experiment][1]
    measurements = np.array(experiments[experiment][2])
    plot_reflected_distribution = np.array(experiments[experiment][3])

    # We use truncnorm from the scipy.stats package. To truncate the normal distribution in the right places we
    # need to "normalise" the limits for each of the measurements.
    a = ((measurements - 0.5) - mean_temperature) / sdv_temperature
    b = ((measurements + 0.5) - mean_temperature) / sdv_temperature

    # generate the true values for each measurement. These are made by sampling from a truncated normal
    # between the upper and lower limits of the "unrounded" measurement.
    n_measurements = len(measurements)
    true_values = np.zeros((n_trials, n_measurements))
    for i in range(n_measurements):
        true_values[:, i] = st.truncnorm.rvs(a[i], b[i], size=n_trials) * sdv_temperature + mean_temperature

    # Calculate the sum of the true values and subtract the sum of the measured values
    true_values = np.sum(true_values, axis=1) - np.sum(measurements)

    # Plot the distribution
    plt.hist(true_values, bins=100, density=True)
    if plot_reflected_distribution:
        plt.hist(-1 * true_values, bins=100, density=True, alpha=0.25)
    plt.title(experiment)

    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()

    # Plot the theoretical maximum uncertainty range also
    x_low = -1 * float(n_measurements) * 0.5
    x_high = float(n_measurements) * 0.5
    plt.plot([x_low, x_low], [0, 0.3 * ylim[1]])
    plt.plot([x_high, x_high], [0, 0.3 * ylim[1]])

    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()

    # Write out some numbers on the plots
    plt.text(
        xlim[0] + 0.02 * (xlim[1] - xlim[0]),
        ylim[0] + 0.95 * (ylim[1] - ylim[0]),
        f"Prior Mean: {mean_temperature} Stdev: {sdv_temperature}"
    )

    plt.text(
        xlim[0] + 0.02 * (xlim[1] - xlim[0]),
        ylim[0] + 0.90 * (ylim[1] - ylim[0]),
        f"{n_measurements} measurements: {measurements}"
    )

    plt.savefig(Path("Figures") / f"{experiment}.png")
    plt.close()

    print(np.mean(true_values))
