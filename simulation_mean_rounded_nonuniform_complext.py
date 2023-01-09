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


# number of samples to draw for each distribution. more samples means more accurate distributions but slower running
n_measurements = 12
n_trials = 1000000

mean_temperature = 15.0
sdv_temperature = 1.0
# Initial "true" values are generated from a normal distribution. These are rounded to give the measurements
true_values = np.random.normal(mean_temperature, sdv_temperature, (n_trials, n_measurements))
rounded_measurements = np.round(true_values)

# Calculate the means of the true values and the rounded values
true_means = np.mean(true_values, axis=1)
rounded_means = np.mean(rounded_measurements, axis=1)

# The errors are the differences between the means based on the true values and rounded values
errors = rounded_means - true_means

# Plot the distribution
plt.hist(errors, bins=100, density=True)

xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()

# Plot the theoretical maximum uncertainty range also
x_low = -0.5
x_high = 0.5
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
    f"{n_measurements} measurements"
)
#plt.show()
plt.savefig(Path("Figures") / f"schematic.png")
plt.close()

print(np.mean(true_values))
