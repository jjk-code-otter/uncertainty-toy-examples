"""
Simple script to show that the variance of the errors in mean of n rounded measurements falls in a predictable way.
A theoretical variance is calculated and compared to a variance calculated from the randomly generated distributions.
"""
import numpy as np
import random

trials = []

number_of_obs = 100
number_of_trials = 100000


# calculate estimated uncertainty for average of number_of_obs values.
# Variance of uncertainty from rounding is same as rectangular distrib with width one (half either side of the whole
# values)
var = (2.0 * 0.5) ** 2 / 12.
estimated_uncertainty = np.sqrt(var / float(number_of_obs))

# do lots of trials
for j in range(number_of_trials):

    # generate random numbers and then round them
    samples = []
    rounded_samples = []
    for i in range(number_of_obs):
        rn = random.gauss(0, 3)
        samples.append(rn)
        rounded_samples.append(float(round(rn)))

    full_precision_average = np.mean(samples)
    rounded_average = np.mean(rounded_samples)

    trials.append(full_precision_average - rounded_average)

print(np.std(trials), estimated_uncertainty)
