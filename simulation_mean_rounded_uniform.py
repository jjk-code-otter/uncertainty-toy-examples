from pathlib import Path
import numpy as np
import random
import matplotlib.pyplot as plt


def make_plot(measurements, n_measurements, plot_dir):
    n_trials = 10000000

    draws = np.random.uniform(-0.5, 0.5, (n_measurements, n_trials))
    measurements_reshaped = np.reshape(np.array(measurements[0:n_measurements]), (n_measurements, 1))
    true_values = draws + measurements_reshaped

    sum_of_true_values = np.sum(true_values, axis=0) / float(n_measurements)

    x_low = np.sum(measurements_reshaped) / float(n_measurements) - 0.5
    x_high = np.sum(measurements_reshaped) / float(n_measurements) + 0.5

    plt.hist(
        sum_of_true_values, bins=100, density=True,
        range=(x_low, x_high)
    )
    ylim = plt.gca().get_ylim()[1]
    plt.plot([x_low, x_low], [0, ylim])
    plt.plot([x_high, x_high], [0, ylim])
    plt.title(f'{n_measurements} measurements')

    plt.savefig(plot_dir / f"mean_distrib_{n_measurements}.png")
    plt.close()

    quantiles = np.quantile(sum_of_true_values, [0.005, 0.995])

    print(f"{n_measurements} measurements: {quantiles}, {[x_low, x_high]}")

    quantiles950 = np.quantile(sum_of_true_values, [0.025, 0.975])
    quantiles990 = np.quantile(sum_of_true_values, [0.005, 0.995])
    quantiles999 = np.quantile(sum_of_true_values, [0.0005, 0.9995])

    return (
        (quantiles950[1] - quantiles950[0]) / 2.,
        (quantiles990[1] - quantiles990[0]) / 2.,
        (quantiles999[1] - quantiles999[0]) / 2.,
        (x_high - x_low) / 2.
    )


if __name__ == '__main__':

    out_dir = Path("Figures")

    in_measurements = [12, 15, 17, 22, 18, 15, 11, 19, 20, 11, 11, 17, 21, 13, 13, 19, 20, 19, 16, 11,
                       15, 17, 22, 18, 15, 11, 19, 20, 11, 11, 17, 21, 13, 13, 19, 20, 19, 16, 22, 19]

    summary = np.zeros((len(in_measurements), 4))

    for n_measurements in range(1, len(in_measurements) + 1):
        qs95, qs99, qs999, kips = make_plot(in_measurements, n_measurements, out_dir)

        summary[n_measurements - 1, 0] = qs95
        summary[n_measurements - 1, 1] = qs99
        summary[n_measurements - 1, 2] = qs999
        summary[n_measurements - 1, 3] = kips

    num_obs = np.arange(1, float(len(in_measurements)) + 0.1, 1.0)

    coverage_factor_95 = 1.96
    analytic95 = coverage_factor_95 * (1. / np.sqrt(12.)) / np.sqrt(num_obs)

    coverage_factor_99 = 2.58
    analytic99 = coverage_factor_99 * (1. / np.sqrt(12.)) / np.sqrt(num_obs)

    coverage_factor_999 = 3.29
    analytic999 = coverage_factor_999 * (1. / np.sqrt(12.)) / np.sqrt(num_obs)

    plt.plot(num_obs, summary[:, 0], label='95%')
    plt.plot(num_obs, summary[:, 1], label='99%')
    plt.plot(num_obs, summary[:, 2], label='99.9%')
    plt.plot(num_obs, summary[:, 3], label='AMU')
    plt.plot(num_obs, analytic95, label='Analytic 95%', linestyle='dashed')
    plt.plot(num_obs, analytic99, label='Analytic 99%', linestyle='dashed')
    plt.plot(num_obs, analytic999, label='Analytic 99.9%', linestyle='dashed')

    plt.xlabel('Number of measurements')
    plt.ylabel('Range (degC)')

    plt.ylim(0.0, 0.75)
    plt.xlim(0, len(in_measurements) + 1)

    plt.legend()

    plt.savefig(out_dir / f"mean_summary.png")
    plt.close()
