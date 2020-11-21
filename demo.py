import warnings

import matplotlib.cbook
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

ALPHA = 0.1
DT = 0.1

xs_true, ys_true = [], []
xs_noisy, ys_noisy = [], []
xs_drifted, ys_drifted = [], []
xs_fused, ys_fused = [], []

x_fused, y_fused = None, None
x2_drifted_prev, y2_drifted_prev = None, None
x_true, y_true, heading_true = 0, 0, 0

for i in range(300):

    velocity = 5 + i / 100 - i ** 2 / 100000
    angular_velocity = 0.1 + i / 10000

    # True values
    x_true = x_true + velocity * np.cos(heading_true) * DT
    y_true = y_true + velocity * np.sin(heading_true) * DT
    heading_true = heading_true + angular_velocity * DT

    # Values with much high freq noise
    x_noisy = x_true + 5 * np.random.randn()
    y_noisy = y_true + 5 * np.random.randn()

    # Values with much low freq noise (drift)
    x_drifted = x_true + 0.5 * np.random.randn() + i * 0.01
    y_drifted = y_true + 0.5 * np.random.randn() - i * 0.05

    # Fused values by complimentary filter
    if i == 0:
        x_fused = x_drifted
        y_fused = y_drifted
        x2_drifted_prev = x_drifted
        y2_drifted_prev = y_drifted
    else:
        delta_x_drifted = x_drifted - x2_drifted_prev
        delta_y_drifted = y_drifted - y2_drifted_prev

        x2_drifted_prev = x_drifted
        y2_drifted_prev = y_drifted

        x_fused = (1 - ALPHA) * (x_fused + delta_x_drifted) + ALPHA * x_noisy
        y_fused = (1 - ALPHA) * (y_fused + delta_y_drifted) + ALPHA * y_noisy

    xs_true.append(x_true)
    ys_true.append(y_true)
    xs_noisy.append(x_noisy)
    ys_noisy.append(y_noisy)
    xs_drifted.append(x_drifted)
    ys_drifted.append(y_drifted)
    xs_fused.append(x_fused)
    ys_fused.append(y_fused)

    if i % 10 == 0:
        plt.cla()
        plt.plot(xs_true, ys_true, "-", label="True")
        plt.plot(xs_noisy, ys_noisy, ".", label="Noisy")
        plt.plot(xs_drifted, ys_drifted, ".", label="Drifted")
        plt.plot(xs_fused, ys_fused, "-", label="Fused")
        plt.axis("equal")
        plt.legend()
        plt.pause(0.1)

plt.show()
