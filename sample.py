import warnings

import matplotlib.cbook
import matplotlib.pyplot as plt
import numpy as np

from complementary_filter import ComplementaryFilter

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

COEFFICIENTS = [-0.0004, 0.1, 0.1, 2, 20]


def func(x, coeff):
    return coeff[0] * x ** 3 + coeff[1] * x ** 2 + coeff[2] * x + coeff[3] + coeff[4] * np.sin(0.2 * x)


t = np.arange(1, 250)
true_signal = func(t, COEFFICIENTS)

sensor1_signal = true_signal + 5 * np.random.randn(len(true_signal)) + 1 * t - 0.01 * t ** 2
sensor2_signal = true_signal + 40 * np.random.randn(len(true_signal))

fuser = ComplementaryFilter()
fuser.optimize_alpha(sensor1_signal, sensor2_signal, true_signal)
fused_signal = fuser.fuse_multiple(sensor1_signal, sensor2_signal)

plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(t, true_signal, "b", label="True")
plt.plot(t, sensor1_signal, "g", label="Sensor 1")
plt.plot(t, sensor2_signal, "y", label="Sensor 2")
plt.plot(t, fused_signal, "r", label="Fusion")
plt.grid()
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(t, true_signal, "b", label="True")
plt.plot(t, fused_signal, "r", label="Fusion")
plt.grid()
plt.legend()
plt.show()
