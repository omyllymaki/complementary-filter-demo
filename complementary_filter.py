from typing import List, Union

import numpy as np

Signal = Union[List[float], np.ndarray]


class ComplementaryFilter:
    """
    Complementary filter fuses signals 1 and 2.

    fused_signal = (1 - alpha) * (fused_signal + delta_signal_from_signal1) + alpha * signal2

    It is assumed that
    - signal1 has high bias but low noise
    - signal2 has low bias but high noise

    Parameter alpha [0,1] will determine how much filter takes signal2 to new estimate. Small alpha will increase bias,
    high alpha will increase noise, and optimal alpha balances these two.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.first_round = True
        self.fused_signal = None
        self.prev_signal_1 = None

    def fuse(self,
             signal1: float,
             signal2: float,
             alpha: float = None) -> float:
        """Fuse individual signals."""

        if alpha is None:
            alpha = self.alpha

        if self.first_round:
            self.fused_signal = signal2
            self.first_round = False
        else:
            delta_signal = signal1 - self.prev_signal_1

            self.fused_signal = (1 - alpha) * (self.fused_signal + delta_signal) + alpha * signal2

        self.prev_signal_1 = signal1

        return self.fused_signal

    def fuse_multiple(self,
                      signals1: Signal,
                      signals2: Signal,
                      alpha: float = None) -> Signal:
        """Fuse multiple signals."""
        self.reset()
        return [self.fuse(s1, s2, alpha) for s1, s2 in zip(signals1, signals2)]

    def optimize_alpha(self,
                       signals1: Signal,
                       signals2: Signal,
                       true_signals: Signal,
                       alpha_candidates: Signal = None) -> None:
        """Find and set optimal alpha based on difference between fused and true signal."""

        if alpha_candidates is None:
            alpha_candidates = np.arange(0, 1, 0.01)

        min_error = np.inf
        optimal_alpha = None

        for alpha in alpha_candidates:
            fused_signal = np.array(self.fuse_multiple(signals1, signals2, alpha=alpha))
            error = np.mean(np.abs(fused_signal - true_signals))
            if error < min_error:
                optimal_alpha = alpha
                min_error = error

        self.alpha = optimal_alpha

    def reset(self) -> None:
        """Reset filter state"""
        self.first_round = True
