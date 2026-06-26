#!/usr/bin/env python3
"""
Fuzzy Logic Controller for ROSbot Navigation
Author: Adham Aboulkheir | University of Essex
"""
import numpy as np


class FuzzyMF:
    """Triangular membership function."""
    def __init__(self, a, b, c):
        self.a, self.b, self.c = a, b, c

    def compute(self, x):
        x = np.atleast_1d(np.array(x, dtype=float))
        left  = (x - self.a) / (self.b - self.a + 1e-9)
        right = (self.c - x) / (self.c - self.b + 1e-9)
        return np.clip(np.minimum(left, right), 0, 1)


class FuzzyNavigationController:
    """
    Type-1 Fuzzy Logic Controller for obstacle avoidance.
    Inputs:  obstacle_distance (cm), robot_speed (0-1)
    Output:  motor_command (-1=stop, +1=full speed)
    """

    def __init__(self):
        self.dist_near   = FuzzyMF(0,   0,   50)
        self.dist_medium = FuzzyMF(30,  80,  130)
        self.dist_far    = FuzzyMF(100, 200, 200)
        self.speed_low   = FuzzyMF(0,   0,   0.5)
        self.speed_high  = FuzzyMF(0.3, 1.0, 1.0)
        self.cmd_stop     = FuzzyMF(-1.0, -1.0, -0.2)
        self.cmd_slow     = FuzzyMF(-0.3,  0.2,  0.5)
        self.cmd_moderate = FuzzyMF( 0.3,  0.6,  0.8)
        self.cmd_fast     = FuzzyMF( 0.6,  1.0,  1.0)

    def compute(self, distance_cm: float, speed: float) -> float:
        """Compute motor command using Mamdani fuzzy inference."""
        d = np.clip(distance_cm, 0, 200)
        s = np.clip(speed, 0, 1)
        near   = float(self.dist_near.compute(d))
        medium = float(self.dist_medium.compute(d))
        far    = float(self.dist_far.compute(d))
        low    = float(self.speed_low.compute(s))
        high   = float(self.speed_high.compute(s))
        r_stop     = min(near, high)
        r_slow_1   = min(near, low)
        r_slow_2   = min(medium, high)
        r_moderate = min(medium, low)
        r_fast     = far
        x_out = np.linspace(-1, 1, 200)
        agg = np.zeros(200)
        agg = np.maximum(agg, r_stop     * self.cmd_stop.compute(x_out))
        agg = np.maximum(agg, max(r_slow_1, r_slow_2) * self.cmd_slow.compute(x_out))
        agg = np.maximum(agg, r_moderate * self.cmd_moderate.compute(x_out))
        agg = np.maximum(agg, r_fast     * self.cmd_fast.compute(x_out))
        denom = np.sum(agg)
        if denom < 1e-9:
            return 0.0
        return float(np.sum(x_out * agg) / denom)


if __name__ == "__main__":
    print("Fuzzy ROSbot Controller Demo")
    ctrl = FuzzyNavigationController()
    test_cases = [(20, 0.9), (80, 0.5), (150, 0.8), (30, 0.3)]
    print("\nDistance(cm) | Speed | Motor Command")
    print("-" * 40)
    for dist, spd in test_cases:
        cmd = ctrl.compute(dist, spd)
        action = "STOP" if cmd < -0.3 else "SLOW" if cmd < 0.3 else "MODERATE" if cmd < 0.7 else "FAST"
        print(f"  {dist:5}cm   |  {spd:.1f}  | {cmd:+.3f}  ({action})")
