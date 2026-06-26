"""
Unit Tests for Fuzzy ROSbot Controller
Author: Adham Aboulkheir | University of Essex
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fuzzy_controller import FuzzyNavigationController
from src.membership_functions import TriangularMF, TrapezoidalMF, GaussianMF


def test_triangular_mf_peak():
    """Triangular MF should return 1.0 at peak."""
    mf = TriangularMF(0, 5, 10)
    assert abs(mf.compute(5)[0] - 1.0) < 1e-6
    assert mf.compute(0)[0] == 0.0
    assert mf.compute(10)[0] == 0.0
    print("  ✓ test_triangular_mf_peak")


def test_triangular_mf_range():
    """Triangular MF values should be in [0, 1]."""
    mf = TriangularMF(0, 50, 100)
    x = np.linspace(-10, 110, 200)
    values = mf.compute(x)
    assert values.min() >= 0.0
    assert values.max() <= 1.0
    print("  ✓ test_triangular_mf_range")


def test_controller_stop_near_obstacle():
    """Controller should output near-stop when obstacle is very close."""
    ctrl = FuzzyNavigationController()
    cmd = ctrl.compute(distance_cm=15, speed=0.9)
    assert cmd < 0.0, f"Expected stop command, got {cmd}"
    print(f"  ✓ test_controller_stop_near_obstacle (cmd={cmd:.3f})")


def test_controller_fast_far_obstacle():
    """Controller should output fast when obstacle is far."""
    ctrl = FuzzyNavigationController()
    cmd = ctrl.compute(distance_cm=180, speed=0.5)
    assert cmd > 0.5, f"Expected fast command, got {cmd}"
    print(f"  ✓ test_controller_fast_far_obstacle (cmd={cmd:.3f})")


def test_controller_output_range():
    """Controller output should always be in [-1, 1]."""
    ctrl = FuzzyNavigationController()
    for dist in np.linspace(0, 200, 20):
        for speed in np.linspace(0, 1, 10):
            cmd = ctrl.compute(dist, speed)
            assert -1.0 <= cmd <= 1.0, f"Out of range: {cmd} for dist={dist}, speed={speed}"
    print("  ✓ test_controller_output_range")


def test_controller_monotonic_distance():
    """As distance increases, command should generally increase."""
    ctrl = FuzzyNavigationController()
    distances = [10, 50, 100, 150, 190]
    commands = [ctrl.compute(d, 0.5) for d in distances]
    # Commands should be non-decreasing overall
    assert commands[-1] > commands[0], f"Expected increasing commands: {commands}"
    print(f"  ✓ test_controller_monotonic_distance")


if __name__ == "__main__":
    print("Running Fuzzy ROSbot Controller Tests")
    print("=" * 40)
    test_triangular_mf_peak()
    test_triangular_mf_range()
    test_controller_stop_near_obstacle()
    test_controller_fast_far_obstacle()
    test_controller_output_range()
    test_controller_monotonic_distance()
    print("\n✓ All tests passed!")
