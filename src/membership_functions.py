"""
Fuzzy Membership Functions
Author: Adham Aboulkheir | University of Essex
"""
import numpy as np
from typing import Union


class TriangularMF:
    """Triangular membership function: /\\"""
    def __init__(self, a: float, b: float, c: float, name: str = ""):
        assert a <= b <= c, f"Parameters must satisfy a <= b <= c, got {a}, {b}, {c}"
        self.a, self.b, self.c = a, b, c
        self.name = name
    
    def compute(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        x = np.atleast_1d(np.array(x, dtype=float))
        left  = (x - self.a) / (self.b - self.a + 1e-9)
        right = (self.c - x) / (self.c - self.b + 1e-9)
        return np.clip(np.minimum(left, right), 0, 1)
    
    def centroid(self) -> float:
        return self.b
    
    def __repr__(self):
        return f"TriangularMF(name={self.name!r}, a={self.a}, b={self.b}, c={self.c})"


class TrapezoidalMF:
    """Trapezoidal membership function: /‾\\"""
    def __init__(self, a: float, b: float, c: float, d: float, name: str = ""):
        self.a, self.b, self.c, self.d = a, b, c, d
        self.name = name
    
    def compute(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        x = np.atleast_1d(np.array(x, dtype=float))
        left  = (x - self.a) / (self.b - self.a + 1e-9)
        right = (self.d - x) / (self.d - self.c + 1e-9)
        return np.clip(np.minimum(np.minimum(left, 1.0), right), 0, 1)


class GaussianMF:
    """Gaussian membership function: bell curve."""
    def __init__(self, mean: float, sigma: float, name: str = ""):
        self.mean = mean
        self.sigma = sigma
        self.name = name
    
    def compute(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        x = np.atleast_1d(np.array(x, dtype=float))
        return np.exp(-0.5 * ((x - self.mean) / (self.sigma + 1e-9)) ** 2)


def build_obstacle_mfs() -> dict:
    """Build membership functions for obstacle distance (0-200 cm)."""
    return {
        "NEAR":   TriangularMF(0,   0,   50,  "NEAR"),
        "MEDIUM": TriangularMF(30,  80,  130, "MEDIUM"),
        "FAR":    TriangularMF(100, 200, 200, "FAR"),
    }


def build_speed_mfs() -> dict:
    """Build membership functions for robot speed (0-1 normalised)."""
    return {
        "LOW":  TriangularMF(0.0, 0.0, 0.5, "LOW"),
        "HIGH": TriangularMF(0.3, 1.0, 1.0, "HIGH"),
    }


def build_command_mfs() -> dict:
    """Build membership functions for motor command (-1 to +1)."""
    return {
        "STOP":     TriangularMF(-1.0, -1.0, -0.2, "STOP"),
        "SLOW":     TriangularMF(-0.3,  0.2,  0.5, "SLOW"),
        "MODERATE": TriangularMF( 0.3,  0.6,  0.8, "MODERATE"),
        "FAST":     TriangularMF( 0.6,  1.0,  1.0, "FAST"),
    }


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    print("Membership Functions Demo")
    x = np.linspace(0, 200, 500)
    mfs = build_obstacle_mfs()
    
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    colors = ["#ff7b72", "#f4a261", "#3fb950"]
    for (name, mf), color in zip(mfs.items(), colors):
        ax.plot(x, mf.compute(x), label=name, color=color, linewidth=2)
    ax.set_xlabel("Obstacle Distance (cm)", color="white")
    ax.set_ylabel("Membership Degree", color="white")
    ax.set_title("Obstacle Distance Membership Functions", color="white")
    ax.legend(facecolor="#161b22", labelcolor="white")
    ax.tick_params(colors="white")
    ax.grid(alpha=0.3, color="#21262d")
    plt.tight_layout()
    plt.savefig("outputs/membership_functions.png", dpi=150, bbox_inches="tight")
    print("  Saved: outputs/membership_functions.png")
    
    # Test values
    for dist in [10, 50, 100, 180]:
        print(f"  Distance={dist}cm: NEAR={mfs['NEAR'].compute(dist)[0]:.2f}, "
              f"MEDIUM={mfs['MEDIUM'].compute(dist)[0]:.2f}, "
              f"FAR={mfs['FAR'].compute(dist)[0]:.2f}")
