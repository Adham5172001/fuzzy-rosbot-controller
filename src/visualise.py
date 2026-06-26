"""
Visualisation for Fuzzy ROSbot Controller
Author: Adham Aboulkheir | University of Essex
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def plot_membership_functions(mfs: dict, x_range: np.ndarray,
                               title: str, xlabel: str,
                               save_path: str = None):
    """Plot all membership functions for a fuzzy variable."""
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    
    colors = ["#ff7b72", "#f4a261", "#3fb950", "#58a6ff", "#d2a8ff"]
    for (name, mf), color in zip(mfs.items(), colors):
        y = mf.compute(x_range)
        ax.fill_between(x_range, y, alpha=0.2, color=color)
        ax.plot(x_range, y, label=name, color=color, linewidth=2.5)
    
    ax.set_xlabel(xlabel, color="white", fontsize=11)
    ax.set_ylabel("Membership Degree", color="white", fontsize=11)
    ax.set_title(title, color="white", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#161b22", labelcolor="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.grid(alpha=0.3, color="#21262d")
    ax.set_ylim(-0.05, 1.1)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved: {save_path}")
    
    plt.close(fig)
    return fig


def plot_navigation_trajectory(positions: list, commands: list,
                                obstacles: list = None, save_path: str = None):
    """Plot the robot navigation trajectory."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d1117")
    
    for ax in axes:
        ax.set_facecolor("#161b22")
    
    # Trajectory plot
    if positions:
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        axes[0].plot(xs, ys, color="#00c9b1", linewidth=2, label="Robot path")
        axes[0].scatter(xs[0], ys[0], color="#3fb950", s=100, zorder=5, label="Start")
        axes[0].scatter(xs[-1], ys[-1], color="#ff7b72", s=100, zorder=5, label="End")
    
    if obstacles:
        for obs in obstacles:
            circle = plt.Circle(obs[:2], obs[2] if len(obs) > 2 else 5,
                                color="#ff7b72", alpha=0.4)
            axes[0].add_patch(circle)
    
    axes[0].set_title("Navigation Trajectory", color="white", fontsize=11)
    axes[0].set_xlabel("X (cm)", color="white")
    axes[0].set_ylabel("Y (cm)", color="white")
    axes[0].legend(facecolor="#161b22", labelcolor="white", fontsize=8)
    axes[0].tick_params(colors="white")
    axes[0].grid(alpha=0.3, color="#21262d")
    
    # Command history
    if commands:
        t = np.arange(len(commands))
        axes[1].plot(t, commands, color="#00c9b1", linewidth=1.5)
        axes[1].axhline(y=0, color="white", linestyle="--", alpha=0.3)
        axes[1].fill_between(t, commands, alpha=0.15, color="#00c9b1")
    
    axes[1].set_title("Motor Command History", color="white", fontsize=11)
    axes[1].set_xlabel("Time Step", color="white")
    axes[1].set_ylabel("Command (-1=stop, +1=fast)", color="white")
    axes[1].tick_params(colors="white")
    axes[1].grid(alpha=0.3, color="#21262d")
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved: {save_path}")
    
    plt.close(fig)
    return fig


if __name__ == "__main__":
    from membership_functions import build_obstacle_mfs, build_speed_mfs, build_command_mfs
    
    os.makedirs("outputs", exist_ok=True)
    
    # Plot all membership functions
    plot_membership_functions(
        build_obstacle_mfs(), np.linspace(0, 200, 500),
        "Obstacle Distance Membership Functions", "Distance (cm)",
        "outputs/obstacle_mfs.png"
    )
    plot_membership_functions(
        build_speed_mfs(), np.linspace(0, 1, 200),
        "Robot Speed Membership Functions", "Speed (normalised)",
        "outputs/speed_mfs.png"
    )
    plot_membership_functions(
        build_command_mfs(), np.linspace(-1, 1, 200),
        "Motor Command Membership Functions", "Command",
        "outputs/command_mfs.png"
    )
    print("All membership function plots saved.")
