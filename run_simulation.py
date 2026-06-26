"""
Fuzzy ROSbot Navigation Simulation
Author: Adham Aboulkheir | University of Essex
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.fuzzy_controller import FuzzyNavigationController
from src.visualise import plot_membership_functions, plot_navigation_trajectory
from src.membership_functions import build_obstacle_mfs, build_speed_mfs


class RobotSimulator:
    """Simple 2D robot navigation simulator."""
    
    def __init__(self, arena_size=(500, 500), n_obstacles=8, seed=42):
        np.random.seed(seed)
        self.arena = arena_size
        self.obstacles = [
            (np.random.uniform(50, 450), np.random.uniform(50, 450), 20)
            for _ in range(n_obstacles)
        ]
        self.controller = FuzzyNavigationController()
        self.reset()
    
    def reset(self):
        self.x = 50.0
        self.y = 50.0
        self.heading = 0.0  # radians
        self.speed = 0.5
        self.positions = [(self.x, self.y)]
        self.commands = []
        self.steps = 0
    
    def get_obstacle_distance(self) -> float:
        """Get distance to nearest obstacle."""
        min_dist = float("inf")
        for ox, oy, radius in self.obstacles:
            dist = np.sqrt((self.x - ox)**2 + (self.y - oy)**2) - radius
            min_dist = min(min_dist, max(0, dist))
        return min(min_dist, 200.0)
    
    def step(self) -> bool:
        """Execute one simulation step. Returns True if still running."""
        dist = self.get_obstacle_distance()
        cmd = self.controller.compute(dist, self.speed)
        
        # Update speed
        self.speed = np.clip(self.speed + cmd * 0.1, 0.0, 1.0)
        
        # Turn to avoid obstacles
        if dist < 50:
            self.heading += np.random.uniform(-0.3, 0.3)
        
        # Move
        step_size = self.speed * 5
        self.x += step_size * np.cos(self.heading)
        self.y += step_size * np.sin(self.heading)
        
        # Boundary handling
        self.x = np.clip(self.x, 0, self.arena[0])
        self.y = np.clip(self.y, 0, self.arena[1])
        
        self.positions.append((self.x, self.y))
        self.commands.append(cmd)
        self.steps += 1
        
        return self.steps < 500
    
    def run(self, verbose=True) -> dict:
        """Run full simulation."""
        self.reset()
        collisions = 0
        
        while self.step():
            dist = self.get_obstacle_distance()
            if dist < 5:
                collisions += 1
        
        return {
            "steps": self.steps,
            "collisions": collisions,
            "final_position": (self.x, self.y),
            "mean_speed": np.mean([abs(c) for c in self.commands]),
            "command_smoothness": 1 - np.std(np.diff(self.commands)) if len(self.commands) > 1 else 0
        }


def main():
    print("=" * 55)
    print("FUZZY ROSBOT NAVIGATION SIMULATION")
    print("Author: Adham Aboulkheir | University of Essex")
    print("=" * 55)
    
    os.makedirs("outputs", exist_ok=True)
    
    # Plot membership functions
    print("\n[1/3] Plotting membership functions...")
    plot_membership_functions(
        build_obstacle_mfs(), np.linspace(0, 200, 500),
        "Obstacle Distance Membership Functions", "Distance (cm)",
        "outputs/obstacle_mfs.png"
    )
    
    # Run simulation
    print("\n[2/3] Running navigation simulation...")
    sim = RobotSimulator(n_obstacles=8, seed=42)
    results = sim.run()
    
    print(f"  Steps: {results['steps']}")
    print(f"  Collisions: {results['collisions']}")
    print(f"  Final position: ({results['final_position'][0]:.1f}, {results['final_position'][1]:.1f})")
    print(f"  Command smoothness: {results['command_smoothness']:.3f}")
    
    # Plot trajectory
    print("\n[3/3] Plotting navigation trajectory...")
    plot_navigation_trajectory(
        sim.positions, sim.commands, sim.obstacles,
        "outputs/navigation_trajectory.png"
    )
    
    # Controller test cases
    print("\nController Test Cases:")
    ctrl = FuzzyNavigationController()
    test_cases = [
        (15, 0.9, "Emergency: obstacle 15cm, speed 0.9"),
        (80, 0.5, "Normal: obstacle 80cm, speed 0.5"),
        (180, 0.8, "Clear: obstacle 180cm, speed 0.8"),
        (30, 0.2, "Caution: obstacle 30cm, speed 0.2"),
    ]
    print(f"  {'Scenario':<45} {'Command':>8} {'Action'}")
    print("  " + "-" * 70)
    for dist, spd, desc in test_cases:
        cmd = ctrl.compute(dist, spd)
        action = "STOP" if cmd < -0.2 else "SLOW" if cmd < 0.3 else "MODERATE" if cmd < 0.7 else "FAST"
        print(f"  {desc:<45} {cmd:>+8.3f}  {action}")
    
    print("\n✓ Simulation complete! Outputs saved to outputs/")


if __name__ == "__main__":
    main()
