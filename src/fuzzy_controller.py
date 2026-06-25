#!/usr/bin/env python3
"""
Fuzzy Logic Controller for ROSbot Navigation
Author: Adham Aboulkheir
University of Essex
"""
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyNavigationController:
    """
    Type-1 Fuzzy Logic Controller for obstacle avoidance and navigation.
    Uses ultrasonic and IR sensor inputs to generate smooth motor commands.
    """

    def __init__(self):
        self._build_fuzzy_system()

    def _build_fuzzy_system(self):
        """Define fuzzy variables, membership functions, and rules."""
        # Input variables
        self.obstacle_dist = ctrl.Antecedent(np.arange(0, 201, 1), 'obstacle_distance')
        self.robot_speed   = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'robot_speed')

        # Output variable
        self.motor_cmd = ctrl.Consequent(np.arange(-1, 1.01, 0.01), 'motor_command')

        # Membership functions — obstacle distance (cm)
        self.obstacle_dist['near']   = fuzz.trimf(self.obstacle_dist.universe, [0, 0, 50])
        self.obstacle_dist['medium'] = fuzz.trimf(self.obstacle_dist.universe, [30, 80, 130])
        self.obstacle_dist['far']    = fuzz.trimf(self.obstacle_dist.universe, [100, 200, 200])

        # Membership functions — robot speed (m/s normalised 0-1)
        self.robot_speed['low']  = fuzz.trimf(self.robot_speed.universe, [0, 0, 0.5])
        self.robot_speed['high'] = fuzz.trimf(self.robot_speed.universe, [0.3, 1, 1])

        # Membership functions — motor command (-1 = stop/reverse, 1 = full forward)
        self.motor_cmd['stop']     = fuzz.trimf(self.motor_cmd.universe, [-1, -1, -0.2])
        self.motor_cmd['slow']     = fuzz.trimf(self.motor_cmd.universe, [-0.3, 0.2, 0.5])
        self.motor_cmd['moderate'] = fuzz.trimf(self.motor_cmd.universe, [0.3, 0.6, 0.8])
        self.motor_cmd['fast']     = fuzz.trimf(self.motor_cmd.universe, [0.6, 1, 1])

        # Fuzzy rules
        rules = [
            ctrl.Rule(self.obstacle_dist['near']   & self.robot_speed['high'], self.motor_cmd['stop']),
            ctrl.Rule(self.obstacle_dist['near']   & self.robot_speed['low'],  self.motor_cmd['slow']),
            ctrl.Rule(self.obstacle_dist['medium'] & self.robot_speed['high'], self.motor_cmd['slow']),
            ctrl.Rule(self.obstacle_dist['medium'] & self.robot_speed['low'],  self.motor_cmd['moderate']),
            ctrl.Rule(self.obstacle_dist['far'],                               self.motor_cmd['fast']),
        ]

        self.control_system = ctrl.ControlSystem(rules)
        self.simulation     = ctrl.ControlSystemSimulation(self.control_system)

    def compute(self, obstacle_distance_cm: float, current_speed: float) -> float:
        """
        Compute motor command from sensor inputs.

        Args:
            obstacle_distance_cm: Distance to nearest obstacle in cm
            current_speed: Current robot speed (normalised 0-1)

        Returns:
            Motor command in range [-1, 1]
        """
        self.simulation.input['obstacle_distance'] = np.clip(obstacle_distance_cm, 0, 200)
        self.simulation.input['robot_speed']       = np.clip(current_speed, 0, 1)
        self.simulation.compute()
        return float(self.simulation.output['motor_command'])


if __name__ == '__main__':
    controller = FuzzyNavigationController()
    # Example: obstacle 30cm away, moving at 80% speed
    cmd = controller.compute(obstacle_distance_cm=30, current_speed=0.8)
    print(f"Motor command: {cmd:.3f}")  # Expected: near stop
