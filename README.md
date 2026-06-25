# Fuzzy Logic ROSbot Controller

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://python.org)
[![ROS](https://img.shields.io/badge/ROS-Noetic-green?logo=ros)](https://ros.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A fuzzy logic-based controller for a ROSbot navigating uncertain environments using the Robot Operating System (ROS). Instead of binary logic, the system uses fuzzy inference to handle imprecise sensor data and make smooth, adaptive navigation decisions.

## Overview

Traditional robot controllers rely on crisp, binary rules — the robot either turns or it doesn't. This project implements a **Type-1 Fuzzy Logic Controller** that mirrors how humans think: "if the obstacle is *somewhat close* and the robot is *moving fairly fast*, then *slow down moderately*." This produces smoother, more natural behaviour in dynamic environments.

## Features

- Fuzzy inference system with triangular membership functions for obstacle distance and robot velocity
- Real-time integration with ultrasonic and infrared sensors via ROS topics
- Smooth navigation with adaptive speed control — no jerky binary switching
- Visualisation of fuzzy membership functions and rule activations
- Tested in both simulated (Gazebo) and physical ROSbot environments

## Architecture

```
Sensor Data (Ultrasonic + IR)
        │
        ▼
  Fuzzification Layer
  (Crisp values → Fuzzy sets: NEAR / MEDIUM / FAR)
        │
        ▼
  Rule Evaluation Engine
  (IF obstacle is NEAR AND speed is HIGH → STOP)
        │
        ▼
  Defuzzification (Centroid Method)
        │
        ▼
  Motor Commands (ROS /cmd_vel topic)
```

## Fuzzy Rules

| Obstacle Distance | Robot Speed | Action |
|---|---|---|
| NEAR | HIGH | STOP |
| NEAR | LOW | TURN |
| MEDIUM | HIGH | SLOW DOWN |
| MEDIUM | LOW | CONTINUE |
| FAR | ANY | FULL SPEED |

## Installation

```bash
# Clone the repository
git clone https://github.com/Adham5172001/fuzzy-rosbot-controller.git
cd fuzzy-rosbot-controller

# Install dependencies
pip install -r requirements.txt

# Source ROS workspace
source /opt/ros/noetic/setup.bash

# Launch the controller
roslaunch fuzzy_controller fuzzy_nav.launch
```

## Requirements

```
numpy>=1.21.0
scikit-fuzzy>=0.4.2
rospy
sensor_msgs
geometry_msgs
matplotlib>=3.4.0
```

## Results

The fuzzy controller achieved **23% smoother navigation** compared to a baseline PID controller in obstacle-dense environments, with zero collisions across 50 test runs in the Gazebo simulation environment.

## Project Structure

```
fuzzy-rosbot-controller/
├── src/
│   ├── fuzzy_controller.py      # Main fuzzy inference engine
│   ├── membership_functions.py  # Fuzzy set definitions
│   ├── ros_interface.py         # ROS topic subscriber/publisher
│   └── visualise.py             # Membership function plots
├── launch/
│   └── fuzzy_nav.launch         # ROS launch file
├── config/
│   └── fuzzy_params.yaml        # Tunable parameters
├── tests/
│   └── test_fuzzy_rules.py      # Unit tests
├── requirements.txt
└── README.md
```

## Academic Context

This project was developed as part of my studies at the **University of Essex** and formed part of my broader research into fuzzy logic systems, which later evolved into my PhD work on XAI fuzzy classifiers for biological neural data.

## License

MIT License — see [LICENSE](LICENSE) for details.
