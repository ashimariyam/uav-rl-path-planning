# uav-rl-path-planning
This project implements a simplified version of the ESPP-RL (Energy-Saving Path Planning using Reinforcement Learning) algorithm inspired by the paper "Reinforcement Learning-Based Energy-Saving Path Planning for UAVs in Turbulent Wind" (Chen et al., 2024).

Using a custom Gym environment, a UAV learns to navigate in a simulated 3D space with wind disturbances, aiming to reach a target location while minimizing energy consumption and avoiding collisions. The agent is trained using the TD3 (Twin Delayed DDPG) algorithm from Stable-Baselines3.
