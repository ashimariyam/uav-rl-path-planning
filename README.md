# UAV RL Path Planning

This project implements a reinforcement learning-based path planning system for UAVs (Unmanned Aerial Vehicles) navigating in 3D space with wind disturbances. The system uses the TD3 (Twin Delayed DDPG) algorithm to learn energy-efficient paths while avoiding obstacles and handling turbulent wind conditions.

## Project Overview

The project consists of three main components:

1. **Custom Gym Environment (`uav_env.py`)**
   - 3D navigation space (20x20x20 grid)
   - Continuous action space for 3D movement
   - Wind field simulation with turbulence
   - Obstacle avoidance
   - Energy consumption modeling
   - Real-time visualization

2. **Training Script (`train_td3.py`)**
   - TD3 algorithm implementation
   - Experience replay and target networks
   - Hyperparameter optimization
   - Training progress tracking with wandb
   - Model checkpointing

3. **Testing Script (`test_td3.py`)**
   - Model evaluation
   - Path visualization
   - Performance metrics calculation
   - Energy efficiency analysis

## Features

- ✅ 3D path planning in continuous space
- ✅ Wind disturbance modeling with turbulence
- ✅ Energy-efficient path optimization
- ✅ Obstacle avoidance
- ✅ Real-time visualization
- ✅ Performance metrics tracking
- ✅ Experiment logging with wandb
- ✅ Automatic model selection for testing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/uav-rl-path-planning.git
cd uav-rl-path-planning
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install wandb  # For experiment tracking
```

## Usage

### Training

To train the agent:
```bash
python train_td3.py
```

The training script will:
- Create a new run directory in `logs/`
- Initialize wandb for experiment tracking
- Train the TD3 agent
- Save checkpoints and the best model
- Log training metrics

### Testing

To test a trained model:
```bash
python test_td3.py
```

The testing script will:
- Find the latest trained model
- Run evaluation episodes
- Generate path visualizations
- Calculate performance metrics
- Save results and plots

## Project Structure

```
uav-rl-path-planning/
├── uav_env.py           # Custom Gym environment
├── train_td3.py         # Training script
├── test_td3.py          # Testing script
├── requirements.txt     # Project dependencies
├── README.md           # This file
├── logs/               # Training logs and models
│   ├── run_*/          # Individual training runs
│   └── best_model/     # Best performing model
└── test_results/       # Testing results and plots
```

## Environment Details

### State Space
- UAV position (x, y, z)
- Wind vector at current position (wx, wy, wz)

### Action Space
- 3D movement vector [-1, 1] for each dimension

### Reward Function
- Distance to target penalty
- Energy consumption penalty
- Progress reward
- Target reached bonus

### Wind Model
- Time-varying turbulent wind field
- Multiple frequency components
- Spatial smoothing
- Random noise

## Results

The trained agent learns to:
- Navigate efficiently to the target
- Minimize energy consumption
- Handle wind disturbances
- Avoid obstacles
- Find optimal paths

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project is inspired by research in UAV path planning and reinforcement learning
- Uses Stable-Baselines3 for the TD3 implementation
- Uses Gymnasium for the environment interface
- Uses wandb for experiment tracking


### Reference

Chen, S., Mo, Y., Wu, X., Xiao, J., & Liu, Q. (2024). Reinforcement Learning-Based Energy-Saving Path Planning for UAVs in Turbulent Wind. Electronics, 13(16), 3190. https://doi.org/10.3390/electronics13163190


