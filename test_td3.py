import os
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from uav_env import UAVEnv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
import logging
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('testing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def plot_episode_path(
    positions: List[np.ndarray],
    target_pos: np.ndarray,
    obstacles: List[Dict[str, np.ndarray]],
    wind_field: np.ndarray,
    save_path: str
) -> None:
    """Plot the UAV's path in 3D space with obstacles and wind field.
    
    Args:
        positions: List of UAV positions during the episode
        target_pos: Target position
        obstacles: List of obstacle dictionaries
        wind_field: Wind field array
        save_path: Path to save the plot
    """
    try:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot UAV path
        positions = np.array(positions)
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'b-', label='UAV Path', linewidth=2)
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
                   'go', label='Start', s=100)
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
                   'ro', label='End', s=100)
        
        # Plot target
        ax.scatter(target_pos[0], target_pos[1], target_pos[2],
                   'g*', label='Target', s=200)
        
        # Plot obstacles
        for obstacle in obstacles:
            min_pos = obstacle['min']
            max_pos = obstacle['max']
            
            # Create cube vertices
            vertices = np.array([
                [min_pos[0], min_pos[1], min_pos[2]],
                [max_pos[0], min_pos[1], min_pos[2]],
                [max_pos[0], max_pos[1], min_pos[2]],
                [min_pos[0], max_pos[1], min_pos[2]],
                [min_pos[0], min_pos[1], max_pos[2]],
                [max_pos[0], min_pos[1], max_pos[2]],
                [max_pos[0], max_pos[1], max_pos[2]],
                [min_pos[0], max_pos[1], max_pos[2]]
            ])
            
            # Define faces
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
                [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
                [vertices[1], vertices[2], vertices[6], vertices[5]]   # right
            ]
            
            # Plot each face
            for face in faces:
                face = np.array(face)
                x, y, z = face[:, 0], face[:, 1], face[:, 2]
                ax.plot_trisurf(x, y, z, color='red', alpha=0.2)
        
        # Plot wind vectors (subsampled)
        step = 4  # Plot every 4th point
        for i in range(0, wind_field.shape[0], step):
            for j in range(0, wind_field.shape[1], step):
                for k in range(0, wind_field.shape[2], step):
                    wind = wind_field[i, j, k]
                    if np.linalg.norm(wind) > 0.1:  # Only plot significant wind
                        ax.quiver(i, j, k, 
                                wind[0], wind[1], wind[2],
                                color='gray', alpha=0.3,
                                length=0.5, normalize=True)
        
        # Set plot limits and labels
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.set_zlim(0, 20)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('UAV Path Planning with Wind Disturbance')
        
        # Save the plot
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        logging.error(f"Failed to plot episode path: {e}")
        raise

def find_latest_model() -> Tuple[str, str]:
    """Find the latest model and environment statistics files."""
    try:
        # Look for best model first
        best_model_path = os.path.join("logs", "best_model", "best_model")
        if os.path.exists(best_model_path + ".zip"):
            return best_model_path, os.path.join("logs", "vec_normalize.pkl")
        
        # If no best model, look for latest run
        log_dir = Path("logs")
        if not log_dir.exists():
            raise FileNotFoundError("No logs directory found")
        
        # Get all run directories
        run_dirs = [d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        if not run_dirs:
            raise FileNotFoundError("No run directories found")
        
        # Get the latest run
        latest_run = max(run_dirs, key=lambda d: d.stat().st_mtime)
        model_path = os.path.join(latest_run, "td3_uav_final")
        env_stats_path = os.path.join(latest_run, "vec_normalize.pkl")
        
        if not os.path.exists(model_path + ".zip"):
            raise FileNotFoundError(f"No model found at {model_path}")
        
        return model_path, env_stats_path
    except Exception as e:
        logging.error(f"Failed to find model files: {e}")
        raise

def test_model(
    model_path: str,
    env_stats_path: str,
    n_episodes: int = 10,
    deterministic: bool = True,
    render: bool = True,
    save_plots: bool = True
) -> Dict[str, Any]:
    """Test a trained model.
    
    Args:
        model_path (str): Path to the trained model.
        env_stats_path (str): Path to the saved VecNormalize statistics.
        n_episodes (int, optional): Number of episodes to test. Defaults to 10.
        deterministic (bool, optional): Whether to use deterministic actions. Defaults to True.
        render (bool, optional): Whether to render the environment. Defaults to True.
        save_plots (bool, optional): Whether to save path plots. Defaults to True.
        
    Returns:
        Dict[str, Any]: Test results and metrics.
    """
    try:
        # Create and wrap the environment
        env = UAVEnv(render_mode="human" if render else None)
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(env_stats_path, env)
        env.training = False  # Do not update stats at test time
        env.norm_reward = False  # Reward normalization is not needed at test time
        
        # Load the trained model
        model = TD3.load(model_path)
        
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"test_results/run_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        # Run test episodes
        episode_rewards = []
        episode_lengths = []
        episode_distances = []
        episode_energies = []
        episode_paths = []
        
        for episode in range(n_episodes):
            obs = env.reset()[0]
            episode_reward = 0
            step = 0
            done = False
            positions = [env.envs[0].current_pos.copy()]
            
            logging.info(f"\nEpisode {episode + 1}:")
            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = env.step(action)
                done = done.any()  # VecEnv returns done as a boolean array
                
                episode_reward += reward[0]  # VecEnv returns rewards as an array
                step += 1
                positions.append(env.envs[0].current_pos.copy())
                
                if render:
                    env.render()
            
            # Get final metrics
            final_pos = env.envs[0].current_pos
            target_pos = env.envs[0].target_pos
            final_distance = np.linalg.norm(final_pos - target_pos)
            total_energy = info[0].get('total_energy_consumed', 0)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(step)
            episode_distances.append(final_distance)
            episode_energies.append(total_energy)
            episode_paths.append(positions)
            
            # Save episode plot
            if save_plots:
                plot_path = os.path.join(results_dir, f"episode_{episode+1}_path.png")
                plot_episode_path(
                    positions,
                    target_pos,
                    env.envs[0].obstacles,
                    env.envs[0].wind,
                    plot_path
                )
            
            logging.info(f"  Steps: {step}")
            logging.info(f"  Total reward: {episode_reward:.2f}")
            logging.info(f"  Final position: {final_pos}")
            logging.info(f"  Distance to target: {final_distance:.2f}")
            logging.info(f"  Total energy consumed: {total_energy:.2f}")
            logging.info("-" * 50)
        
        # Calculate summary statistics
        results = {
            "average_episode_length": float(np.mean(episode_lengths)),
            "std_episode_length": float(np.std(episode_lengths)),
            "average_episode_reward": float(np.mean(episode_rewards)),
            "std_episode_reward": float(np.std(episode_rewards)),
            "average_final_distance": float(np.mean(episode_distances)),
            "std_final_distance": float(np.std(episode_distances)),
            "average_energy_consumption": float(np.mean(episode_energies)),
            "std_energy_consumption": float(np.std(episode_energies)),
            "success_rate": float(np.mean([d < 1.0 for d in episode_distances])),
            "episode_paths": [p.tolist() for p in episode_paths]
        }
        
        # Save results
        with open(os.path.join(results_dir, "test_results.json"), "w") as f:
            json.dump(results, f, indent=4)
        
        # Print summary statistics
        logging.info("\nTest Summary:")
        logging.info(f"Average episode length: {results['average_episode_length']:.1f} ± {results['std_episode_length']:.1f}")
        logging.info(f"Average episode reward: {results['average_episode_reward']:.1f} ± {results['std_episode_reward']:.1f}")
        logging.info(f"Average final distance: {results['average_final_distance']:.2f} ± {results['std_final_distance']:.2f}")
        logging.info(f"Average energy consumption: {results['average_energy_consumption']:.2f} ± {results['std_energy_consumption']:.2f}")
        logging.info(f"Success rate: {results['success_rate']:.1%}")
        
        return results
    except Exception as e:
        logging.error(f"Testing failed: {e}")
        raise
    finally:
        if 'env' in locals():
            env.close()

def main():
    try:
        # Find the latest model and environment statistics
        model_path, env_stats_path = find_latest_model()
        
        # Run the test
        test_model(model_path, env_stats_path)
    except Exception as e:
        logging.error(f"Main function failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 