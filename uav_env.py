import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Dict, List, Any, Optional
import numpy.typing as npt
from dataclasses import dataclass
import warnings

@dataclass
class Obstacle:
    min_pos: npt.NDArray[np.float32]
    max_pos: npt.NDArray[np.float32]

class UAVEnv(gym.Env):
    """A custom Gymnasium environment for UAV navigation in 3D space with turbulent wind."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode: Optional[str] = None) -> None:
        """Initialize the UAV environment.
        
        Args:
            render_mode (Optional[str]): The render mode to use. Defaults to None.
        """
        super().__init__()
        
        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -2, -2, -2], dtype=np.float32),
            high=np.array([20, 20, 20, 2, 2, 2], dtype=np.float32),
            dtype=np.float32
        )
        
        # Environment parameters
        self.grid_size = 20
        self.start_pos = np.array([0., 0., 0.], dtype=np.float32)
        self.target_pos = np.array([18., 18., 18.], dtype=np.float32)
        self.max_steps = 200  # Maximum episode length
        self.target_threshold = 1.0  # Distance threshold for reaching target
        
        # Energy parameters
        self.base_energy_cost = 0.1  # Base energy cost for any movement
        self.wind_resistance_factor = 0.5  # Factor for wind resistance energy cost
        self.energy_penalty_weight = 0.2  # Weight for energy consumption in reward
        
        # Initialize wind field with more realistic turbulent components
        self.wind = np.zeros((self.grid_size, self.grid_size, self.grid_size, 3), dtype=np.float32)
        self.wind_phase = np.random.uniform(0, 2*np.pi, (self.grid_size, self.grid_size, self.grid_size, 3))
        self.wind_frequency = 0.1  # Wind change frequency
        self.wind_scale = 0.5  # Scale of wind variations
        self.wind_noise = 0.2  # Random noise in wind
        
        # Initialize obstacles (cubes in the environment)
        self.obstacles: List[Obstacle] = [
            Obstacle(
                min_pos=np.array([5, 5, 5], dtype=np.float32),
                max_pos=np.array([7, 7, 7], dtype=np.float32)
            ),
            Obstacle(
                min_pos=np.array([12, 12, 12], dtype=np.float32),
                max_pos=np.array([14, 14, 14], dtype=np.float32)
            ),
            Obstacle(
                min_pos=np.array([8, 8, 8], dtype=np.float32),
                max_pos=np.array([10, 10, 10], dtype=np.float32)
            )
        ]
        
        # Initialize state
        self.current_pos: Optional[npt.NDArray[np.float32]] = None
        self.step_count: int = 0
        self.total_energy_consumed: float = 0.0
        
        # Rendering setup
        self.render_mode = render_mode
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[Axes3D] = None
        
        # Initialize the environment
        self.reset()
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[npt.NDArray[np.float32], Dict[str, Any]]:
        """Reset the environment to initial state.
        
        Args:
            seed (Optional[int]): The seed for random number generation.
            options (Optional[Dict]): Additional options for reset.
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Initial observation and info dictionary.
        """
        super().reset(seed=seed)
        
        self.current_pos = self.start_pos.copy()
        self.step_count = 0
        self.total_energy_consumed = 0.0
        self._update_wind()
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, info
    
    def step(self, action: npt.NDArray[np.float32]) -> Tuple[npt.NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment.
        
        Args:
            action (np.ndarray): The action to take.
            
        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: 
                Observation, reward, terminated, truncated, info.
        """
        self.step_count += 1
        
        # Clip action to valid range
        action = np.clip(action, -1, 1)
        
        # Calculate new position
        new_pos = self.current_pos + action
        
        # Check if new position is within bounds
        if not self._is_within_bounds(new_pos):
            new_pos = np.clip(new_pos, 0, self.grid_size - 1)
        
        # Check for obstacle collision
        if self._check_collision(new_pos):
            new_pos = self.current_pos  # Stay in current position if collision
        
        # Update wind field
        self._update_wind()
        
        # Get wind at current position
        wind = self._get_wind_at_pos(self.current_pos)
        
        # Calculate energy cost
        movement_energy = self.base_energy_cost * np.linalg.norm(action)
        
        # Enhanced wind resistance energy cost
        wind_magnitude = np.linalg.norm(wind)
        if wind_magnitude > 0:
            wind_direction = wind / wind_magnitude
            action_against_wind = np.dot(action, wind_direction)
            wind_resistance = self.wind_resistance_factor * wind_magnitude * max(0, action_against_wind)
        else:
            wind_resistance = 0
        
        total_energy_cost = movement_energy + wind_resistance
        self.total_energy_consumed += total_energy_cost
        
        # Update position
        self.current_pos = new_pos
        
        # Enhanced reward calculation
        distance_to_target = np.linalg.norm(self.current_pos - self.target_pos)
        distance_reward = -distance_to_target
        energy_penalty = -self.energy_penalty_weight * total_energy_cost
        
        # Progress reward (encourages moving towards target)
        progress = np.linalg.norm(self.current_pos - self.start_pos) - np.linalg.norm(self.target_pos - self.start_pos)
        progress_reward = 0.1 * progress
        
        reward = distance_reward + energy_penalty + progress_reward
        
        # Check if target reached (terminated)
        terminated = distance_to_target < self.target_threshold
        if terminated:
            reward += 100  # Bonus for reaching target
        
        # Check if maximum steps reached (truncated)
        truncated = self.step_count >= self.max_steps
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, terminated, truncated, info
    
    def _get_obs(self) -> npt.NDArray[np.float32]:
        """Get current observation.
        
        Returns:
            np.ndarray: The current observation (position and wind).
        """
        wind = self._get_wind_at_pos(self.current_pos)
        return np.concatenate([self.current_pos, wind]).astype(np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current info.
        
        Returns:
            Dict[str, Any]: Information about the current state.
        """
        return {
            "distance_to_target": np.linalg.norm(self.current_pos - self.target_pos),
            "step_count": self.step_count,
            "total_energy_consumed": self.total_energy_consumed,
            "current_wind": self._get_wind_at_pos(self.current_pos)
        }
    
    def _get_wind_at_pos(self, pos: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Get wind vector at given position.
        
        Args:
            pos (np.ndarray): Position to get wind at.
            
        Returns:
            np.ndarray: Wind vector at the given position.
        """
        x, y, z = np.clip(pos.astype(int), 0, self.grid_size - 1)
        return self.wind[x, y, z].astype(np.float32)
    
    def _is_within_bounds(self, pos: npt.NDArray[np.float32]) -> bool:
        """Check if position is within grid bounds.
        
        Args:
            pos (np.ndarray): Position to check.
            
        Returns:
            bool: True if position is within bounds.
        """
        return np.all(pos >= 0) and np.all(pos < self.grid_size)
    
    def _check_collision(self, pos: npt.NDArray[np.float32]) -> bool:
        """Check if position collides with any obstacle.
        
        Args:
            pos (np.ndarray): Position to check.
            
        Returns:
            bool: True if position collides with an obstacle.
        """
        for obstacle in self.obstacles:
            if (np.all(pos >= obstacle.min_pos) and np.all(pos <= obstacle.max_pos)):
                return True
        return False
    
    def _update_wind(self) -> None:
        """Update wind field with more realistic turbulent components."""
        t = self.step_count * self.wind_frequency
        
        # Base wind with multiple frequencies for more realistic turbulence
        base_wind = (
            np.sin(t + self.wind_phase) * 0.5 +
            np.sin(2*t + self.wind_phase) * 0.3 +
            np.sin(0.5*t + self.wind_phase) * 0.2
        ) * self.wind_scale
        
        # Add random noise
        random_component = np.random.uniform(-self.wind_noise, self.wind_noise, self.wind.shape)
        
        # Smooth the wind field
        smoothed_wind = np.zeros_like(self.wind)
        for i in range(1, self.grid_size-1):
            for j in range(1, self.grid_size-1):
                for k in range(1, self.grid_size-1):
                    smoothed_wind[i,j,k] = np.mean(self.wind[i-1:i+2, j-1:j+2, k-1:k+2], axis=(0,1,2))
        
        self.wind = (base_wind + random_component + 0.3 * smoothed_wind).astype(np.float32)
    
    def _render_frame(self) -> None:
        """Render the current frame."""
        if self.fig is None:
            self.fig = plt.figure(figsize=(10, 10))
            self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.clear()
        
        # Plot obstacles
        for obstacle in self.obstacles:
            min_pos = obstacle.min_pos
            max_pos = obstacle.max_pos
            
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
                self.ax.plot_trisurf(x, y, z, color='red', alpha=0.2)
        
        # Plot UAV position
        self.ax.scatter(self.current_pos[0], self.current_pos[1], self.current_pos[2], 
                       c='blue', marker='o', s=100, label='UAV')
        
        # Plot target position
        self.ax.scatter(self.target_pos[0], self.target_pos[1], self.target_pos[2], 
                       c='green', marker='*', s=200, label='Target')
        
        # Plot wind vectors (subsampled)
        step = 4  # Plot every 4th point
        for i in range(0, self.grid_size, step):
            for j in range(0, self.grid_size, step):
                for k in range(0, self.grid_size, step):
                    wind = self.wind[i, j, k]
                    if np.linalg.norm(wind) > 0.1:  # Only plot significant wind
                        self.ax.quiver(i, j, k, 
                                     wind[0], wind[1], wind[2],
                                     color='gray', alpha=0.3,
                                     length=0.5, normalize=True)
        
        # Set plot limits and labels
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_zlim(0, self.grid_size)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend()
        
        # Update the display
        plt.draw()
        plt.pause(0.01)
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment.
        
        Returns:
            Optional[np.ndarray]: RGB array of the rendered frame if mode is "rgb_array".
        """
        if self.render_mode == "human":
            self._render_frame()
            return None
        elif self.render_mode == "rgb_array":
            if self.fig is None:
                self._render_frame()
            return np.array(self.fig.canvas.renderer.buffer_rgba())
        return None  # Add rgb_array support if needed
    
    def close(self) -> None:
        """Close the environment."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
