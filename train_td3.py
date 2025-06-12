import os
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch
from uav_env import UAVEnv
import wandb
from datetime import datetime
import logging
from typing import Optional, Dict, Any
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def setup_wandb(config: Optional[Dict[str, Any]] = None) -> None:
    if config is None:
        config = {
            "algorithm": "TD3",
            "env": "UAVEnv",
            "total_timesteps": 200_000,
            "learning_rate": 3e-4,
            "buffer_size": 1_000_000,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "policy_delay": 2,
            "target_noise_clip": 0.5,
            "target_policy_noise": 0.2,
        }
    try:
        wandb.init(
            project="uav-rl-path-planning",
            config=config,
            sync_tensorboard=True
        )
    except Exception as e:
        logging.warning(f"Failed to initialize wandb: {e}")


def make_env():
    try:
        env = UAVEnv(render_mode=None)
        env = Monitor(env)
        return env
    except Exception as e:
        logging.error(f"Failed to create environment: {e}")
        raise


class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_energies = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if infos:
            info = infos[0]
            done = self.locals.get("dones", [False])[0]
            reward = self.locals.get("rewards", [0])[0]

            if done:
                self.episode_rewards.append(reward)
                self.episode_lengths.append(self.num_timesteps)
                self.episode_energies.append(info.get("total_energy_consumed", 0))

                if len(self.episode_rewards) % 10 == 0:
                    wandb.log({
                        "episode_reward": float(np.mean(self.episode_rewards[-10:])),
                        "episode_length": float(np.mean(self.episode_lengths[-10:])),
                        "energy_consumption": float(np.mean(self.episode_energies[-10:])),
                        "global_step": self.num_timesteps
                    })
        return True


def main():
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/run_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)

        setup_wandb()

        env = DummyVecEnv([make_env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

        eval_env = DummyVecEnv([make_env])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False)

        n_actions = env.action_space.shape[-1]
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.2 * np.ones(n_actions),
            theta=0.15,
            dt=0.1
        )

        policy_kwargs = dict(
            net_arch=dict(
                pi=[512, 256, 128],
                qf=[512, 256, 128]
            ),
            activation_fn=torch.nn.ReLU,
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs=dict(weight_decay=1e-4)
        )

        model = TD3(
            "MlpPolicy",
            env,
            action_noise=action_noise,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            policy_delay=2,
            target_noise_clip=0.5,
            target_policy_noise=0.2,
            train_freq=1,
            gradient_steps=1,
            verbose=1,
            device='auto',
            tensorboard_log=log_dir
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(log_dir, "best_model"),
            log_path=log_dir,
            eval_freq=1000,
            n_eval_episodes=5,
            deterministic=True,
            render=False
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=os.path.join(log_dir, "checkpoints"),
            name_prefix="td3_uav"
        )

        callbacks = CallbackList([
            eval_callback,
            checkpoint_callback,
            WandbCallback()
        ])

        logging.info("Starting training...")
        model.learn(
            total_timesteps=200_000,
            callback=callbacks,
            progress_bar=True
        )

        final_model_path = os.path.join(log_dir, "td3_uav_final")
        model.save(final_model_path)
        logging.info(f"Training completed! Model saved in {log_dir}")

        env_stats_path = os.path.join(log_dir, "vec_normalize.pkl")
        env.save(env_stats_path)

    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise
    finally:
        if 'env' in locals():
            env.close()
        if 'eval_env' in locals():
            eval_env.close()
        if wandb is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
