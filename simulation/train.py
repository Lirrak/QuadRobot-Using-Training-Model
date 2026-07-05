import os
import argparse
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback, CallbackList

# Import our custom environment
from sesame_gym_env import SesameGymEnv

class EntropyDecayCallback(BaseCallback):
    def __init__(self, initial_ent_coef: float, verbose: int = 0):
        super().__init__(verbose)
        self.initial_ent_coef = initial_ent_coef

    def _on_step(self) -> bool:
        # self.model._current_progress_remaining goes from 1.0 down to 0.0
        self.model.ent_coef = self.initial_ent_coef * self.model._current_progress_remaining
        return True

def train_policy(total_timesteps=1000000, n_envs=8, log_dir="./logs/", continue_train=False, checkpoint_path=None, ent_coef=0.01):
    os.makedirs(log_dir, exist_ok=True)
    
    initial_ent_coef = ent_coef
    print(f"Using initial entropy coefficient: {initial_ent_coef}")
    
    norm_path = os.path.join(log_dir, "sesame_vec_normalize.pkl")
    model_path = os.path.join(log_dir, "sesame_ppo_model.zip")
    
    # Option 1: Load from a specific checkpoint file for fine-tuning
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path} to fine-tune...")
        env = make_vec_env(SesameGymEnv, n_envs=n_envs)
        if os.path.exists(norm_path):
            env = VecNormalize.load(norm_path, env)
            env.training = True
            env.norm_reward = True
            print(f"  Loaded normalizer from {norm_path}")
        else:
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
            print("  No normalizer found, starting fresh normalizer.")
        model = PPO.load(checkpoint_path, env=env, ent_coef=ent_coef, device="cpu")
        print(f"  Checkpoint loaded (ent_coef={ent_coef}). Resuming training for {total_timesteps} more steps.")
    # Option 2: Continue from latest saved model
    elif continue_train and os.path.exists(model_path) and os.path.exists(norm_path):
        print(f"Loading existing model and normalizer from {log_dir} to continue training...")
        env = VecNormalize.load(norm_path, make_vec_env(SesameGymEnv, n_envs=n_envs))
        env.training = True
        env.norm_reward = True
        model = PPO.load(model_path, env=env, ent_coef=ent_coef, device="cpu")
    else:
        if continue_train:
            print("Warning: Existing model/normalizer files not found. Starting training from scratch.")
        print(f"Starting training from scratch for {total_timesteps} timesteps with {n_envs} environments...")
        env = make_vec_env(SesameGymEnv, n_envs=n_envs)
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        
        # Tiny network architecture (2 hidden layers of 128 units)
        # This guarantees <1.5ms execution time on Pi Zero W / Zero 2W.
        policy_kwargs = dict(
            net_arch=dict(
                pi=[128, 128],
                vf=[128, 128]
            )
        )
        
        # Instantiate PPO Agent
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ent_coef,
            device="cpu",
            verbose=1,
            tensorboard_log=os.path.join(log_dir, "tb_logs")
        )
    
    # Set up checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000, total_timesteps // 10),
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="sesame_ppo"
    )
    
    # Set up callbacks list
    callbacks = [checkpoint_callback]
    if initial_ent_coef > 0:
        callbacks.append(EntropyDecayCallback(initial_ent_coef))
        print(f"Added EntropyDecayCallback with initial ent_coef={initial_ent_coef}")
    
    callback_list = CallbackList(callbacks)
    
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        progress_bar=True
    )
    duration = time.time() - start_time
    print(f"Training finished in {duration:.1f} seconds.")
    
    # Save the model and normalization parameters
    model.save(os.path.join(log_dir, "sesame_ppo_model"))
    env.save(norm_path)
    print("Saved final model and normalizer.")
    env.close()

def evaluate_policy(model_path="./logs/sesame_ppo_model", norm_path="./logs/sesame_vec_normalize.pkl", render=True, record=False):
    print(f"Evaluating model: {model_path}")
    
    # Create evaluation environment
    render_mode = "rgb_array" if record else ("human" if render else None)
    eval_env = SesameGymEnv(render_mode=render_mode)
    
    # Turn off domain randomization for clean evaluation
    eval_env.randomize = False
    
    # Load normalizer
    if os.path.exists(norm_path):
        from stable_baselines3.common.vec_env import VecNormalize
        # We need a dummy vectorized env to load the normalizer stats
        dummy_env = make_vec_env(lambda: eval_env, n_envs=1)
        vec_env = VecNormalize.load(norm_path, dummy_env)
        # Turn off updates to normalizer during evaluation
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        print("Warning: Normalization file not found. Running without observation normalization.")
        vec_env = make_vec_env(lambda: eval_env, n_envs=1)

    # Load trained model
    model = PPO.load(model_path, env=vec_env, device="cpu")
    
    # Set up video writer if recording
    video_writer = None
    if record:
        record_dir = os.path.dirname(model_path)
        os.makedirs(record_dir, exist_ok=True)
        record_path = os.path.join(record_dir, "evaluation.mp4")
        print(f"Recording evaluation to: {record_path}")
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(record_path, fourcc, 30.0, (640, 480))
        
    obs = vec_env.reset()
    total_reward = 0
    total_orig_reward = 0
    steps = 0
    
    try:
        while True:
            # Predict action using deterministic policy
            action, _states = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, rewards, dones, infos = vec_env.step(action)
            total_reward += rewards[0]
            if hasattr(vec_env, "get_original_reward"):
                total_orig_reward += vec_env.get_original_reward()[0]
            else:
                total_orig_reward += rewards[0]
            steps += 1
            
            if record:
                # Capture frame
                frame = eval_env.render()
                if frame is not None:
                    # Convert RGB to BGR for OpenCV
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_writer.write(bgr_frame)
                # Save 1 full episode (max 500 steps)
                if steps >= 500 or dones[0]:
                    print(f"Recording finished! Steps: {steps}, Total scaled reward: {total_reward:.2f}, Total original reward: {total_orig_reward:.2f}")
                    break
            else:
                if render:
                    eval_env.render()
                    # Pause slightly to match physical real-time (50Hz = 20ms step duration)
                    time.sleep(0.02)
                    
                if dones[0]:
                    print(f"Episode finished! Steps: {steps}, Total scaled reward: {total_reward:.2f}, Total original reward: {total_orig_reward:.2f}")
                    obs = vec_env.reset()
                    total_reward = 0
                    total_orig_reward = 0
                    steps = 0
                
    except KeyboardInterrupt:
        print("Evaluation stopped by user.")
    finally:
        if video_writer is not None:
            video_writer.release()
            print("Video saved.")
        eval_env.close()
        vec_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate Sesame Quadruped walking policy.")
    parser.add_argument("--train", action="store_true", help="Train a policy")
    parser.add_argument("--eval", action="store_true", help="Evaluate a trained policy")
    parser.add_argument("--continue", action="store_true", dest="continue_train", help="Continue training from the existing model/normalizer")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a specific .zip checkpoint to fine-tune from")
    parser.add_argument("--steps", type=int, default=1000000, help="Number of steps to train (default: 1,000,000)")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering during evaluation")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments for training")
    parser.add_argument("--record", action="store_true", help="Record simulation run to a video file")
    parser.add_argument("--ent-coef", type=float, default=0.01, dest="ent_coef", help="Entropy coefficient (default: 0.01, set 0.0 to disable exploration bonus)")
    
    args = parser.parse_args()
    
    # Default to train if nothing is selected
    if not args.train and not args.eval:
        args.train = True
        
    if args.train:
        train_policy(total_timesteps=args.steps, n_envs=args.n_envs, continue_train=args.continue_train, checkpoint_path=args.checkpoint, ent_coef=args.ent_coef)
    elif args.eval:
        evaluate_policy(render=not args.no_render, record=args.record)
