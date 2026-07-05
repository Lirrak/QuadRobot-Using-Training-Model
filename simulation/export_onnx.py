import os
import argparse
import pickle
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

# Define the PyTorch wrapper that packages observation normalization inside the ONNX graph
class SesameONNXPolicy(torch.nn.Module):
    def __init__(self, policy, obs_mean, obs_var, clip_obs=10.0):
        super().__init__()
        self.policy = policy
        
        # Register normalization parameters as buffers (so they are saved/exported as constants)
        self.register_buffer("obs_mean", torch.tensor(obs_mean, dtype=torch.float32))
        self.register_buffer("obs_var", torch.tensor(obs_var, dtype=torch.float32))
        self.clip_obs = clip_obs

    def forward(self, obs):
        # 1. Normalize observation: (obs - mean) / sqrt(var + 1e-8)
        # Note: we use 1e-8 to match stable-baselines3 epsilon value
        norm_obs = (obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)
        norm_obs = torch.clamp(norm_obs, -self.clip_obs, self.clip_obs)
        
        # 2. Feedforward pass through PPO Actor network (MlpPolicy)
        features = self.policy.features_extractor(norm_obs)
        latent_pi, _ = self.policy.mlp_extractor(features)
        actions = self.policy.action_net(latent_pi)
        # Clamp actions to [-1.0, 1.0] to match stable-baselines3 default action clipping
        actions = torch.clamp(actions, -1.0, 1.0)
        return actions

def export_to_onnx(model_path="./logs/sesame_ppo_model.zip", norm_path="./logs/sesame_vec_normalize.pkl", output_path="./deploy/sesame_policy.onnx"):
    print(f"Loading model from: {model_path}")
    print(f"Loading normalizer from: {norm_path}")
    
    if not os.path.exists(model_path):
        # Check if the path needs .zip suffix
        if os.path.exists(model_path + ".zip"):
            model_path = model_path + ".zip"
        else:
            print(f"Error: Model file '{model_path}' not found!")
            return

    if not os.path.exists(norm_path):
        print(f"Error: Normalization file '{norm_path}' not found!")
        return

    # Create deploy folder if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Load the running mean and variance stats from VecNormalize
    with open(norm_path, "rb") as f:
        vec_normalize_stats = pickle.load(f)
        
    obs_mean = vec_normalize_stats.obs_rms.mean
    obs_var = vec_normalize_stats.obs_rms.var
    clip_obs = getattr(vec_normalize_stats, "clip_obs", 10.0)
    
    print("Normalizer statistics loaded:")
    print("  Mean shape:", obs_mean.shape)
    print("  Var shape:", obs_var.shape)

    # 2. Load the trained SB3 model
    # We load onto CPU since the Pi will run CPU inference
    model = PPO.load(model_path, device="cpu")
    policy = model.policy
    policy.eval()  # Set policy to evaluation mode
    
    # 3. Create the wrapped PyTorch model
    wrapped_model = SesameONNXPolicy(policy, obs_mean, obs_var, clip_obs)
    wrapped_model.eval()

    # 4. Export to ONNX
    # Observation dimension is 27
    dummy_input = torch.randn(1, 27, dtype=torch.float32)
    
    print(f"Exporting ONNX model to: {output_path} ...")
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action": {0: "batch_size"}
        }
    )
    
    print("ONNX export completed successfully!")
    
    # 5. Verify the ONNX model using ONNX Runtime
    try:
        import onnxruntime as ort
        import numpy as np
        
        session = ort.InferenceSession(output_path)
        print("\nVerifying ONNX model with ONNX Runtime:")
        print("  Inputs:", [inp.name for inp in session.get_inputs()])
        print("  Outputs:", [out.name for out in session.get_outputs()])
        
        # Test run
        test_obs = np.random.randn(1, 27).astype(np.float32)
        onnx_outputs = session.run(["action"], {"observation": test_obs})
        print("  Test inference output shape:", onnx_outputs[0].shape)
        print("  Sample actions:", onnx_outputs[0][0].round(4))
        
    except ImportError:
        print("onnxruntime is not installed. Skipping local verification.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export stable-baselines3 model to ONNX format.")
    parser.add_argument("--model", type=str, default="./logs/sesame_ppo_model", help="Path to trained SB3 model")
    parser.add_argument("--norm", type=str, default="./logs/sesame_vec_normalize.pkl", help="Path to VecNormalize pkl file")
    parser.add_argument("--output", type=str, default="./deploy/sesame_policy.onnx", help="Output path for ONNX model")
    
    args = parser.parse_args()
    export_to_onnx(args.model, args.norm, args.output)
