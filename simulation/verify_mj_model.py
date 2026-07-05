import os
import random
import numpy as np

try:
    import mujoco
except ImportError:
    print("Error: 'mujoco' package is not installed. Please run: pip install mujoco")
    exit(1)

def main():
    model_path = os.path.join(os.path.dirname(__file__), "sesame.xml")
    print(f"Loading MJCF from: {model_path}")
    
    if not os.path.exists(model_path):
        print("Error: sesame.xml does not exist!")
        return

    try:
        # Load MJCF model in MuJoCo
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        print("Successfully loaded model!")
        print(f"Number of joints: {model.njnt}")
        print(f"Number of actuators/actuated joints: {model.nu}")
        print(f"Number of bodies: {model.nbody}")
        
        # Verify joint names
        for i in range(model.njnt):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            print(f"  Joint {i}: name='{joint_name}', type={model.jnt_type[i]}")
            
    except Exception as e:
        print(f"Failed to load XML model: {e}")
        return

    # Let's run a passive simulation to verify gravity and collisions
    print("\nRunning test simulation (1000 steps)...")
    
    # Reset model state
    mujoco.mj_resetData(model, data)
    
    # Place base_link slightly above the ground (e.g. z = 0.15 m)
    # The first 7 components of qpos represent the free joint of the base_link (x, y, z, qx, qy, qz, qw)
    data.qpos[2] = 0.15  # Height offset
    
    for step in range(1000):
        # Apply random joint controls within active bounds using built-in actuators
        for i in range(model.nu):
            data.ctrl[i] = random.uniform(-0.1, 0.1)
        
        mujoco.mj_step(model, data)
        
        # Periodically check base height and stability
        if step % 200 == 0:
            pos = data.qpos[0:3]
            vel = data.qvel[0:3]
            print(f"  Step {step:4d}: Base Pos = {pos.round(4)}, Base Lin Vel = {vel.round(4)}")
            
            if np.isnan(pos).any():
                print("Error: NaN detected in state! Simulation exploded.")
                return

    print("Passive simulation test completed successfully!")

if __name__ == "__main__":
    main()
