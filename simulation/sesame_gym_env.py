import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

class SesameGymEnv(gym.Env):
    """
    Custom Gym environment for the 8-DOF Sesame Quadruped Robot using MuJoCo.
    """
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        
        # Load native MJCF model
        model_path = os.path.join(os.path.dirname(__file__), "sesame.xml")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Simulation parameters
        self.dt_physics = self.model.opt.timestep  # usually 0.002s
        self.control_freq = 50.0  # 50 Hz control loop
        self.decimation = int((1.0 / self.control_freq) / self.dt_physics)  # 10 steps
        
        # Joint Limits (in radians, matching mechanical limitations)
        # Hips: FR, FL range [-45, 90] deg; BR, BL range [-90, 45] deg
        # Knees: All range [-135, 0] deg (0 is extended)
        self.joint_limits_lower = np.array([
            -0.7854, -1.5708,  # FR Hip, FR Knee
            -1.5708, -1.5708,  # BR Hip, BR Knee
            -0.7854, -1.5708,  # FL Hip, FL Knee
            -1.5708, -1.5708   # BL Hip, BL Knee
        ])
        self.joint_limits_upper = np.array([
            1.5708, 0.0,      # FR Hip, FR Knee
            0.7854, 0.0,      # BR Hip, BR Knee
            1.5708, 0.0,      # FL Hip, FL Knee
            0.7854, 0.0       # BL Hip, BL Knee
        ])
        
        # Standing configuration (all knees extended, hips rotated)
        # FR: 45 deg, BR: -45 deg, FL: 45 deg, BL: -45 deg
        self.stand_angles = np.array([
            0.7854, -0.8,      # FR
            -0.7854, -0.8,     # BR
            0.7854, -0.8,      # FL
            -0.7854, -0.8      # BL
        ])
        
        # Environment spaces
        # Action space: 8 joint target deltas in [-1.0, 1.0] scaled to max 0.1 rad (~5.7 deg)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        
        # Observation space (29 dimensions):
        # - Joint positions (8)
        # - Joint velocities (8)
        # - Base orientation: roll, pitch (2)
        # - Base linear velocity: vx, vy, vz (3)
        # - Base angular velocity: wx, wy, wz (3)
        # - Command velocity: vx_cmd, vy_cmd, yaw_cmd (3)
        # - Gait clock phase: sin(phase), cos(phase) (2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32
        )
        
        # State variables
        self.joint_targets = self.stand_angles.copy()
        self.cmd_velocity = np.zeros(3)  # [vx_cmd, vy_cmd, yaw_cmd]
        
        # Domain Randomization ranges
        self.randomize = True
        self.base_mass_range = [0.100, 0.160]      # Default 130g ± 30g
        self.friction_range = [0.4, 1.2]            # Ground friction
        self.actuator_delay_range = [1, 3]          # Delay in control steps (20ms to 60ms)
        self.sensor_noise_std = {
            'joint_pos': 0.02,                      # rad
            'joint_vel': 0.1,                       # rad/s
            'imu_angle': 0.03,                      # rad
            'imu_vel': 0.1                          # rad/s
        }
        
        # Buffers for actuator delay simulation
        self.target_history = []
        self.actuator_delay = 1
        self.prev_action = np.zeros(8)
        
        # Rendering
        self.render_mode = render_mode
        self.viewer = None
        self.renderer = None
        self._step_counter = 0

    def step(self, action):
        self._step_counter += 1
        
        # Scale action to max angle delta (e.g. 0.03 rad per control step)
        max_delta = 0.03
        delta_q = action * max_delta
        
        # Update and clip joint targets
        self.joint_targets = np.clip(
            self.joint_targets + delta_q, self.joint_limits_lower, self.joint_limits_upper
        )
        
        # Simulating actuator delay: push target to history, get delayed target
        self.target_history.append(self.joint_targets.copy())
        if len(self.target_history) > 5:
            self.target_history.pop(0)
            
        # Select target with delay index
        delayed_target = self.target_history[-min(len(self.target_history), self.actuator_delay + 1)]
        
        # Run physics decimation steps
        for _ in range(self.decimation):
            # Send targets to MuJoCo position actuators
            # MuJoCo handles PD computation internally in C
            self.data.ctrl[:] = delayed_target
            
            # Step physics
            mujoco.mj_step(self.model, self.data)
            
        # Get base states
        base_pos = self.data.qpos[0:3]
        base_quat = self.data.qpos[3:7]  # [w, x, y, z]
        base_lin_vel = self.data.qvel[0:3]
        base_ang_vel = self.data.qvel[3:6]
        
        # Convert quaternion to roll, pitch
        roll, pitch = self._quat_to_roll_pitch(base_quat)
        
        # Fetch current observations with noise
        obs = self._get_obs(roll, pitch, base_lin_vel, base_ang_vel)
        
        # Compute rewards
        reward = self._compute_reward(roll, pitch, base_pos, base_lin_vel, base_ang_vel, action)
        
        # Check termination (fall conditions or height limit)
        # The standing height of the robot is around 0.08m to 0.10m.
        # If the base falls below 0.04m or tilts more than 40 degrees (~0.7 rad), terminate.
        terminated = False
        truncated = False
        
        if base_pos[2] < 0.020 or abs(roll) > 0.7 or abs(pitch) > 0.7:
            terminated = True
            print(f"DEBUG TERMINATION: Height={base_pos[2]:.4f}, Roll={roll:.4f}, Pitch={pitch:.4f}")
            # Penalize early termination to break the collapse exploit
            reward -= 50.0
            
        if self._step_counter >= 500:  # 10 seconds max episode length
            truncated = True
            
        # Update action history
        self.prev_action = action.copy()
            
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_counter = 0
        self.prev_action = np.zeros(8)
        
        # Reset physics
        mujoco.mj_resetData(self.model, self.data)
        
        # Domain Randomization
        if self.randomize:
            # 1. Randomize body mass
            new_mass = np.random.uniform(self.base_mass_range[0], self.base_mass_range[1])
            self.model.body_mass[1] = new_mass  # body_mass[1] is base_link (body_mass[0] is world)
            
            # 2. Randomize friction
            new_friction = np.random.uniform(self.friction_range[0], self.friction_range[1])
            self.model.geom_friction[0, 0] = new_friction  # Floor geom friction
            
            # 3. Randomize actuator delay
            self.actuator_delay = np.random.randint(self.actuator_delay_range[0], self.actuator_delay_range[1] + 1)
        else:
            self.actuator_delay = 1
            
        # Reset delay history
        self.target_history = [self.stand_angles.copy()]
        self.joint_targets = self.stand_angles.copy()
        
        # Set initial base position slightly off the ground (e.g. z = 0.038 m)
        self.data.qpos[0:3] = [0.0, 0.0, 0.038]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # [w, x, y, z]
        # Initialize joint position to standing angles
        self.data.qpos[7:15] = self.stand_angles
        
        if self.randomize:
            # Randomize Command Velocity
            # vx: [-0.1, 0.4] m/s, vy: [-0.05, 0.05] m/s, yaw: [-0.3, 0.3] rad/s
            self.cmd_velocity = np.array([
                np.random.uniform(-0.1, 0.4),
                np.random.uniform(-0.05, 0.05),
                np.random.uniform(-0.3, 0.3)
            ])
            # 10% chance of command standing still
            if np.random.rand() < 0.1:
                self.cmd_velocity *= 0
        else:
            # Fixed forward walk command for clean evaluation
            self.cmd_velocity = np.array([0.25, 0.0, 0.0])
            
        mujoco.mj_forward(self.model, self.data)
        
        # Get base states
        base_quat = self.data.qpos[3:7]
        roll, pitch = self._quat_to_roll_pitch(base_quat)
        base_lin_vel = self.data.qvel[0:3]
        base_ang_vel = self.data.qvel[3:6]
        
        obs = self._get_obs(roll, pitch, base_lin_vel, base_ang_vel)
        return obs, {}

    def _get_obs(self, roll, pitch, lin_vel, ang_vel):
        q = self.data.qpos[7:15].copy()
        dq = self.data.qvel[6:14].copy()
        
        # Add noise if randomization is enabled
        if self.randomize:
            q += np.random.normal(0, self.sensor_noise_std['joint_pos'], size=8)
            dq += np.random.normal(0, self.sensor_noise_std['joint_vel'], size=8)
            roll += np.random.normal(0, self.sensor_noise_std['imu_angle'])
            pitch += np.random.normal(0, self.sensor_noise_std['imu_angle'])
            lin_vel = lin_vel + np.random.normal(0, self.sensor_noise_std['imu_vel'], size=3)
            ang_vel = ang_vel + np.random.normal(0, self.sensor_noise_std['imu_vel'], size=3)
            
        # Gait clock phase (sin and cos) at target frequency (e.g. 1.5 Hz)
        gait_freq = 1.5
        phase = 2 * np.pi * gait_freq * (self._step_counter / self.control_freq)
        sin_phase = np.sin(phase)
        cos_phase = np.cos(phase)
        
        # Concatenate into 29-dim observation
        obs = np.concatenate([
            q,                      # 8
            dq,                     # 8
            [roll, pitch],          # 2
            lin_vel,                # 3
            ang_vel,                # 3
            self.cmd_velocity,      # 3
            [sin_phase, cos_phase]  # 2 -> Total 29
        ]).astype(np.float32)
        
        return obs

    def _compute_reward(self, roll, pitch, pos, lin_vel, ang_vel, action):
        # 1. Velocity tracking rewards
        # Project world velocity to base local frame (simplified since roll/pitch are small)
        vx_local = lin_vel[0]
        vy_local = lin_vel[1]
        wz_local = ang_vel[2]
        
        r_vx = np.exp(-((vx_local - self.cmd_velocity[0]) ** 2) / 0.02)
        r_vy = np.exp(-((vy_local - self.cmd_velocity[1]) ** 2) / 0.02)
        r_wz = np.exp(-((wz_local - self.cmd_velocity[2]) ** 2) / 0.02)
        
        # Additive reward structure is much easier for RL to optimize than multiplicative
        # Prioritize forward velocity (0.7) and penalize lateral and angular velocities
        tracking_reward = 0.7 * r_vx + 0.15 * r_vy + 0.15 * r_wz
        
        # 2. Penalties
        # Roll and Pitch tilt penalty (keep robot flat)
        tilt_penalty = -5.0 * (roll ** 2 + pitch ** 2)
        
        # Height penalty (keep base around 0.042m stand height)
        # Use absolute error with a gain of 100.0 to prevent vanishing gradients
        height_err = pos[2] - 0.042
        height_penalty = -100.0 * abs(height_err)
        
        # Smoothness / Torque penalty using actuators force
        # In MuJoCo, data.actuator_force contains the output of the actuators
        actuator_forces = self.data.actuator_force.copy()
        torque_penalty = -0.05 * np.sum(np.abs(actuator_forces))
        
        # Joint speed penalty (penalize jitter)
        jerk_penalty = -0.01 * np.sum(np.abs(self.data.qvel[6:14]))
        
        # Yaw tracking penalty (always active, penalizes deviation from commanded yaw rate)
        yaw_drift_penalty = -5.0 * ((wz_local - self.cmd_velocity[2]) ** 2)
            
        # Action rate penalty (penalize joint target changes)
        action_rate_penalty = -0.02 * np.sum(np.square(action - self.prev_action))
            
        # Combine rewards
        reward = (
            5.0 * tracking_reward
            + tilt_penalty
            + height_penalty
            + torque_penalty
            + jerk_penalty
            + yaw_drift_penalty
            + action_rate_penalty
        )
        
        # Small living reward to prevent sitting down
        reward += 0.5
        
        return reward

    def _quat_to_roll_pitch(self, q):
        # MuJoCo quaternion format [w, x, y, z]
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        # roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = np.copysign(np.pi / 2.0, sinp)
        else:
            pitch = np.arcsin(sinp)
            
        return roll, pitch

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            
            # Setup tracking camera to follow base_link (body id 1)
            camera = mujoco.MjvCamera()
            camera.trackbodyid = 1  # base_link
            camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            camera.distance = 0.4
            camera.elevation = -25
            camera.azimuth = 135
            
            self.renderer.update_scene(self.data, camera=camera)
            return self.renderer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer = None
