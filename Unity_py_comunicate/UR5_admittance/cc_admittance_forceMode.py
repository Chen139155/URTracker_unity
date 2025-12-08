#!/usr/bin/env python3
# coding: utf-8
__author__ = ["Chen Chen", "Huang Yihang"]
__date__ = '2025/11/20'
__version__ = '0.1'
"""
Admittance control using UR force_mode (External Control + RTDE)
- Requires URBasic (https://github.com/DavidUrz/URBasic) available and working
- Requires hex.udp_get() to return force sensor reading [fx, fy, fz, tx, ty, tz]

Modified for rehabilitation robot with on-demand assistance strategy
"""

import sys
import os
import time
import math
import threading
import numpy as np
import logging

# ensure URBasic package is importable; adapt path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import URBasic
import URBasic.robotModel
import URBasic.urScriptExt

# user-specific force sensor reader (your existing module)
import hex  # keep using your hex.udp_get() - must return array-like [fx,fy,fz,tx,ty,tz]

logger = logging.getLogger(__name__)

class KalmanFilter:
    """
    A simple Kalman Filter implementation for 1D or multi-dimensional systems.
    """
    def __init__(self, initial_state, initial_uncertainty, process_noise, measurement_noise):
        self.state = np.array(initial_state, dtype=float)  # Initial state estimate
        self.uncertainty = np.array(initial_uncertainty, dtype=float)  # Initial uncertainty (covariance)
        self.process_noise = np.array(process_noise, dtype=float)  # Process noise covariance
        self.measurement_noise = np.array(measurement_noise, dtype=float)  # Measurement noise covariance
        
    def predict(self, dt=1.0, control_input=None):
        """
        Prediction step of the Kalman filter
        For simplicity, assume constant velocity model in this example
        """
        # State transition matrix (constant velocity model)
        # [x, vx] => [x+vx*dt, vx]
        n_dim = len(self.state) // 2
        F = np.eye(len(self.state))
        for i in range(n_dim):
            F[i, i+n_dim] = dt
            
        # Predict new state
        self.state = F @ self.state
        if control_input is not None:
            B = np.zeros((len(self.state), len(control_input)))
            for i in range(n_dim):
                B[i, i] = 0.5 * dt**2
                B[i+n_dim, i] = dt
            self.state += B @ np.array(control_input)
            
        # Predict new uncertainty
        self.uncertainty = F @ self.uncertainty @ F.T + self.process_noise
        
    def update(self, measurement):
        """
        Update step of the Kalman filter
        """
        # Measurement matrix (only positions are measured)
        n_dim_state = len(self.state)  # Total state dimensions
        n_dim_measure = len(measurement)  # Measurement dimensions
        
        # Create measurement matrix H
        H = np.zeros((n_dim_measure, n_dim_state))
        # Map state dimensions to measurement dimensions (assume first n_dim_measure/2 state positions map to measurements)
        for i in range(min(n_dim_measure, n_dim_state // 2)):
            H[i, i] = 1
            
        # Innovation (residual)
        innovation = np.array(measurement) - H @ self.state
        innovation_cov = H @ self.uncertainty @ H.T + self.measurement_noise
        
        try:
            # Kalman gain
            kalman_gain = self.uncertainty @ H.T @ np.linalg.inv(innovation_cov)
        except np.linalg.LinAlgError:
            # If matrix is singular, use pseudo-inverse
            kalman_gain = self.uncertainty @ H.T @ np.linalg.pinv(innovation_cov)
        
        # Update state and uncertainty
        self.state = self.state + kalman_gain @ innovation
        self.uncertainty = (np.eye(len(self.state)) - kalman_gain @ H) @ self.uncertainty

class AdmittanceForceMode:
    def __init__(self, 
                 ur_host='10.168.2.209',
                 freq_hz=125,
                 desired_pose=None,
                 # Safety parameters for rehabilitation robot
                 adm_M=np.array([50.0, 50.0, 50.0]),      # Larger mass for smooth motion
                 adm_D=np.array([20.0, 20.0, 20.0]),      # Appropriate damping
                 adm_K=np.array([5.0, 5.0, 5.0]),         # Lower stiffness
                 force_gain_pos=np.array([10.0, 10.0, 0.0]),  # Base position gain
                 force_gain_vel=np.array([2.0, 2.0, 0.0]),    # Base velocity gain
                 max_wrench=np.array([15.0, 15.0, 0.0]),     # Safe maximum force limit
                 selection_vector=[1, 1, 0, 0, 0, 0],
                 wrench_limits=[0.1, 0.1, 0.05, 0.1, 0.1, 0.1],
                 f_type=2,
                 bias_calibration_duration=3.0,
                 circle_center=None,
                 circle_radius=0.08,  # 8cm radius as required
                 circle_frequency=0.1):   # Lower frequency for safe motion
        # UR connection parameters
        self.ur_host = ur_host
        self.freq_hz = freq_hz
        self.dt = 1.0 / freq_hz  # target loop time (s)
        if desired_pose is None:
            desired_pose = np.array([0.625, -0.05, 0.400, 0.0, math.pi, 0.0])
        self.desired_pose = np.array(desired_pose, dtype=float)
        self.max_vel = 0.2  # Lower max velocity for safety
        self.max_acc = 0.2  # Lower max acceleration for safety

        # State variables initialization
        self.arm_pose = np.zeros(6, dtype=float)
        self.arm_pos = np.zeros(3, dtype=float)
        self.arm_vel = np.zeros(6, dtype=float)
        self.adm_x_e = np.zeros(3, dtype=float)
        self.adm_v = np.zeros(3, dtype=float)

        self.target_vel = np.zeros(3, dtype = float)  # 目标速度估计
        self.target_acc = np.zeros(3, dtype = float)  # 目标加速度估计
        self.arm_acc = np.zeros(3, dtype = float) # 末端加速度估计
        self.last_target_pos = np.zeros(3,dtype = float)
        self.last_target_vel = np.zeros(3,dtype = float)
        self.last_arm_vel = np.zeros(3, dtype = float)

        # Force sensor variables
        self.force_raw = np.zeros(3, dtype=float)
        self.force_filt = np.zeros(3, dtype=float)
        self.force_alpha = 0.7
        self.force_threshold = 1.5

        # On-demand assistance strategy parameters
        self.critical_error_threshold = 0.03  # 6cm threshold
        self.low_gain_coefficient = 20.0
        self.high_gain_coefficient = 50.0
        self.cubic_coefficient = 3000.0

        # Flags and locks
        self.running = False
        self.lock = threading.Lock()

        # Initialize Kalman filters BEFORE using them
        # This is the critical fix to resolve the AttributeError
        logger.info("[AdmittanceForceMode] Initializing Kalman filters...")
        initial_state_tcp = np.zeros(6)
        initial_uncertainty_tcp = np.eye(6) * 0.15
        process_noise_tcp = np.eye(6) * 0.01
        measurement_noise_tcp = np.eye(3) * 0.05
        
        self.tcp_kalman_filter = KalmanFilter(
            initial_state=initial_state_tcp,
            initial_uncertainty=initial_uncertainty_tcp,
            process_noise=process_noise_tcp,
            measurement_noise=measurement_noise_tcp
        )
        
        initial_state_force = np.zeros(3)
        initial_uncertainty_force = np.eye(3) * 0.15
        process_noise_force = np.eye(3) * 0.01
        measurement_noise_force = np.eye(3) * 0.05
        
        self.force_kalman_filter = KalmanFilter(
            initial_state=initial_state_force,
            initial_uncertainty=initial_uncertainty_force,
            process_noise=process_noise_force,
            measurement_noise=measurement_noise_force
        )

        # High-level admittance parameters
        self.M = np.array(adm_M, dtype=float)
        self.D = np.array(adm_D, dtype=float)
        self.K = np.array(adm_K, dtype=float)
        # self.Fk = np.array(force_gain_pos, dtype=float)
        # self.Fd = np.array(force_gain_vel, dtype=float)
        self.max_wrench = np.array(max_wrench, dtype=float)

        # Force mode parameters
        self.selection_vector = list(selection_vector)
        self.wrench_limits = list(wrench_limits)
        self.f_type = int(f_type)

        # UR connection objects
        self.robotModel = URBasic.robotModel.RobotModel()
        self.UR = URBasic.urScriptExt.UrScriptExt(host=self.ur_host, robotModel=self.robotModel)
        
        # Ensure robot is up
        try:
            self.UR.reset_error()
        except Exception:
            # Ignore here; user will see exceptions when running
            pass

        # Initialize robot-side force_mode program
        logger.info("[AdmittanceForceMode] Initializing force mode...")
        self.UR.init_force_remote(task_frame=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], f_type=self.f_type)

        # Now we can safely call get_state() since filters are initialized
        self.get_state()

        # Bias calibration
        self.bias_calibration_duration = bias_calibration_duration
        self.bias_x = 0.0
        self.bias_y = 0.0
        self.is_bias_calibrated = False
        self.calibration_samples = []
        self._calibrate_sensor_bias()

        # Get initial position as circle center
        logger.info("[AdmittanceForceMode] Getting initial robot pose for circle center...")
        initial_positions = []
        for _ in range(5):  # Read 5 times and average to reduce noise
            self.get_state()
            initial_positions.append(self.arm_pos.copy())
            time.sleep(0.1)
        
        # Use average as initial position
        self.circle_center = np.mean(initial_positions, axis=0)
        logger.info(f"[CRITICAL] Circle center set to initial position: {self.circle_center}")
        
        # Use provided center if available
        if circle_center is not None:
            self.circle_center = circle_center
            logger.info(f"[CRITICAL] Using provided circle center: {self.circle_center}")
        
        self.circle_radius = circle_radius
        self.circle_frequency = circle_frequency
        logger.info(f"[CRITICAL] Circle parameters: radius={circle_radius}m, frequency={circle_frequency}Hz")

        # Initialize robot to standstill
        logger.info("[AdmittanceForceMode] Setting initial zero wrench...")
        self.send_force_mode_command(wrench=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     selection_vector=self.selection_vector,
                                     limits=self.wrench_limits,
                                     task_frame=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     f_type=self.f_type)

        # Additional state variables
        self.trajectory_mode = "circle"
        self.start_time = None
        self.last_force_time = 0
        self.in_admittance_mode = False
        self.current_angle = 0.0  # For debugging, track current angle

    # small helper to call URBasic's set_force_remote safely
    def send_force_mode_command(self, wrench, selection_vector=None, limits=None, task_frame=None, f_type=None):
        if selection_vector is None:
            selection_vector = self.selection_vector
        if limits is None:
            limits = self.wrench_limits
        if task_frame is None:
            task_frame = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if f_type is None:
            f_type = self.f_type
        # URBasic method already does RTDE.setData(...) + sendData()
        try:
            self.UR.set_force_remote(task_frame=task_frame,
                                     selection_vector=selection_vector,
                                     wrench=wrench,
                                     limits=limits,
                                     f_type=f_type)
            return True
        except Exception as e:
            print("send_force_mode_command exception:", e)
            return False

    def _calibrate_sensor_bias(self):
        # Calibrate sensor
        print(f"[AdmittanceForceMode] Starting sensor calibration ({self.bias_calibration_duration}s)...")
        
        calibration_samples_x = []
        calibration_samples_y = []
        start_time = time.perf_counter()
        sample_count = 0
        
        # collect samples for calibration with higher frequency
        while (time.perf_counter() - start_time) < self.bias_calibration_duration:
            try:
                raw_data = hex.udp_get()
                if isinstance(raw_data, (list, tuple)) and len(raw_data) >= 3:
                    raw = np.asarray(raw_data, dtype=float)
                    calibration_samples_x.append(raw[0])
                    calibration_samples_y.append(raw[1])
                    sample_count += 1
                time.sleep(0.005)  # Increased sampling frequency
            except Exception as e:
                print(f"Calibration sample error: {e}")
                continue
        
        # calculate average biases
        if len(calibration_samples_x) > 0:
            self.bias_x = -np.mean(calibration_samples_x)
            self.bias_y = -np.mean(calibration_samples_y)
            self.is_bias_calibrated = True
            print(f"[AdmittanceForceMode] Calibration finished: bias_x={self.bias_x:.3f}, bias_y={self.bias_y:.3f}, samples={sample_count}")
        else:
            # if no samples, set default values
            self.bias_x = 0.1
            self.bias_y = 0.3
            self.is_bias_calibrated = False
            logger.warning("Calibration failed, using default values")
    

    def read_force_sensor(self):
        """
        Read force with improved sensitivity and appropriate filtering
        """
        try:
            raw_data = hex.udp_get()
            if not isinstance(raw_data, (list, tuple)) or len(raw_data) != 6:
                raise ValueError("Invalid data format from sensor")
            raw = np.asarray(raw_data, dtype=float)
        except (ValueError, TypeError) as e:
            print(f"Force sensor read error: {e}")
            raw = np.zeros(6, dtype=float)

        # Take only forces x,y,z
        f = raw[0:3]

        # Bias compensation
        f[0] += self.bias_x
        f[1] += self.bias_y
        f[2] = 0

        # Deadzone for noise filtering
        DEADZONE_THRESHOLD = 0.5  # 适当的死区阈值，过滤噪声

        for i in range(2):
            if abs(f[i]) < DEADZONE_THRESHOLD:
                f[i] = 0.0
            else:
                f[i] = np.sign(f[i]) * (abs(f[i]) - DEADZONE_THRESHOLD)

        # 使用卡尔曼滤波平滑力信号
        self.force_kalman_filter.predict(dt=self.dt)
        self.force_kalman_filter.update(f)
        
        self.force_filt = self.force_kalman_filter.state.copy()
        self.force_raw = f.copy()
        
        return self.force_filt

    def get_state(self):
        """
        Read UR pose & velocity via URBasic functions (blocking calls)
        关键方法：通过URBasic库获取机器人的实际TCP位姿和速度
        """
        try:
            # 关键修改：确保使用正确的方法获取机器人状态
            pose = np.asarray(self.UR.get_actual_tcp_pose(), dtype=float)
            vel = np.asarray(self.UR.get_actual_tcp_speed(), dtype=float)
        except Exception as e:
            print(f"Get state error: {e}")
            # fallback to last known values
            pose = self.arm_pose.copy()
            vel = self.arm_vel.copy()

        # 使用卡尔曼滤波器平滑位姿
        measurement = pose[:3]  # 只使用位置信息
        self.tcp_kalman_filter.predict(dt=self.dt)
        self.tcp_kalman_filter.update(measurement)
        
        # 获取滤波后的位姿和速度
        filtered_pose = self.tcp_kalman_filter.state[:3]
        filtered_vel = self.tcp_kalman_filter.state[3:]
        
        with self.lock:
            self.arm_pose[:3] = filtered_pose
            self.arm_pose[3:] = pose[3:] 
            self.arm_pos = filtered_pose.copy()
            self.last_arm_vel[:3] = self.arm_vel[:3]
            self.arm_vel[:3] = filtered_vel
            self.arm_vel[3:] = vel[3:]

    def get_circle_trajectory(self, t):
        """
        Calculate target position on circular trajectory with safe parameters
        Starts from the initial position and moves in a circle of 8cm radius in XY plane
        """
        # Calculate phase angle and normalize it to 0-2π range to prevent numerical issues
        raw_angle = 2 * math.pi * self.circle_frequency * t
        self.current_angle = raw_angle % (2 * math.pi)  # Normalize angle to 0-2π
        
        # Calculate point on circle (XY plane)
        x = self.circle_center[0] + self.circle_radius * math.cos(self.current_angle)
        y = self.circle_center[1] + self.circle_radius * math.sin(self.current_angle)
        z = self.circle_center[2]  # Keep z coordinate constant
        
        target_pos = np.array([x, y, z])
        
        # Calculate actual distance from center for debugging
        distance_from_center = np.linalg.norm(target_pos[:2] - self.circle_center[:2])
        
        # Add detailed debugging information every 0.5 seconds
        current_time = time.perf_counter() - self.start_time if self.start_time else 0
        if current_time % 0.5 < 0.02:  # Print every 0.5 seconds
            print(f"[DEBUG] Circle point: Angle={math.degrees(self.current_angle):.1f}°, Target=[{target_pos[0]:.6f}, {target_pos[1]:.6f}, {target_pos[2]:.6f}], Distance from center={distance_from_center:.6f}m")
        
        return target_pos

    def is_force_applied(self):
        """
        检测是否有外力作用
        """
        force_magnitude = np.linalg.norm(self.force_filt[:2])  # 只考虑XY平面的力
        is_applied = force_magnitude > self.force_threshold
        
        # 调试信息
        if is_applied:
            logger.debug(f"[DEBUG] Force detected! Magnitude: {force_magnitude:.2f}N, Threshold: {self.force_threshold}N")
        
        return is_applied

    def compute_target_position(self, elapsed_time):
        """
        Calculate target position for circular trajectory
        
        Args:
            elapsed_time: Time elapsed since trajectory start
            
        Returns:
            numpy array: Target position [x, y, z]
        """
        # Read force sensor data
        self.read_force_sensor()
        # Read robot state
        self.get_state()
        # Calculate target position using circular trajectory
        # target_pos = self.get_circle_trajectory(elapsed_time)
        target_pos = self.desired_pose[0:3].copy()
        
        self.arm_acc = (self.arm_vel[:3]-self.last_arm_vel[:3])/elapsed_time
        self.last_target_vel=self.target_vel.copy()
        self.target_vel = (self.desired_pose[:3]-self.last_target_pos[:3])/elapsed_time
        self.target_acc = (self.target_vel[:3]-self.last_target_vel[:3])/elapsed_time
        
        # Modified debug frequency to 2 seconds
        if int(elapsed_time) % 2 == 0 and elapsed_time - int(elapsed_time) < 0.1:  # Every 2 seconds
            current_pos = self.arm_pos.copy()
            pos_error = np.linalg.norm(target_pos - current_pos)
            distance_from_center = np.linalg.norm(current_pos[:2] - self.circle_center[:2])
            logger.debug(f"[DEBUG] Elapsed: {elapsed_time:.2f}s, Target: {target_pos}, Current: {current_pos}, Error: {pos_error:.3f}m")
            logger.debug(f"[DEBUG] Distance from center: {distance_from_center:.3f}m, Target radius: {self.circle_radius}m")
        
        return target_pos

    # def pos_error_to_wrench(self, target_pos):
    #     """
    #     Convert position error to wrench commands using on-demand assistance strategy
    #     """
    #     # Debug: Track position error for XY axes separately
    #     with self.lock:
    #         pos = self.arm_pos.copy()
    #         arm_vel = self.arm_vel[0:3].copy()
    #         target_vel = self.target_vel.copy()
    #         arm_acc = self.arm_acc.copy()
    #         target_acc = self.target_acc.copy()
    #         K = self.K.copy()
    #         D = self.D.copy()
    #         M = self.M.copy()

            

    #     # Calculate position errors vector
    #     pos_err = target_pos - pos
    #     vel_err = target_vel - arm_vel
    #     acc_err = target_acc - arm_acc
        
    #     # Calculate error magnitude
    #     error_magnitude = np.linalg.norm(pos_err[:2])
        
    #     # Initialize wrench
    #     wrench = np.zeros(3)
        
    #     # Axis-specific control gains to address elliptical trajectory
    #     # Different gains for X and Y axes to compensate for mechanical differences
    #     x_gain = 200.0  # X-axis proportional gain
    #     y_gain = 250.0  # Y-axis proportional gain (higher to compensate if needed)
    #     x_vel_damping = 0.4  # X-axis velocity damping
    #     y_vel_damping = 0.4  # Y-axis velocity damping
        
    #     # CRITICAL FIX: Use axis-specific gains for each coordinate to achieve circular trajectory
    #     for i in range(2):  # Only process XY plane
    #         error = pos_err[i]
            
    #         # Apply axis-specific proportional control
    #         if error_magnitude < self.critical_error_threshold:
    #             # Small error case: use axis-specific proportional control
    #             gain = x_gain if i == 0 else y_gain
    #             wrench[i] = gain * error
    #         else:
    #             # Large error case: use piecewise function with axis-specific gains
    #             direction = pos_err[i] / (error_magnitude + 1e-6)  # Avoid division by zero
                
    #             error_excess = error_magnitude - self.critical_error_threshold
                
    #             # Different gains for each axis in the piecewise function
    #             if i == 0:  # X-axis
    #                 base_gain = 400.0      # Can remain the same or be slightly increased
    #                 linear_gain = 900.0    # Consider increasing for faster response beyond 3cm
    #                 cubic_gain = 22000.0   # Consider increasing for even faster response to large errors
    #             else:  # Y-axis
    #                 base_gain = 400.0      # Can remain the same or be slightly increased
    #                 linear_gain = 900.0    # Consider increasing for faster response beyond 3cm
    #                 cubic_gain = 22000.0   # Consider increasing for even faster response to large errors
                     
    #             force_magnitude = (base_gain * self.critical_error_threshold +
    #                             linear_gain * error_excess +
    #                             cubic_gain * error_excess**3)
                
    #             wrench[i] = force_magnitude * direction
            
    #         # Use axis-specific velocity damping
    #         vel_damping = x_vel_damping if i == 0 else y_vel_damping
    #         wrench[i] += vel_damping * vel_err[i]
        
    #     # Axis-specific maximum force limits
    #     max_force_x = 40.0  # X-axis maximum force
    #     max_force_y = 40.0  # Y-axis maximum force
        
    #     # Clamp wrench values to axis-specific maximum limits
    #     for i in range(2):
    #         max_force = max_force_x if i == 0 else max_force_y
    #         if abs(wrench[i]) > max_force:
    #             wrench[i] = math.copysign(max_force, wrench[i])
    #             # Only print this debug info every 2 seconds
    #             current_time = time.perf_counter() - self.start_time
    #             if int(current_time) % 2 == 0 and current_time - int(current_time) < 0.1:
    #                 axis_name = 'X' if i == 0 else 'Y'
    #                 print(f"[DEBUG] {axis_name}-axis wrench clamped to max: {wrench[i]}N")

    #     # Complete 6D wrench
    #     wrench_full = [float(wrench[0]), float(wrench[1]), float(wrench[2]), 0.0, 0.0, 0.0]
        
    #     # Only print wrench info every 2 seconds
    #     current_time = time.perf_counter() - self.start_time
    #     if int(current_time) % 2 == 0 and current_time - int(current_time) < 0.1:
    #         print(f"[DEBUG] Sending wrench: {wrench_full}")
        
    #     return wrench_full
    def pos_error_to_wrench(self, target_pos):
        """
        Impedance/admittance style controller:
            F = M * (a_d - a) + D * (v_d - v) + K * (x_d - x)

        Uses:
        - self.M, self.D, self.K  (each can be length-3 array or 3x3 matrix)
        - self.target_vel, self.target_acc (estimated)
        - self.arm_vel, self.arm_acc (measured/estimated)
        - self.max_wrench (axis-wise limits, length-3)
        Returns 6D wrench list [Fx, Fy, Fz, 0,0,0]
        """

        # -------- 1) Thread-safe snapshot of variables --------
        with self.lock:
            pos = np.asarray(self.arm_pos.copy(), dtype=float)          # 3
            arm_vel = np.asarray(self.arm_vel[0:3].copy(), dtype=float) # 3
            arm_acc = np.asarray(self.arm_acc.copy(), dtype=float)      # 3

            target_vel = np.asarray(getattr(self, "target_vel", np.zeros(3)), dtype=float)
            target_acc = np.asarray(getattr(self, "target_acc", np.zeros(3)), dtype=float)

            K_raw = np.asarray(self.K, dtype=float)
            D_raw = np.asarray(self.D, dtype=float)
            M_raw = np.asarray(self.M, dtype=float)

            f_ext = np.asarray(self.force_filt, dtype=float)

            max_wrench_raw = np.asarray(getattr(self, "max_wrench", np.array([40.0, 40.0, 40.0])), dtype=float)

        # -------- 2) Ensure matrices are 3x3 --------
        def to_matrix(x):
            x = np.asarray(x, dtype=float)
            if x.ndim == 1 and x.size == 3:
                return np.diag(x)
            if x.shape == (3,):
                return np.diag(x)
            if x.shape == (3, 3):
                return x
            # Fallback: try to coerce
            return np.diag(np.asarray(x).flatten()[:3])

        K_mat = to_matrix(K_raw)
        D_mat = to_matrix(D_raw)
        M_mat = to_matrix(M_raw)

        # -------- 3) Compute errors (3D) --------
        target_pos = np.asarray(target_pos, dtype=float)
        pos_err = target_pos - pos              # x_d - x
        vel_err = target_vel - arm_vel          # v_d - v
        acc_err = target_acc - arm_acc          # a_d - a

        # -------- 4) Impedance formula --------
        # F = M*(a_d - a) + D*(v_d - v) + K*(x_d - x)
        wrench_xyz = M_mat.dot(acc_err) + D_mat.dot(vel_err) + K_mat.dot(pos_err) + f_ext

        # -------- 5) On-demand scaling (optional, modest): reduce assistance when error tiny --------
        # (keeps behavior gentle near target)
        # err_xy = np.linalg.norm(pos_err[:2])
        # if hasattr(self, "critical_error_threshold"):
        #     thr = float(self.critical_error_threshold)
        #     if err_xy < thr:
        #         # scale down smoothly when inside threshold
        #         scale = err_xy / (thr + 1e-9)  # in (0,1)
        #         # don't zero-out completely — keep small stiffness for stability
        #         min_scale = 0.15
        #         scale = max(scale, min_scale)
        #         wrench_xyz = wrench_xyz * scale

        # -------- 6) Axis-wise limits (use self.max_wrench if provided) --------
        # interpret max_wrench_raw as per-axis absolute limits for Fx,Fy,Fz
        max_limits = np.asarray(max_wrench_raw, dtype=float)
        # If only 2 entries given, assume Z limit = same as second or small
        if max_limits.size == 2:
            max_limits = np.array([max_limits[0], max_limits[1], 0.0])
        if max_limits.size == 1:
            max_limits = np.array([max_limits[0], max_limits[0], max_limits[0]])
        if max_limits.size < 3:
            tmp = np.zeros(3)
            tmp[:max_limits.size] = max_limits
            max_limits = tmp

        # Clip by axis limits
        for i in range(3):
            wrench_xyz[i] = float(np.clip(wrench_xyz[i], -abs(max_limits[i]), abs(max_limits[i])))
        
        # 限制z轴为0 
        wrench_xyz[2] = 0.0

        # -------- 7) Rate limiting: prevent large instant jumps in commanded force --------
        # store last command in self.last_wrench (3) if exists
        last_wrench = np.asarray(getattr(self, "last_wrench", np.zeros(3)), dtype=float)
        # choose max rate (N per second) — conservative default
        max_rate = getattr(self, "max_force_rate", 200.0)  # N/s (tunable)
        max_step = max_rate * max(self.dt, 1e-6)
        delta = wrench_xyz - last_wrench
        # clip delta per-axis
        delta_clipped = np.clip(delta, -max_step, max_step)
        wrench_xyz = last_wrench + delta_clipped
        # save back
        self.last_wrench = wrench_xyz.copy()

        # -------- 8) Build full 6D wrench and debug print every ~2s --------
        wrench_full = [float(wrench_xyz[0]), float(wrench_xyz[1]), float(wrench_xyz[2]), 0.0, 0.0, 0.0]

        if hasattr(self, "start_time") and self.start_time is not None:
            current_time = time.perf_counter() - self.start_time
            if int(current_time) % 2 == 0 and current_time - int(current_time) < 0.1:
                print(f"[IMPED] t={current_time:.1f}s pos_err={pos_err}, vel_err={vel_err}, acc_err={acc_err}")
                print(f"[IMPED] wrench_xyz={wrench_xyz.tolist()} (limits={max_limits.tolist()})")

        return wrench_full
    
    def control_loop(self):
        """
        Main control loop for rehabilitation robot with on-demand assistance
        Critical fix: Improve timing and trajectory updates
        """
        self.running = True
        next_tick = time.perf_counter()
        self.start_time = time.perf_counter()
        print(f"[INFO] Control loop started at {self.freq_hz} Hz")
        print(f"[CRITICAL] CIRCLE PARAMETERS: center={self.circle_center}, radius={self.circle_radius}m, frequency={self.circle_frequency}Hz")
        
        # Initialize with zero wrench
        zero_wrench = [0, 0, 0, 0, 0, 0]
        default_task_frame = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 使用基础坐标系

        # 确保机器人在开始时静止
        print("[INFO] Initializing robot to standstill...")
        self.send_force_mode_command(
            wrench=zero_wrench,
            selection_vector=self.selection_vector,
            limits=self.wrench_limits,
            task_frame=default_task_frame,
            f_type=self.f_type
        )
        time.sleep(1.0)  # 等待机器人稳定

        try:
            loop_count = 0
            while self.running:
                # 关键修改：使用绝对时间而不是相对时间计算elapsed_time
                current_time = time.perf_counter()
                elapsed_time = current_time - self.start_time
                
                # Read robot state
                self.get_state()
                
                # 强制使用圆形轨迹模式，确保专注于轨迹跟踪
                self.trajectory_mode = "circle"
                
                # Compute target position on circular trajectory
                target_pos = self.compute_target_position(elapsed_time)
                
                # Convert position error to wrench command
                cmd_wrench = self.pos_error_to_wrench(target_pos)

                # Send to robot
                ok = self.send_force_mode_command(
                    wrench=cmd_wrench,
                    selection_vector=self.selection_vector,
                    limits=self.wrench_limits,
                    task_frame=default_task_frame,
                    f_type=self.f_type
                )
                if not ok:
                    print("[ERROR] Failed to send force_mode command - CRITICAL")

                # Modified status update frequency to every 2 seconds
                loop_count += 1
                if loop_count % 250 == 0:  # Every 2 seconds (assuming 125Hz loop)
                    current_pos = self.arm_pos.copy()
                    pos_error = np.linalg.norm(target_pos - current_pos)
                    distance_from_center = np.linalg.norm(current_pos[:2] - self.circle_center[:2])
                    print(f"[STATUS] Running: {elapsed_time:.1f}s, Error: {pos_error:.3f}m, Distance from center: {distance_from_center:.3f}m")
                    print(f"[STATUS] Target radius: {self.circle_radius}m, Center: {self.circle_center}")

                # 关键修改：改进时间管理，确保循环频率
                next_tick += self.dt
                current_time = time.perf_counter()
                if current_time < next_tick:
                    time.sleep(next_tick - current_time)  # 使用非阻塞等待

        finally:
            # Safe shutdown
            print("[INFO] Initiating safe shutdown...")
            self.send_force_mode_command(
                wrench=zero_wrench,
                selection_vector=[0, 0, 0, 0, 0, 0],  # 关闭所有自由度
                limits=[0.01]*6,
                task_frame=default_task_frame,
                f_type=self.f_type
            )
            print("[INFO] Control loop stopped")

    def control_loop_test(self,data_g2r_q, data_r2g_q, command_q):
        """
        Main control loop for rehabilitation robot with on-demand assistance
        Critical fix: Improve timing and trajectory updates
        """
        self.running = True
        next_tick = time.perf_counter()
        self.start_time = time.perf_counter()
        logger.info(f"Control loop started at {self.freq_hz} Hz")
        logger.info(f"CIRCLE PARAMETERS: center={self.circle_center}, radius={self.circle_radius}m, frequency={self.circle_frequency}Hz")
        
        # Initialize with zero wrench
        zero_wrench = [0, 0, 0, 0, 0, 0]
        default_task_frame = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 使用基础坐标系

        # 确保机器人在开始时静止
        print("[INFO] Initializing robot to standstill...")
        self.send_force_mode_command(
            wrench=zero_wrench,
            selection_vector=self.selection_vector,
            limits=self.wrench_limits,
            task_frame=default_task_frame,
            f_type=self.f_type
        )
        time.sleep(1.0)  # 等待机器人稳定

        try:
            loop_count = 0
            while self.running:
                # 关键修改：使用绝对时间而不是相对时间计算elapsed_time
                current_time = time.perf_counter()
                elapsed_time = current_time - self.start_time
                
                if not command_q.empty():
                    command = command_q.get()
                    logger.info('command_q: ', command)
                    if isinstance(command, dict) and 'type' in command:
                        if command['type']=='update_params':
                            self.update_admittance_params(
                                command['K'],
                                command['D'],
                                command['M']
                                )
                if not data_g2r_q.empty():
                    data_g2r = data_g2r_q.get()
                    self.update_desired_pose(data_g2r)

                
                
                # 强制使用圆形轨迹模式，确保专注于轨迹跟踪
                # self.trajectory_mode = "circle"
                
                # Compute target position on circular trajectory
                target_pos = self.compute_target_position(elapsed_time)
                
                # Convert position error to wrench command
                cmd_wrench = self.pos_error_to_wrench(target_pos)

                # Send to robot
                ok = self.send_force_mode_command(
                    wrench=cmd_wrench,
                    selection_vector=self.selection_vector,
                    limits=self.wrench_limits,
                    task_frame=default_task_frame,
                    f_type=self.f_type
                )
                if not ok:
                    print("[ERROR] Failed to send force_mode command - CRITICAL")

                # Modified status update frequency to every 2 seconds
                loop_count += 1
                if loop_count % 250 == 0:  # Every 2 seconds (assuming 125Hz loop)
                    current_pos = self.arm_pos.copy()
                    pos_error = np.linalg.norm(target_pos - current_pos)
                    distance_from_center = np.linalg.norm(current_pos[:2] - self.circle_center[:2])
                    print(f"[STATUS] Running: {elapsed_time:.1f}s, Error: {pos_error:.3f}m, Distance from center: {distance_from_center:.3f}m")
                    print(f"[STATUS] Target radius: {self.circle_radius}m, Center: {self.circle_center}")

                # 关键修改：改进时间管理，确保循环频率
                next_tick += self.dt
                current_time = time.perf_counter()
                if current_time < next_tick:
                    time.sleep(next_tick - current_time)  # 使用非阻塞等待

        finally:
            # Safe shutdown
            print("[INFO] Initiating safe shutdown...")
            self.send_force_mode_command(
                wrench=zero_wrench,
                selection_vector=[0, 0, 0, 0, 0, 0],  # 关闭所有自由度
                limits=[0.01]*6,
                task_frame=default_task_frame,
                f_type=self.f_type
            )
            print("[INFO] Control loop stopped")


    def start(self,data_g2r_q, data_r2g_q, command_q):
        if self.running:
            logger.warning("Control Loop Already running")
            return
        # Start thread
        self.thread = threading.Thread(target=self.control_loop_test, args=(data_g2r_q, data_r2g_q, command_q), daemon=True)
        self.thread.start()
        logger.info(" Control started in separate thread")

    def stop(self):
        """Stop the admittance control and release all resources"""
        logger.info("Stopping control...")
        self.running = False
        
        # Wait for control thread to finish
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning("Control thread did not terminate gracefully")
        
        # Ensure robot returns to safe state
        try:
            print("[AdmittanceForceMode] Resetting robot errors...")
            self.UR.reset_error()
        except Exception as e:
            print(f"[AdmittanceForceMode] Warning: Failed to reset robot errors: {e}")
        
        # Stop force mode by sending zero wrench with all axes non-compliant
        try:
            print("[AdmittanceForceMode] Disabling force mode...")
            self.send_force_mode_command(
                wrench=[0, 0, 0, 0, 0, 0],
                selection_vector=[0, 0, 0, 0, 0, 0],
                limits=[0.01]*6,
                task_frame=[0, 0, 0, 0, 0, 0],
                f_type=self.f_type
            )
        except Exception as e:
            print(f"[AdmittanceForceMode] Warning: Failed to disable force mode: {e}")
        
        # Close UR connection
        try:
            print("[AdmittanceForceMode] Closing UR connection...")
            if hasattr(self.UR, 'close') and callable(self.UR.close):
                self.UR.close()
            elif hasattr(self.UR, 'disconnect') and callable(self.UR.disconnect):
                self.UR.disconnect()
        except Exception as e:
            print(f"[AdmittanceForceMode] Warning: Failed to close UR connection: {e}")
        
        # close socket
        try:
            if hasattr(hex, 'close') and callable(hex.close):
                print("[AdmittanceForceMode] Closing sensor connection...")
                hex.close()
            elif hasattr(hex, 'disconnect') and callable(hex.disconnect):
                print("[AdmittanceForceMode] Disconnecting sensor...")
                hex.disconnect()
        except Exception as e:
            print(f"[AdmittanceForceMode] Warning: Failed to close sensor connection: {e}")
        
        print("[AdmittanceForceMode] All resources released")

    # utility APIs to update params safely
    def update_desired_pose(self, pose2):
        with self.lock:
            self.last_target_pos = self.desired_pose[:3]
            self.desired_pose[0] = pose2[0]
            self.desired_pose[1] = pose2[1]

    def update_admittance_params(self, M=None, D=None, K=None):
        with self.lock:
            if M is not None:
                self.M = np.array(M, dtype=float)
            if D is not None:
                self.D = np.array(D, dtype=float)
            if K is not None:
                self.K = np.array(K, dtype=float)
                
    def set_trajectory_mode(self, mode):
        """
        设置轨迹模式: "circle" 或 "fixed"
        """
        with self.lock:
            if mode in ["circle", "fixed"]:
                self.trajectory_mode = mode
                print(f"[DEBUG] Set trajectory mode to: {mode}")
                
    def set_circle_params(self, center=None, radius=None, frequency=None):
        """
        设置圆形轨迹参数
        """
        with self.lock:
            if center is not None:
                self.circle_center = np.array(center, dtype=float)
                print(f"[DEBUG] Updated circle center: {self.circle_center}")
            if radius is not None:
                self.circle_radius = float(radius)
                print(f"[DEBUG] Updated circle radius: {self.circle_radius}m")
            if frequency is not None:
                self.circle_frequency = float(frequency)
                print(f"[DEBUG] Updated circle frequency: {self.circle_frequency}Hz")


if __name__ == '__main__':
    # 康复机器人安全参数配置
    adm = AdmittanceForceMode(
            ur_host='10.168.2.209',
            freq_hz=125,
            desired_pose=([0.625, -0.05, 0.400, 0.0, math.pi, 0.0]),  # 将使用机器人实际初始位姿
            # 关键修改：调整参数以确保精确跟踪圆形轨迹
            adm_M=np.array([40.0, 40.0, 40.0]),
            adm_D=np.array([100.0, 100.0, 100.0]),
            adm_K=np.array([250.0, 250.0, 250.0]),
            force_gain_pos=np.array([10.0, 10.0, 0.0]),  # 基础位置增益
            force_gain_vel=np.array([2.0, 2.0, 0.0]),    # 基础速度增益
            max_wrench=np.array([40.0, 40.0, 0.0]),     # 安全的最大力限制
            selection_vector=[1, 1, 0, 0, 0, 0],        # 只控制XY平面
            wrench_limits=[0.1, 0.1, 0.05, 0.1, 0.1, 0.1],
            f_type=2,
            bias_calibration_duration=3.0,
            circle_center=None,  # 将使用初始位置作为圆心
            circle_radius=0.08,  # 8cm半径，符合康复需求
            circle_frequency=0.10)  # 较低的频率，确保运动缓慢

    try:
        adm.start()
        print("\n[INFO] Rehabilitation robot control started. Running in circle trajectory mode.")
        print("Press Ctrl+C to stop...\n")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        adm.stop()