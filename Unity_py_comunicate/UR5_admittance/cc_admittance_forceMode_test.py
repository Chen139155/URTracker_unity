#!/usr/bin/env python3
# coding: utf-8
__author__ = 'Chen Chen'
__date__ = '2025/11/20'
__version__ = '0.1'
"""
Admittance control using UR force_mode (External Control + RTDE)
- Requires URBasic (https://github.com/DavidUrz/URBasic) available and working
- Requires hex.udp_get() to return force sensor reading [fx, fy, fz, tx, ty, tz]

"""

import sys
import os
import time
import math
import threading
import numpy as np

# ensure URBasic package is importable; adapt path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import URBasic
import URBasic.robotModel
import URBasic.urScriptExt

# user-specific force sensor reader (your existing module)
import hex  # keep using your hex.udp_get() - must return array-like [fx,fy,fz,tx,ty,tz]


class AdmittanceForceMode:
    def __init__(self,
                 ur_host='10.168.2.209',
                 freq_hz=125,                      # RTDE update rate (125Hz -> 0.008s)
                 desired_pose=None,                # initial desired pose (x,y,z,rx,ry,rz)
                 adm_M=np.array([40.0, 40.0, 40.0]),   # (virtual) M for high-level admittance integrator (not UR internal)
                 adm_D=np.array([80.0, 80.0, 80.0]),
                 adm_K=np.array([250.0, 250.0, 250.0]),
                 force_gain_pos=np.array([50.0, 50.0, 50.0]),   # gain to convert position error -> command wrench
                 force_gain_vel=np.array([10.0, 10.0, 10.0]),   # damping on velocity error -> command wrench
                 max_wrench=np.array([30.0, 30.0, 30.0]),       # N, clamp commanded wrench on x,y,z
                 selection_vector=[1, 1, 1, 0, 0, 0],           # compliant axes: 1-> compliant
                 wrench_limits=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # UR interpretation: speeds/position limits
                 f_type=2,                                      # 1: force, 2: torque
                 bias_calibration_duration=3.0):                # duration of bias calibration
        # UR connection parameters
        self.ur_host = ur_host
        self.freq_hz = freq_hz
        self.dt = 1.0 / freq_hz  # target loop time (s)
        if desired_pose is None:
            desired_pose = np.array([0.625, -0.05, 0.400, 0.0, math.pi, 0.0])
        self.desired_pose = np.array(desired_pose, dtype=float)   # target reference pose (may be updated externally)

        # high-level admittance params (used to compute a desired position offset x_e from measured F_ext)
        self.M = np.array(adm_M, dtype=float)
        self.D = np.array(adm_D, dtype=float)
        self.K = np.array(adm_K, dtype=float)

        # mapping from pos/vel errors -> wrench (what we will send into UR force_mode)
        self.Fk = np.array(force_gain_pos, dtype=float)
        self.Fd = np.array(force_gain_vel, dtype=float)
        self.max_wrench = np.array(max_wrench, dtype=float)

        # force_mode parameters
        self.selection_vector = list(selection_vector)
        self.wrench_limits = list(wrench_limits)
        self.f_type = int(f_type)

        # state vars
        self.arm_pose = np.zeros(6, dtype=float)
        self.arm_pos = np.zeros(3, dtype=float)
        self.arm_vel = np.zeros(6, dtype=float)
        self.adm_x_e = np.zeros(3, dtype=float)      # integrated position offset from admittance
        self.adm_v = np.zeros(3, dtype=float)        # integrated velocity for admittance internal integrator

        # external force reading (filtered)
        self.force_raw = np.zeros(3, dtype=float)
        self.force_filt = np.zeros(3, dtype=float)
        self.force_alpha = 0.8   # exponential smoothing factor for force sensor

        # safety & limits
        self.max_vel = 0.5
        self.max_acc = 0.5

        # flags & locks
        self.running = False
        self.lock = threading.Lock()

        # UR connection objects
        self.robotModel = URBasic.robotModel.RobotModel()
        self.UR = URBasic.urScriptExt.UrScriptExt(host=self.ur_host, robotModel=self.robotModel)
        # ensure robot is up
        ok = self.UR_.reset_error() if hasattr(self, 'UR_') else None  # noop if not present
        # but our UR object is self.UR (from URBasic)
        # call reset_error in safe way
        try:
            self.UR.reset_error()
        except Exception:
            # ignore here; user will see exceptions when running
            pass

        # initialize robot-side force_mode program
        # This will send the real-time URScript program that reads RTDE registers and calls force_mode()
        self.UR.init_force_remote(task_frame=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], f_type=self.f_type)

        # set initial wrench/selection/limits via RTDE
        self.send_force_mode_command(wrench=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     selection_vector=self.selection_vector,
                                     limits=self.wrench_limits,
                                     task_frame=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     f_type=self.f_type)
        
        # bias calibration
        self.bias_calibration_duration = bias_calibration_duration
        self.bias_x = 0.0
        self.bias_y = 0.0
        self.is_bias_calibrated = False
        self.calibration_samples = []

        self._calibrate_sensor_bias()

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
        print(f"[AdmittanceForceMode] start senor cailbration ({self.bias_calibration_duration}s)...")
        
        calibration_samples_x = []
        calibration_samples_y = []
        start_time = time.perf_counter()
        
        # collect samples for calibration
        while (time.perf_counter() - start_time) < self.bias_calibration_duration:
            try:
                raw_data = hex.udp_get()
                if isinstance(raw_data, (list, tuple)) and len(raw_data) >= 3:
                    raw = np.asarray(raw_data, dtype=float)
                    calibration_samples_x.append(raw[0])
                    calibration_samples_y.append(raw[1])
                time.sleep(0.01)  # 10ms sampling duation
            except Exception:
                continue
        
        # calculate average biases
        if len(calibration_samples_x) > 0:
            self.bias_x = -np.mean(calibration_samples_x)
            self.bias_y = -np.mean(calibration_samples_y)
            self.is_bias_calibrated = True
            print(f"[AdmittanceForceMode] cailbration finish: bias_x={self.bias_x:.3f}, bias_y={self.bias_y:.3f}")
        else:
            # if no samples, set default values
            self.bias_x = 0.1
            self.bias_y = 0.3
            self.is_bias_calibrated = False
            print("[AdmittanceForceMode] cailbration failed, using default values")
    

    def read_force_sensor(self):
        """Read force from hex module and apply filtering / bias compensation / deadzone"""
        try:
            raw_data = hex.udp_get()
            # check if data is valid
            if not isinstance(raw_data, (list, tuple)) or len(raw_data) != 6:
                raise ValueError("Invalid data format from sensor")
            raw = np.asarray(raw_data, dtype=float)
        except (ValueError, TypeError):
            raw = np.zeros(6, dtype=float)
    
        # take only forces x,y,z
        f = raw[0:3]
    
        # bias compensation & deadzone (tunable)
        # these values should ideally be configurable parameters
        DEADZONE_THRESHOLD = 0.5
    
        f[0] += self.bias_x
        f[1] += self.bias_y
        f[2] = 0  # assuming no z-axis force is measured and to counteract with gravity
    
        # deadzone:
        for i in range(2):  # only apply to x and y axes
            if abs(f[i]) < DEADZONE_THRESHOLD:
                f[i] = 0.0
            else:
                # smooth transition from deadzone to full force
                f[i] = np.sign(f[i]) * (abs(f[i]) - DEADZONE_THRESHOLD)
    
        # exponential smoothing
        self.force_filt = self.force_alpha * self.force_filt + (1 - self.force_alpha) * f
        self.force_raw = f.copy()
    
        return self.force_filt

    def get_state(self):
        """Read UR pose & velocity via URBasic functions (blocking calls)"""
        try:
            pose = np.asarray(self.UR.get_actual_tcp_pose(), dtype=float)
            vel = np.asarray(self.UR.get_actual_tcp_speed(), dtype=float)
        except Exception:
            # fallback to last known values
            pose = self.arm_pose.copy()
            vel = self.arm_vel.copy()
        with self.lock:
            self.arm_pose = pose
            self.arm_pos = pose[0:3].copy()
            self.arm_vel = vel.copy()

    def compute_admittance_high_level(self, dt):
        """
        Compute high-level admittance integrator using measured external force.
        This outputs admittance_desired_position = desired_pose + adm_x_e
        We'll then convert pos error -> commanded wrench (force) and send to UR force_mode.
        """
        # read filtered external force
        F_ext = self.read_force_sensor()[0:3]   # use x,y,z
        # high-level admittance:
        # M * x_ddot + D * x_dot + K * x = F_ext
        # solve for x_ddot = (F_ext - D*x_dot - K*x)/M
        x = self.adm_x_e.copy()      # displacement from nominal desired
        v = self.adm_v.copy()
        

        # compute spring-damper correction (coupling wrench)
        coupling = self.D * v + self.K * x
        x_dd = (F_ext - coupling) / self.M

        # clamp acceleration
        acc_norm = np.linalg.norm(x_dd)
        if acc_norm > self.max_acc:
            x_dd = x_dd * (self.max_acc / acc_norm)

        # integrate
        v += x_dd * dt
        v_norm = np.linalg.norm(v)
        if v_norm > self.max_vel:
            v = v * (self.max_vel / v_norm)
        x += v * dt

        # update
        with self.lock:
            self.adm_v = v
            self.adm_x_e = x

        # return the desired absolute position
        return self.desired_pose[0:3] + self.adm_x_e

    def pos_error_to_wrench(self, target_pos):
        """
        Convert position error & velocity error to a commanded wrench (Fx, Fy, Fz)
        Simple PD mapping:
            wrench = Fk * pos_error + Fd * vel_error
        Note: vel_error = 0 - actual_tool_linear_velocity (we want to damp current motion)
        """
        with self.lock:
            pos = self.arm_pos.copy()
            vel = self.arm_vel[0:3].copy()

        pos_err = target_pos - pos
        vel_err = -vel   # we want to damp current velocity

        # fix position of z-axis
        pos_err[2] = 0.0
        vel_err[2] = 0.0

        # compute wrench
        wrench = self.Fk * pos_err + self.Fd * vel_err

        # clamp commanded wrench
        for i in range(2):
            if abs(wrench[i]) > self.max_wrench[i]:
                wrench[i] = math.copysign(self.max_wrench[i], wrench[i])


        # full 6d wrench: set torques to zero (or tune later)
        wrench_full = [float(wrench[0]), float(wrench[1]), float(wrench[2]), 0.0, 0.0, 0.0]
        return wrench_full

    def control_loop(self):
        """Main loop to compute high-level admittance and send wrench each iteration."""
        self.running = True
        next_tick = time.perf_counter()
        print("[AdmittanceForceMode] control loop started at {} Hz".format(self.freq_hz))

        # Safety: set initial parameters with small wrench and strict limits
        # Already set in __init__, but we reaffirm:
        self.send_force_mode_command(wrench=[0, 0, 0, 0, 0, 0],
                                     selection_vector=self.selection_vector,
                                     limits=self.wrench_limits,
                                     task_frame=[0, 0, 0, 0, 0, 0],
                                     f_type=self.f_type)

        while self.running:
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(next_tick - now)
            t0 = time.perf_counter()
            dt = self.dt  # target dt; could also compute t0-last_time for adaptive if needed

            # read UR state and force sensor
            self.get_state()
            # compute desired position using high-level admittance (so external force drives adm_x_e)
            adm_des_pos = self.compute_admittance_high_level(dt)

            # map pos error -> commanded wrench, and update UR force_mode inputs
            cmd_wrench = self.pos_error_to_wrench(adm_des_pos)

            # send to robot side
            ok = self.send_force_mode_command(wrench=cmd_wrench,
                                              selection_vector=self.selection_vector,
                                              limits=self.wrench_limits,
                                              task_frame=[0, 0, 0, 0, 0, 0],
                                              f_type=self.f_type)
            if not ok:
                print("[AdmittanceForceMode] Warning: failed to send force_mode command")

            # increment tick
            next_tick += self.dt
            # small debug print (comment out in normal runs)
            # print("wrench:", cmd_wrench, "pos_err:", adm_des_pos - self.arm_pos)

        # on exit: set wrench zero and stop force_mode by sending zero wrench
        self.send_force_mode_command(wrench=[0, 0, 0, 0, 0, 0],
                                     selection_vector=[0, 0, 0, 0, 0, 0],
                                     limits=[0.01]*6,
                                     task_frame=[0, 0, 0, 0, 0, 0],
                                     f_type=self.f_type)
        print("[AdmittanceForceMode] control loop stopped")

    def start(self):
        if self.running:
            print("Already running")
            return
        # make sure RTDE is running before starting loop (URBasic should already have started it)
        # Start thread
        self.thread = threading.Thread(target=self.control_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the admittance control and release all resources"""
        print("[AdmittanceForceMode] Stopping control...")
        self.running = False
        
        # Wait for control thread to finish
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                print("[AdmittanceForceMode] Warning: Control thread did not terminate gracefully")
        
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
    def update_desired_pose(self, pose6):
        with self.lock:
            self.desired_pose = np.array(pose6, dtype=float)

    def update_admittance_params(self, M=None, D=None, K=None):
        with self.lock:
            if M is not None:
                self.M = np.array(M, dtype=float)
            if D is not None:
                self.D = np.array(D, dtype=float)
            if K is not None:
                self.K = np.array(K, dtype=float)


if __name__ == '__main__':
    # example usage (tune gains very small for first run)
    adm = AdmittanceForceMode(
        ur_host='10.168.2.209',
        freq_hz=125,
        desired_pose=np.array([0.625, -0.05, 0.400, 0.0, math.pi, 0.0]),
        adm_M=np.array([10.0, 10.0, 10.0]),
        adm_D=np.array([10.0, 100.0, 100.0]),
        adm_K=np.array([50.0, 50.0, 50.0]),
        force_gain_pos=np.array([40.0, 40.0, 0]),  # pos->force mapping (start smaller)
        force_gain_vel=np.array([8.0, 8.0, 0]),
        max_wrench=np.array([20.0, 20.0, 0]),
        selection_vector=[1, 1, 0, 0, 0, 0],
        wrench_limits=[0.05, 0.05, 0.05, 0.1, 0.1, 0.1],
        f_type=2
    )

    try:
        adm.start()
        print("Admittance (force_mode) running. Press Ctrl-C to stop.")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        adm.stop()

