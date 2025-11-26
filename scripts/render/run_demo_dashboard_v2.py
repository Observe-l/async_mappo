#!/usr/bin/env python3
"""
HUD Dashboard with Model Comparison (Live vs Pretrained)
Based on PyQt5 + Matplotlib + SUMO + Pandas
"""
from __future__ import annotations

import sys
import argparse
import random
import time
import datetime
import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict as _Dict, Optional as _Optional

# PyQt5 Imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QLabel, QFrame, QProgressBar)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QColor, QPalette

# Matplotlib Imports
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Add repo root to path
REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# SUMO & RL Imports
try:
    import libsumo as traci
except Exception:
    import traci  # type: ignore

import torch
from onpolicy.config import get_config as _get_cfg
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Critic
from onpolicy.utils.shared_buffer import SharedReplayBuffer
from onpolicy.utils.valuenorm import ValueNorm

from onpolicy.envs.demo.mqtt_demo import async_scheduling
from onpolicy.iot.mqtt_bridge import MqttBridge, BridgeConfig

# --- Path Configuration ---
BASE_PATH = "/home/lwh/Documents/Code/RL-Scheduling/result/rul_threshold_7/async_mappo/2025-05-03-00-20/exp_hpAM/1000"

# -----------------------------------------------------------------------------
# Custom Environment Wrapper for Dashboard Layout
# -----------------------------------------------------------------------------

class DashboardAsyncScheduling(async_scheduling):
    def _init_sumo(self):
        """Override to position SUMO window to the left"""
        if getattr(self, '_sumo_started', False):
            return
        try:
            traci.close()
        except Exception:
            pass
        
        sumo_cmd = [
            "sumo-gui", 
            "-c", "/home/lwh/Documents/Code/RL-Scheduling/map/sg_map/osm.sumocfg", 
            "--threads", "20", 
            "--no-warnings", "True",
            "--window-pos", "0,0",
            "--window-size", "960,1000"
        ]
        
        if not self.use_gui:
            sumo_cmd[0] = "sumo"
            # Remove gui-specific args if needed, but for dashboard we assume gui
            
        traci.start(sumo_cmd)
        self._sumo_started = True

# -----------------------------------------------------------------------------
# Helper Classes (Copied/Adapted from run_demo_schedule_mqtt.py)
# -----------------------------------------------------------------------------

@dataclass
class EnvArgs:
    num_agents: int = 12
    use_rul_agent: bool = True
    rul_threshold: float = 7.0
    rul_state: bool = False
    use_gui: bool = True
    exp_type: str = "rul_schedule_demo_dashboard_v2"

class DistanceTracker:
    """Tracks per-vehicle total distance from SUMO, robust to vehicle delete/recreate resets."""
    def __init__(self):
        self._prev = {}
        self._offset = {}

    def total_m(self, veh_id: str) -> float:
        try:
            cur = float(traci.vehicle.getDistance(veh_id))
        except Exception:
            cur = None
        prev = self._prev.get(veh_id)
        offset = self._offset.get(veh_id, 0.0)
        
        if cur is None or cur < 0 or cur < -1e5:
            if prev is None:
                return offset
            return offset + max(prev, 0.0)
        
        if prev is None:
            self._prev[veh_id] = cur
            self._offset.setdefault(veh_id, 0.0)
            return offset + cur
        
        if cur + 1e-6 < prev and cur >= 0.0:
            offset += prev
            self._offset[veh_id] = offset
            
        self._prev[veh_id] = cur
        return offset + cur

class RemotePolicy:
    def __init__(self, bridge: MqttBridge, action_dim: int, device_map: _Dict[str, str] = None):
        self.bridge = bridge
        self.action_dim = action_dim
        self._step_id = 0
        self.device_map = device_map or {}
        self._last_action = {}
        self._last_rul = {}

    def next_step(self):
        self._step_id += 1

    def act(self, agent_label: str, obs_vec: np.ndarray, sensor_features) -> int:
        action = None; device_id = None; ok = False; _rul = None
        try:
            if agent_label in self.device_map:
                device_id = self.device_map[agent_label]
                ret = self.bridge.request_to(device_id=device_id, step_id=self._step_id, agent_id=agent_label,
                                             env_obs=obs_vec, sensor=sensor_features, action_dim=self.action_dim)
            else:
                ret = self.bridge.request(step_id=self._step_id, agent_id=agent_label,
                                          env_obs=obs_vec, sensor=sensor_features, action_dim=self.action_dim)
            if isinstance(ret, (tuple, list)):
                if len(ret) == 5:
                    _rul, action, log_prob, ok, device_id = ret
                else:
                    _rul, action, ok, device_id = ret
        except Exception as e:
            print(f"[SERVER][ERROR] request failed step={self._step_id} agent={agent_label}: {e}")
            ok = False
        
        if ok and isinstance(action, int):
            act = int(action) % self.action_dim
            self._last_action[agent_label] = act
            try:
                self._last_rul[agent_label] = float(_rul)
            except Exception:
                pass
            return act
            
        if agent_label in self._last_action:
            return self._last_action[agent_label]
        
        act = random.randrange(self.action_dim)
        self._last_action[agent_label] = act
        return act

    def get_last_rul(self, agent_label: str):
        return self._last_rul.get(agent_label)

    def pickup(self, agent_label: str, candidates = None, action_dim: int = 45) -> int:
        candidates = candidates if candidates is not None else list(range(action_dim))
        action = None; ok = False; _rul = None
        try:
            if agent_label in self.device_map:
                ret = self.bridge.request_to(device_id=self.device_map[agent_label], step_id=self._step_id,
                                             agent_id=agent_label, env_obs=[], sensor=[], action_dim=action_dim,
                                             decision_type='pickup', pickup_candidates=candidates)
            else:
                ret = self.bridge.request(step_id=self._step_id, agent_id=agent_label, env_obs=[], sensor=[],
                                           action_dim=action_dim, decision_type='pickup', pickup_candidates=candidates)
            if isinstance(ret, (tuple, list)):
                if len(ret) == 5:
                    _rul, action, _log_prob, ok, _dev = ret
                else:
                    _rul, action, ok, _dev = ret
        except Exception as e:
            print(f"[SERVER][ERROR] pickup request failed step={self._step_id} agent={agent_label}: {e}")
            ok = False
            
        if ok and isinstance(action, int):
            act = int(action) % action_dim
            self._last_action[agent_label] = act
            try:
                self._last_rul[agent_label] = float(_rul)
            except Exception:
                pass
            return act
            
        act = random.randrange(action_dim)
        self._last_action[agent_label] = act
        return act

# -----------------------------------------------------------------------------
# Data Loader Class
# -----------------------------------------------------------------------------

class BaselineLoader:
    def __init__(self, base_path):
        print(f"[BASELINE] Loading data from {base_path}")
        try:
            # 1. Load Files
            df_prod = pd.read_csv(f"{base_path}/product.csv")
            df_dist = pd.read_csv(f"{base_path}/distance.csv")
            # df_res = pd.read_csv(f"{base_path}/result.csv") # Optional if you need reward

            # 2. Align Data (Ensure same length/index)
            # Assuming 'step_length' or index aligns them. 
            # Let's clean column names just in case (strip spaces)
            df_dist.columns = df_dist.columns.str.strip()
            df_prod.columns = df_prod.columns.str.strip()

            # 3. Calculate Aggregates
            # Sum all 'total_truck_X' columns for total distance
            truck_cols = [c for c in df_dist.columns if 'total_' in c and 'truck' in c]
            self.total_distance = df_dist[truck_cols].sum(axis=1)
            
            self.total_product = df_prod['total']
            # Assuming 10 is your step interval, adjust if needed. 
            # Based on previous scripts, logging might be every step or interval.
            # Let's assume the index corresponds to the logging interval.
            # If logging is every step, then index is step.
            # If logging is every 200s, then index * 200 is time.
            # Let's assume index roughly maps to simulation progress for comparison.
            # For better alignment, we should use the 'time' column if available.
            if 'time' in df_prod.columns:
                 self.steps = df_prod['time'] * 3600 # Convert hours back to seconds if needed, or just use time
            else:
                 self.steps = df_prod.index * 10 
            
            # 4. Calculate Profit (The Formula)
            # Profit = (Prod * 10) - (Dist * 0.00001)
            self.profit = (self.total_product * 10) - (self.total_distance * 0.00001)
            print(f"[BASELINE] Loaded {len(self.steps)} data points")
        except Exception as e:
            print(f"[BASELINE] Error loading data: {e}")
            self.steps = []
            self.total_product = []
            self.profit = []

    def get_full_data(self):
        """Returns full dataset for plotting as reference"""
        return self.steps, self.total_product, self.profit

# -----------------------------------------------------------------------------
# HUD Dashboard Class
# -----------------------------------------------------------------------------

class HUD_Dashboard(QMainWindow):
    def __init__(self, env_args: EnvArgs, bridge_cfg: BridgeConfig):
        super().__init__()
        self.env_args = env_args
        self.bridge_cfg = bridge_cfg
        
        # --- UI Setup: Translucent & Frameless ---
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # Move Dashboard to the right side (960, 50) to avoid blocking SUMO (0,0)
        self.setGeometry(960, 50, 900, 600)
        
        # Create semi-transparent background
        self.central_widget = QWidget()
        self.central_widget.setStyleSheet("""
            QWidget { background-color: rgba(20, 20, 20, 200); border-radius: 15px; border: 1px solid #555; }
            QLabel { color: white; background: transparent; }
        """)
        self.setCentralWidget(self.central_widget)
        
        # --- Data Initialization ---
        self.baseline = BaselineLoader(BASE_PATH)
        self.live_steps = []
        self.live_prod = []
        self.live_profit = []
        
        # Live Metric Accumulators
        self.current_prod = 0
        self.current_dist_sum = 0.0
        self.step_count = 0
        
        # RL State
        self.obs = None
        self.waiting_for_action = False
        self.dist_tracker = DistanceTracker()
        
        # --- Legacy Features Setup ---
        self.init_env()
        
        # --- UI Elements ---
        self.init_ui_elements()
        
        # --- Game Loop ---
        self.timer = QTimer()
        self.timer.setInterval(50) 
        self.timer.timeout.connect(self.update_loop)
        self.timer.start()

    def init_env(self):
        """Initialize SUMO and RL Environment"""
        # Use the custom wrapper that positions SUMO correctly
        self.env = DashboardAsyncScheduling(self.env_args)
        self.obs = self.env.reset()
        print("[DASHBOARD] Env initialized")
        
        # Setup Policy
        action_dim = self.env.factory_num if self.env.use_rul_agent else self.env.factory_num + 1
        self.bridge = MqttBridge(self.bridge_cfg)
        
        try:
            device_ids = list(self.bridge_cfg.devices)
        except Exception:
            device_ids = []
        fixed_map = {f'truck_{i}': device_ids[i] for i in range(min(self.env.truck_num, len(device_ids)))}
        
        self.policy = RemotePolicy(self.bridge, action_dim, device_map=fixed_map)
        self.truck_ids = []

    def init_ui_elements(self):
        layout = QVBoxLayout(self.central_widget)
        
        # Header
        header = QLabel("Async-MAPPO Logistics HUD | Live vs Pretrained")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #00ffff; margin-bottom: 10px;")
        layout.addWidget(header)
        
        # Charts Layout
        charts_layout = QHBoxLayout()
        
        # Chart 1: Production
        self.fig1 = Figure(figsize=(4, 3), dpi=100)
        self.canvas1 = FigureCanvas(self.fig1)
        self.ax1 = self.fig1.add_subplot(111)
        self.setup_chart(self.fig1, self.ax1, "Production Comparison")
        
        # Chart 2: Profit
        self.fig2 = Figure(figsize=(4, 3), dpi=100)
        self.canvas2 = FigureCanvas(self.fig2)
        self.ax2 = self.fig2.add_subplot(111)
        self.setup_chart(self.fig2, self.ax2, "Profit Comparison")
        
        charts_layout.addWidget(self.canvas1)
        charts_layout.addWidget(self.canvas2)
        
        layout.addLayout(charts_layout)
        
        # Footer / Status
        self.status_label = QLabel("Status: Running")
        self.status_label.setAlignment(Qt.AlignRight)
        layout.addWidget(self.status_label)

    def setup_chart(self, fig, ax, title):
        fig.patch.set_facecolor('none') # Transparent figure background
        ax.set_facecolor('none') # Transparent axes background
        ax.set_title(title, color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.grid(True, color='#444', linestyle=':', alpha=0.5)

    def update_loop(self):
        if traci.simulation.getMinExpectedNumber() <= 0:
            self.status_label.setText("Status: Completed")
            self.timer.stop()
            return

        # 1. SUMO Step & RL Logic (Legacy Logic)
        self.run_rl_step()
        
        current_sim_time = traci.simulation.getTime()

        # 2. Legacy: Camera Track
        self.camera_track_logic()

        # 4. Calculate Live Metrics
        if not self.truck_ids:
            self.truck_ids = [f'truck_{i}' for i in range(self.env.truck_num)]
        
        # A. Distance
        # Use DistanceTracker for robust cumulative distance
        total_distance_m = 0.0
        for tid in self.truck_ids:
            total_distance_m += self.dist_tracker.total_m(tid)
        self.current_dist_sum = total_distance_m
        
        # B. Production
        final_factories = ["Factory45", "Factory46", "Factory47", "Factory48", "Factory49"]
        current_prod = 0
        for fid in final_factories:
            if fid in self.env.factory:
                current_prod += self.env.factory[fid].total_final_product
        self.current_prod = current_prod

        # C. Profit Formula
        # Profit = (Prod * 10) - (Dist * 0.00001)
        current_profit = (self.current_prod * 10) - (self.current_dist_sum * 0.00001)
        
        # 5. Store Data
        # We use step count or sim time for x-axis. 
        # Baseline uses 'time' column which is hours. 
        # Let's use hours for x-axis to match baseline if possible, or just raw seconds/steps.
        # BaselineLoader converts time to seconds if 'time' column exists.
        self.live_steps.append(current_sim_time)
        self.live_prod.append(self.current_prod)
        self.live_profit.append(current_profit)

        # 6. Refresh UI (Visual Comparison)
        if len(self.live_steps) % 5 == 0: # Update every 5 steps to save CPU
            self.redraw_charts()

    def run_rl_step(self):
        """Executes one step of the RL/SUMO loop (adapted from run_dashboard.py)"""
        if not self.waiting_for_action:
            if self.obs:
                actions = {}
                self.policy.next_step()
                for aid, o in self.obs.items():
                    truck = self.env.truck_agents[aid]
                    if getattr(truck, 'needs_pickup', False):
                        act = self.policy.pickup(f'truck_{aid}', candidates=list(range(45)), action_dim=45)
                    else:
                        sensor_features = getattr(truck, 'eng_obs', [])
                        act = self.policy.act(f'truck_{aid}', np.asarray(o, dtype=np.float32), sensor_features)
                    actions[int(aid)] = int(act)
                    rv = self.policy.get_last_rul(f'truck_{aid}')
                    if rv is not None:
                        try: self.env.truck_agents[aid].rul = float(rv)
                        except Exception: pass
                try:
                    self.env.gui_prepare_actions(actions)
                    self.waiting_for_action = True
                except Exception as e:
                    print(f"Env prepare error: {e}")
                    self.timer.stop()
            else:
                try:
                    self.env.producer.produce_step()
                    traci.simulationStep()
                    self.step_count += 1
                except Exception: pass
        else:
            try:
                ready = self.env.gui_tick_until_operable(max_sim_seconds=1.0)
                self.step_count += 1
            except Exception as e:
                print(f"Env tick error: {e}")
                self.timer.stop()
                return
            if ready:
                try:
                    new_obs, rewards, done = self.env.gui_finalize_step()
                    self.obs = new_obs
                    self.waiting_for_action = False
                except Exception as e:
                    print(f"Env finalize error: {e}")
                    self.timer.stop()

    def camera_track_logic(self):
        """Auto-follow the first truck"""
        if self.env.use_gui:
            try:
                views = list(traci.gui.getIDList())
                if views:
                    # Track truck_0 by default
                    traci.gui.trackVehicle(views[0], "truck_0")
            except Exception:
                pass

    def redraw_charts(self):
        # 1. Production Chart
        self.ax1.clear()
        # Plot Baseline (Full)
        b_steps, b_prod, _ = self.baseline.get_full_data()
        self.ax1.plot(b_steps, b_prod, color='gray', linestyle='--', alpha=0.6, label='Pretrained')
        # Plot Live
        self.ax1.plot(self.live_steps, self.live_prod, color='#39ff14', linewidth=2, label='Live')
        self.ax1.legend(facecolor='#2b2b2b', labelcolor='white', fontsize='small')
        self.ax1.set_title("Production", color='white')
        self.ax1.set_xlabel("Time (s)", color='white')
        self.ax1.grid(True, color='#444', linestyle=':', alpha=0.5)
        self.canvas1.draw()
        
        # 2. Profit Chart
        self.ax2.clear()
        # Plot Baseline (Full)
        b_steps, _, b_profit = self.baseline.get_full_data()
        self.ax2.plot(b_steps, b_profit, color='gray', linestyle='--', alpha=0.6, label='Pretrained')
        # Plot Live
        self.ax2.plot(self.live_steps, self.live_profit, color='#ffd700', linewidth=2, label='Live')
        self.ax2.legend(facecolor='#2b2b2b', labelcolor='white', fontsize='small')
        self.ax2.set_title("Profit", color='white')
        self.ax2.set_xlabel("Time (s)", color='white')
        self.ax2.grid(True, color='#444', linestyle=':', alpha=0.5)
        self.canvas2.draw()

def main():
    p = argparse.ArgumentParser(description='Run HUD Dashboard V2')
    p.add_argument('--num-agents', type=int, default=4)
    p.add_argument('--use-rul-agent', action='store_true', default=True)
    p.add_argument('--rul-threshold', type=float, default=7.0)
    p.add_argument('--rul-state', action='store_true', default=False)
    p.add_argument('--mqtt-host', default='127.0.0.1')
    p.add_argument('--mqtt-port', type=int, default=1883)
    p.add_argument('--mqtt-devices', default='edge-00,edge-01,edge-02,edge-03')
    p.add_argument('--mqtt-timeout-ms', type=int, default=1000)
    
    args = p.parse_args()
    
    env_args = EnvArgs(num_agents=args.num_agents, use_rul_agent=args.use_rul_agent,
                       rul_threshold=args.rul_threshold, rul_state=args.rul_state,
                       use_gui=True)
                       
    devices = tuple([d.strip() for d in args.mqtt_devices.split(',') if d.strip()])
    bridge_cfg = BridgeConfig(host=args.mqtt_host, port=args.mqtt_port, timeout_ms=args.mqtt_timeout_ms,
                              devices=devices, mode='mqtt')
    
    app = QApplication(sys.argv)
    
    # Set dark theme palette for Qt
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, Qt.white)
    app.setPalette(palette)
    
    dashboard = HUD_Dashboard(env_args, bridge_cfg)
    dashboard.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
