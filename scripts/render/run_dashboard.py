#!/usr/bin/env python3
"""
RL Logistics Real-time Dashboard
Based on PyQt5 + Matplotlib + SUMO
"""
from __future__ import annotations

import sys
import argparse
import random
import time
import datetime
import numpy as np
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
    exp_type: str = "rul_schedule_demo_dashboard"

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
# Dashboard Class
# -----------------------------------------------------------------------------

class CyberDashboard(QMainWindow):
    def __init__(self, env_args: EnvArgs, bridge_cfg: BridgeConfig):
        super().__init__()
        self.env_args = env_args
        self.bridge_cfg = bridge_cfg
        
        # Initialize Data
        self.init_data_containers()
        
        # Initialize UI
        self.init_ui()
        
        # Initialize Environment (SUMO)
        self.init_env()
        
        # Main Loop Timer
        self.timer = QTimer()
        self.timer.setInterval(50) # 50ms = 20 FPS
        self.timer.timeout.connect(self.game_loop)
        self.timer.start()

    def init_data_containers(self):
        """Use deques for efficient sliding window data"""
        self.history_len = 200
        self.steps_data = deque(maxlen=self.history_len)
        self.production_data = deque(maxlen=self.history_len)
        self.profit_data = deque(maxlen=self.history_len)
        self.reward_data = deque(maxlen=self.history_len)
        
        # Metric Accumulators
        self.total_prod = 0
        self.total_profit = 0.0
        self.cum_reward = 0.0
        
        # RL State
        self.obs = None
        self.waiting_for_action = False
        self.dist_tracker = DistanceTracker()

    def init_ui(self):
        """Setup Layout, Style, and Matplotlib Figures"""
        self.setWindowTitle("Async-MAPPO Logistics Monitor")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #1e1e1e; color: white;")
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # --- Zone A: KPI Cards ---
        kpi_layout = QHBoxLayout()
        
        self.card_prod = self.create_kpi_card("TOTAL PRODUCTION", "0", "#39ff14") # Neon Green
        self.card_profit = self.create_kpi_card("TOTAL PROFIT", "$0", "#ffd700") # Gold
        self.card_reward = self.create_kpi_card("CURRENT REWARD", "0.0", "#00ffff") # Cyan
        
        kpi_layout.addWidget(self.card_prod)
        kpi_layout.addWidget(self.card_profit)
        kpi_layout.addWidget(self.card_reward)
        
        main_layout.addLayout(kpi_layout)
        
        # --- Zone B: Charts ---
        charts_layout = QHBoxLayout()
        
        # Chart 1: Production
        self.fig1 = Figure(figsize=(4, 3), dpi=100)
        self.canvas1 = FigureCanvas(self.fig1)
        self.ax1 = self.fig1.add_subplot(111)
        self.setup_chart(self.fig1, self.ax1, "Production Rate", "#39ff14")
        
        # Chart 2: Profit
        self.fig2 = Figure(figsize=(4, 3), dpi=100)
        self.canvas2 = FigureCanvas(self.fig2)
        self.ax2 = self.fig2.add_subplot(111)
        self.setup_chart(self.fig2, self.ax2, "Financial Health", "#ffd700")
        
        # Chart 3: Reward
        self.fig3 = Figure(figsize=(4, 3), dpi=100)
        self.canvas3 = FigureCanvas(self.fig3)
        self.ax3 = self.fig3.add_subplot(111)
        self.setup_chart(self.fig3, self.ax3, "Agent Intelligence (Reward)", "#00ffff")
        
        charts_layout.addWidget(self.canvas1)
        charts_layout.addWidget(self.canvas2)
        charts_layout.addWidget(self.canvas3)
        
        main_layout.addLayout(charts_layout, stretch=1)
        
        # --- Zone C: Control & Status ---
        status_layout = QHBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 3600) # Assuming 1 hour sim
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #00ffff; }")
        
        self.status_label = QLabel("Status: Initializing...")
        self.status_label.setFont(QFont("Arial", 12))
        
        status_layout.addWidget(QLabel("Progress:"))
        status_layout.addWidget(self.progress_bar)
        status_layout.addWidget(self.status_label)
        
        main_layout.addLayout(status_layout)

    def create_kpi_card(self, title, value, color):
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet(f"border: 2px solid {color}; border-radius: 10px; background-color: #2b2b2b;")
        layout = QVBoxLayout(frame)
        
        lbl_title = QLabel(title)
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet("color: #aaaaaa; font-size: 14px;")
        
        lbl_value = QLabel(value)
        lbl_value.setAlignment(Qt.AlignCenter)
        lbl_value.setStyleSheet(f"color: {color}; font-size: 32px; font-weight: bold;")
        lbl_value.setObjectName("value_label") # Tag for easy update
        
        layout.addWidget(lbl_title)
        layout.addWidget(lbl_value)
        return frame

    def update_kpi_card(self, card, value):
        lbl = card.findChild(QLabel, "value_label")
        if lbl:
            lbl.setText(str(value))

    def setup_chart(self, fig, ax, title, color):
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#1e1e1e')
        ax.set_title(title, color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        self.lines = {} # Store line references if needed, but simple plot is fine for now

    def init_env(self):
        """Initialize SUMO and RL Environment"""
        self.env = async_scheduling(self.env_args)
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
        self.status_label.setText("Status: Running")

    def game_loop(self):
        """
        The Core Function:
        1. Step SUMO
        2. Get RL Data
        3. Update UI
        """
        if traci.simulation.getMinExpectedNumber() <= 0:
            self.status_label.setText("Status: Completed")
            self.timer.stop()
            return

        # Logic adapted from run_demo_gui step_loop
        if not self.waiting_for_action:
            if self.obs:
                # We have observations, need to take actions
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
                    
                    # Update RUL from edge
                    rv = self.policy.get_last_rul(f'truck_{aid}')
                    if rv is not None:
                        try:
                            self.env.truck_agents[aid].rul = float(rv)
                        except Exception:
                            pass
                
                try:
                    self.env.gui_prepare_actions(actions)
                    self.waiting_for_action = True
                except Exception as e:
                    print(f"Env prepare error: {e}")
                    self.timer.stop()
                    return
            else:
                # No obs, just step
                try:
                    self.env.producer.produce_step()
                    traci.simulationStep()
                except Exception:
                    pass
        else:
            # Waiting for trucks to become operable
            try:
                # Tick a bit
                ready = self.env.gui_tick_until_operable(max_sim_seconds=1.0) # Small step for GUI responsiveness
            except Exception as e:
                print(f"Env tick error: {e}")
                self.timer.stop()
                return
            
            if ready:
                try:
                    new_obs, rewards, done = self.env.gui_finalize_step()
                    self.obs = new_obs
                    self.waiting_for_action = False
                    
                    # Update Cumulative Reward
                    step_reward = np.sum(rewards)
                    self.cum_reward += step_reward
                    
                except Exception as e:
                    print(f"Env finalize error: {e}")
                    self.timer.stop()
                    return

        # Update Metrics & UI
        self.update_metrics()
        self.refresh_charts()
        
        # Update Progress
        try:
            sim_t = traci.simulation.getTime()
            self.progress_bar.setValue(int(sim_t))
        except Exception:
            pass

    def update_metrics(self):
        # 1. Total Production
        # Sum of total_final_product from all final factories
        final_factories = ["Factory45", "Factory46", "Factory47", "Factory48", "Factory49"]
        current_prod = 0
        for fid in final_factories:
            if fid in self.env.factory:
                current_prod += self.env.factory[fid].total_final_product
        self.total_prod = current_prod
        
        # 2. Total Profit
        # Profit = (Revenue per Unit * P_total) - (Fuel Cost * Total Distance)
        revenue = 100 * self.total_prod
        
        total_distance_m = 0.0
        for i in range(self.env.truck_num):
            vid = f'truck_{i}'
            total_distance_m += self.dist_tracker.total_m(vid)
            
        cost = 0.1 * total_distance_m
        self.total_profit = revenue - cost
        
        # 3. Update Deques
        self.steps_data.append(self.env.episode_len)
        self.production_data.append(self.total_prod)
        self.profit_data.append(self.total_profit)
        self.reward_data.append(self.cum_reward)
        
        # 4. Update KPI Cards
        self.update_kpi_card(self.card_prod, str(int(self.total_prod)))
        self.update_kpi_card(self.card_profit, f"${self.total_profit:,.2f}")
        self.update_kpi_card(self.card_reward, f"{self.cum_reward:.2f}")

    def refresh_charts(self):
        # Helper to plot
        def plot_data(ax, canvas, x, y, color):
            ax.clear()
            ax.plot(x, y, color=color, linewidth=2)
            ax.set_facecolor('#1e1e1e')
            ax.grid(True, color='#444444', linestyle='--', alpha=0.5)
            canvas.draw()

        if len(self.steps_data) > 1:
            x = list(self.steps_data)
            
            # Chart 1: Production
            plot_data(self.ax1, self.canvas1, x, list(self.production_data), "#39ff14")
            self.ax1.set_title("Production Rate", color='white')
            
            # Chart 2: Profit
            plot_data(self.ax2, self.canvas2, x, list(self.profit_data), "#ffd700")
            self.ax2.set_title("Financial Health", color='white')
            
            # Chart 3: Reward
            plot_data(self.ax3, self.canvas3, x, list(self.reward_data), "#00ffff")
            self.ax3.set_title("Agent Intelligence (Reward)", color='white')

def main():
    p = argparse.ArgumentParser(description='Run SUMO RL Dashboard')
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
    
    dashboard = CyberDashboard(env_args, bridge_cfg)
    dashboard.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
