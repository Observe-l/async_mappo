#!/usr/bin/env python3
"""
SUMO RL Truck Scheduling MQTT Demo

- Headless or GUI demo that offloads action inference via MQTT.
- Keeps the original run_demo_schedule.py local demo untouched; use this file for MQTT runs.

Environment:
- Uses onpolicy.envs.rul_schedule.demo_schedule.async_scheduling
- For each operable agent, sends obs over MQTT and waits for (rul, action)

Usage examples:
  python3 scripts/render/run_demo_schedule_mqtt.py --mode debug --num-agents 4 \
      --max-steps 50 --mqtt mqtt --mqtt-devices edge-01,edge-02,edge-03,edge-04 --mqtt-timeout-ms 300

Clients:
- scripts/iot/edge_client.py --device-id edge-01 --rul-model <ts.pt> --actor-model <ts.pt>
"""
from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    import libsumo as traci
except Exception:
    import traci  # type: ignore

import sys
REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from onpolicy.envs.demo.mqtt_demo import async_scheduling
from onpolicy.iot.mqtt_bridge import MqttBridge, BridgeConfig


@dataclass
class EnvArgs:
    num_agents: int = 12
    use_rul_agent: bool = True
    rul_threshold: float = 7.0
    # For MQTT demo, do NOT include RUL in obs; the edge client computes it.
    rul_state: bool = False
    use_gui: bool = True
    exp_type: str = "rul_schedule_demo_mqtt"


class RemotePolicy:
    """Offloads action selection to remote devices via MQTT bridge."""
    def __init__(self, bridge: MqttBridge, action_dim: int, device_map: Optional[Dict[str, str]] = None):
        self.bridge = bridge
        self.action_dim = action_dim
        self._step_id = 0
        self.device_map = device_map or {}
        # Track last action per agent to allow safe repeat on timeout instead of random churn
        self._last_action: Dict[str, int] = {}
        # Track last predicted RUL per agent for GUI display
        self._last_rul: Dict[str, float] = {}

    def next_step(self):
        self._step_id += 1

    def act(self, agent_label: str, obs_vec: np.ndarray, sensor_features) -> int:
        """Request remote decision. sensor_features should be the truck.eng_obs list so edge can compute RUL.
        Fallback strategy on timeout/error: repeat previous action if available else random.
        NOTE: This server never loads a local RL actor or RUL predictor; ALL decisions must come from edge devices.
        """
        action: Optional[int] = None
        device_id: Optional[str] = None
        ok: bool = False
        try:
            if agent_label in self.device_map:
                device_id = self.device_map[agent_label]
                _rul, action, ok, device_id = self.bridge.request_to(device_id=device_id, step_id=self._step_id, agent_id=agent_label,
                                                                      env_obs=obs_vec, sensor=sensor_features, action_dim=self.action_dim)
            else:
                _rul, action, ok, device_id = self.bridge.request(step_id=self._step_id, agent_id=agent_label,
                                                                  env_obs=obs_vec, sensor=sensor_features, action_dim=self.action_dim)
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
            print(f"[SERVER] remote action step={self._step_id} agent={agent_label} device={device_id} action={act}")
            return act

        # Timeout or error -> fallback
        if agent_label in self._last_action:
            act = self._last_action[agent_label]
            print(f"[SERVER][TIMEOUT] step={self._step_id} agent={agent_label} device={device_id}; repeating last_action={act}")
            return act
        act = random.randrange(self.action_dim)
        self._last_action[agent_label] = act
        print(f"[SERVER][TIMEOUT] step={self._step_id} agent={agent_label} device={device_id}; random_fallback={act}")
        return act

    def get_last_rul(self, agent_label: str) -> Optional[float]:
        return self._last_rul.get(agent_label)

    def pickup(self, agent_label: str, candidates: Optional[list] = None, action_dim: int = 45) -> int:
        """Request a pickup destination selection from the edge client.
        Returns selected factory index in [0, action_dim).
        """
        action: Optional[int] = None
        device_id: Optional[str] = None
        ok: bool = False
        candidates = candidates if candidates is not None else list(range(action_dim))
        try:
            if agent_label in self.device_map:
                device_id = self.device_map[agent_label]
                _rul, action, ok, device_id = self.bridge.request_to(device_id=device_id, step_id=self._step_id, agent_id=agent_label,
                                                                      env_obs=[], sensor=[], action_dim=action_dim,
                                                                      decision_type="pickup", pickup_candidates=candidates)
            else:
                _rul, action, ok, device_id = self.bridge.request(step_id=self._step_id, agent_id=agent_label,
                                                                  env_obs=[], sensor=[], action_dim=action_dim,
                                                                  decision_type="pickup", pickup_candidates=candidates)
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
            print(f"[SERVER] remote pickup step={self._step_id} agent={agent_label} device={device_id} factory={act}")
            return act

        # Timeout or error -> fallback uniformly
        act = random.randrange(action_dim)
        self._last_action[agent_label] = act
        print(f"[SERVER][TIMEOUT] pickup step={self._step_id} agent={agent_label} device={device_id}; random_fallback={act}")
        return act


def run_demo_debug(env_args: EnvArgs, max_steps: int, bridge_cfg: BridgeConfig):
    env_args.use_gui = False
    env = async_scheduling(env_args)
    obs = env.reset()

    action_dim = env.factory_num if env.use_rul_agent else env.factory_num + 1
    bridge = MqttBridge(bridge_cfg)
    policy = RemotePolicy(bridge, action_dim)

    steps = 0
    try:
        while steps < max_steps:
            actions: Dict[int, int] = {}
            policy.next_step()
            for aid, o in obs.items():
                # Decide whether this is a pickup decision
                try:
                    truck = env.truck_agents[aid]
                except Exception:
                    truck = None
                if truck is not None and getattr(truck, 'needs_pickup', False):
                    # Pickup: select among raw material factories [0..44]
                    act = policy.pickup(f"truck_{aid}", candidates=list(range(45)), action_dim=45)
                    actions[int(aid)] = int(act)
                else:
                    # RL action with remote RUL
                    try:
                        sensor_features = getattr(truck, 'eng_obs', []) if truck is not None else []
                    except Exception:
                        sensor_features = []
                    act = policy.act(f"truck_{aid}", np.asarray(o, dtype=np.float32), sensor_features)
                    actions[int(aid)] = int(act)
            # Log decisions
            if actions:
                dec_parts = []
                for aid, act in actions.items():
                    target_label = f"Factory{act}" if env.use_rul_agent else ("MAINTAIN" if act == env.factory_num else f"Factory{act}")
                    dec_parts.append(f"agent{aid}->{target_label}")
                print("decisions: " + ", ".join(dec_parts))
            obs, _rew, done, _info = env.step(actions)
            steps += 1
            parts = [f"step={steps}"]
            try:
                sim_t = traci.simulation.getTime()
                parts.append(f"t={sim_t:.1f}s")
            except Exception:
                pass
            print(" | ".join(parts))
            if bool(np.all(done)):
                print("All trucks done; exiting.")
                break
    finally:
        try:
            traci.close()
        except Exception:
            pass


def run_demo_gui(env_args: EnvArgs, max_steps: int, bridge_cfg: BridgeConfig, step_interval_ms: int = 100):
    import tkinter as tk
    from tkinter import ttk

    env_args.use_gui = True
    env = async_scheduling(env_args)
    obs = env.reset()

    action_dim = env.factory_num if env.use_rul_agent else env.factory_num + 1
    bridge = MqttBridge(bridge_cfg)
    policy = RemotePolicy(bridge, action_dim)

    root = tk.Tk()
    root.title("SUMO RL Scheduling MQTT Demo")

    # Selection and camera follow
    tk.Label(root, text="Track truck:").grid(row=0, column=0, sticky="w")
    sel_var = tk.StringVar(value="truck_0")
    truck_ids = [f"truck_{i}" for i in range(env.truck_num)]
    sel_box = ttk.Combobox(root, textvariable=sel_var, values=truck_ids, state="readonly")
    sel_box.grid(row=0, column=1, sticky="ew")
    cam_follow_var = tk.BooleanVar(value=True)
    tk.Checkbutton(root, text="Camera follow", variable=cam_follow_var).grid(row=0, column=2, padx=8, sticky="w")

    # Tracking panel variables
    status_var = tk.StringVar(value="-")
    dist_var = tk.StringVar(value="0.0 m")
    dest_var = tk.StringVar(value="-")
    road_var = tk.StringVar(value="-")
    rul_var = tk.StringVar(value="-")
    cargo_var = tk.StringVar(value="-")

    def make_row(r, label, var):
        tk.Label(root, text=label).grid(row=r, column=0, sticky="w")
        tk.Label(root, textvariable=var).grid(row=r, column=1, sticky="w")

    make_row(1, "Status:", status_var)
    make_row(2, "Total distance:", dist_var)
    make_row(3, "Destination:", dest_var)
    make_row(4, "Road:", road_var)
    make_row(5, "RUL:", rul_var)
    make_row(6, "Cargo (prod, weight):", cargo_var)

    # Summary window
    summary = tk.Toplevel(root)
    summary.title("Summary Information")
    cols = ("Truck", "Status", "Distance(m)", "Destination", "Road")
    tree = ttk.Treeview(summary, columns=cols, show="headings", height=min(20, env.truck_num))
    for c in cols:
        tree.heading(c, text=c)
        tree.column(c, width=140, anchor="center")
    tree.pack(fill="both", expand=True)
    for i in range(env.truck_num):
        tree.insert("", "end", iid=f"truck_{i}", values=(f"truck_{i}", "-", "0.0", "-", "-"))

    def update_camera():
        if env.use_gui and cam_follow_var.get():
            try:
                views = list(traci.gui.getIDList())
                if views:
                    traci.gui.trackVehicle(views[0], sel_var.get())
            except Exception:
                pass

    sel_box.bind("<<ComboboxSelected>>", lambda e: update_camera())

    steps = 0
    state = {"waiting": False}
    last_gui_update = {"t": 0.0}

    def refresh_panels(force: bool = False):
        """Update tracking & summary panels even while waiting for next step (for live distance/RUL updates)."""
        now = time.time()
        if not force and (now - last_gui_update["t"]) < 0.5:  # throttle 2 Hz
            return
        last_gui_update["t"] = now
        try:
            tid = sel_var.get()
            idx = int(tid.split("_")[-1])
            t = env.truck_agents[idx]
            status_var.set(t.state)
            dist_var.set(f"{t.total_distance:.1f} m")
            dest_var.set(t.destination)
            rul_val = getattr(t, 'rul', 0.0)
            rul_var.set(f"{rul_val:.1f}")
            cargo_var.set(f"{t.product}, {t.weight:.1f}")
            try:
                road_id = traci.vehicle.getRoadID(tid)
            except Exception:
                road_id = "-"
            road_var.set(road_id)
            for i in range(env.truck_num):
                vid = f"truck_{i}"
                ti = env.truck_agents[i]
                try:
                    rid = traci.vehicle.getRoadID(vid)
                except Exception:
                    rid = "-"
                tree.item(vid, values=(vid, ti.state, f"{ti.total_distance:.1f}", ti.destination, rid))
        except Exception:
            pass

    def step_loop():
        nonlocal obs, steps
        if steps >= max_steps:
            finish("Max steps reached")
            return
        if not state["waiting"]:
            if obs:
                actions: Dict[int, int] = {}
                policy.next_step()
                for aid, o in obs.items():
                    try:
                        truck = env.truck_agents[aid]
                    except Exception:
                        truck = None
                    if truck is not None and getattr(truck, 'needs_pickup', False):
                        act = policy.pickup(f"truck_{aid}", candidates=list(range(45)), action_dim=45)
                        actions[int(aid)] = int(act)
                    else:
                        try:
                            sensor_features = getattr(truck, 'eng_obs', []) if truck is not None else []
                        except Exception:
                            sensor_features = []
                        act = policy.act(f"truck_{aid}", np.asarray(o, dtype=np.float32), sensor_features)
                        actions[int(aid)] = int(act)
                if actions:
                    dec_parts = []
                    for aid, act in actions.items():
                        target_label = f"Factory{act}" if env.use_rul_agent else ("MAINTAIN" if act == env.factory_num else f"Factory{act}")
                        dec_parts.append(f"agent{aid}->{target_label}")
                    print("decisions: " + ", ".join(dec_parts))
                # Update trucks' RUL from edge predictions for GUI purposes
                try:
                    for aid in actions.keys():
                        lab = f"truck_{aid}"
                        rul_val = policy.get_last_rul(lab)
                        if rul_val is not None:
                            env.truck_agents[aid].rul = float(rul_val)
                except Exception:
                    pass
                try:
                    env.gui_prepare_actions(actions)
                except Exception as e:
                    print(f"Env prepare error: {e}")
                    finish("Env prepare error")
                    return
                state["waiting"] = True
            else:
                # No operable trucks yet. Advance SUMO minimally so trucks can become operable.
                try:
                    env.producer.produce_step()
                    traci.simulationStep()
                except Exception:
                    pass
                refresh_panels()  # show evolving distances even before first decision
        else:
            try:
                ready = env.gui_tick_until_operable(max_sim_seconds=30.0)
            except Exception as e:
                print(f"Env tick error: {e}")
                finish("Env tick error")
                return
            if ready:
                try:
                    obs, _rew, done = env.gui_finalize_step()
                except Exception as e:
                    print(f"Env finalize error: {e}")
                    finish("Env finalize error")
                    return
                steps += 1
                state["waiting"] = False
                refresh_panels(force=True)
                # Logging
                parts = [f"step={steps}"]
                try:
                    sim_t = traci.simulation.getTime()
                    parts.append(f"t={sim_t:.1f}s")
                except Exception:
                    pass
                print(" | ".join(parts))
                if bool(np.all(done)):
                    finish("All trucks done")
                    return
        # Periodic passive refresh while waiting
        refresh_panels()
        root.after(step_interval_ms, step_loop)

    def finish(msg: str):
        print(msg + "; closing.")
        try:
            traci.close()
        except Exception:
            pass
        try:
            root.destroy()
        except Exception:
            pass

    update_camera()
    # Initial population
    refresh_panels(force=True)
    root.after(step_interval_ms, step_loop)
    try:
        root.mainloop()
    finally:
        try:
            traci.close()
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser(description="Run SUMO RL Scheduling MQTT Demo (async_scheduling)")
    p.add_argument("--mode", choices=["debug", "gui"], default="debug")
    p.add_argument("--num-agents", type=int, default=4)
    p.add_argument("--use-rul-agent", action="store_true", default=True)
    p.add_argument("--rul-threshold", type=float, default=7.0)
    # Default False so edge client computes RUL
    p.add_argument("--rul-state", action="store_true", default=False)
    p.add_argument("--max-steps", type=int, default=1000)
    # MQTT
    p.add_argument("--mqtt-host", default="127.0.0.1")
    p.add_argument("--mqtt-port", type=int, default=1883)
    p.add_argument("--mqtt-devices", default="edge-00,edge-01,edge-02,edge-03")
    p.add_argument("--mqtt-timeout-ms", type=int, default=300)
    args = p.parse_args()

    env_args = EnvArgs(
        num_agents=args.num_agents,
        use_rul_agent=args.use_rul_agent,
        rul_threshold=args.rul_threshold,
        rul_state=args.rul_state,
        use_gui=(args.mode == "gui"),
    )

    devices = tuple([d.strip() for d in args.mqtt_devices.split(',') if d.strip()])
    bridge_cfg = BridgeConfig(host=args.mqtt_host, port=args.mqtt_port, timeout_ms=args.mqtt_timeout_ms, devices=devices, mode="mqtt")

    if args.mode == "gui":
        run_demo_gui(env_args, args.max_steps, bridge_cfg)
    else:
        run_demo_debug(env_args, args.max_steps, bridge_cfg)


if __name__ == "__main__":
    main()
