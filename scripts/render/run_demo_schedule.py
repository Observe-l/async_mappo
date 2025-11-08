#!/usr/bin/env python3
"""
SUMO RL Truck Scheduling Demo runner (based on async_scheduling)

- Starts SUMO (GUI or headless)
- Instantiates onpolicy.envs.rul_schedule.demo_schedule.async_scheduling
- Loads a trained RL policy from --actor-dir if available (actor.pt)
- Runs a control loop that sends actions when trucks are operable
- GUI mode: opens Tkinter windows (Agent Tracking + Summary) updating live
  from env.truck_agents and the SUMO GUI camera follows the selected truck.

This script focuses on the display/visualization and minimal policy inference.
If the actor cannot be loaded or has mismatched shapes, it falls back to random.
"""
from __future__ import annotations

import argparse
import os
import queue
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    import torch
    from torch import nn
except Exception:
    torch = None

# Prefer libsumo if available; otherwise fall back to traci (pure-Python)
try:
    import libsumo as traci
except Exception:
    import traci  # type: ignore

# Ensure repo root on path if needed
import sys
REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from onpolicy.envs.rul_schedule.demo_schedule import async_scheduling


# --------------------------- Arg container ---------------------------

@dataclass
class EnvArgs:
    num_agents: int = 12
    use_rul_agent: bool = True
    rul_threshold: float = 7.0
    rul_state: bool = True
    use_gui: bool = True
    exp_type: str = "rul_schedule_demo"


# --------------------------- Policy wrapper ---------------------------

class TrainedPolicy:
    """Wraps the official R_MAPPO actor architecture and loads its state dict.

    Falls back to random actions if loading fails or torch unavailable.
    """
    def __init__(self, actor_dir: Optional[str], env_obs_space, env_share_obs_space, env_act_space,
                 use_recurrent_policy: bool = True, recurrent_N: int = 6):
        self.device = torch.device("cpu") if torch else None
        self.valid = False
        self._loaded_from = None
        self.actor = None  # Will become R_Actor instance
        self.args = None
        if torch is None:
            print("Torch not available; using random actions.")
            return
        try:
            from onpolicy.config import get_config
            from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
            parser = get_config()
            # Use default parser (no CLI args) then override needed fields
            self.args = parser.parse_args([])
            # Provide missing attributes expected by R_Actor/ACTLayer when not in base config
            if not hasattr(self.args, 'grid_goal'):
                self.args.grid_goal = False
            if not hasattr(self.args, 'goal_grid_size'):
                self.args.goal_grid_size = 4
            # Ensure consistency with training
            self.args.algorithm_name = 'mappo'
            self.args.use_recurrent_policy = bool(use_recurrent_policy)
            self.args.use_naive_recurrent_policy = False
            self.args.recurrent_N = int(recurrent_N)
            # Build actor network structure matching training code
            self.actor = R_Actor(self.args, env_obs_space, env_act_space, device=self.device)
            if actor_dir:
                state_path = Path(actor_dir) / 'actor.pt'
                if state_path.exists():
                    state_dict = torch.load(str(state_path), map_location=self.device)
                    try:
                        self.actor.load_state_dict(state_dict)
                        self.valid = True
                        self._loaded_from = str(state_path)
                        print(f"Loaded actor state dict from {state_path}")
                    except Exception as e:
                        print(f"Failed to load actor state dict ({e}); using random actions.")
                else:
                    print(f"actor.pt not found in {actor_dir}; using random actions.")
        except Exception as e:
            print(f"Policy init error: {e}; using random actions.")
            self.actor = None

    def act(self, obs_vec: np.ndarray) -> int:
        if self.actor is None or torch is None:
            # random discrete action size derived from env_act_space
            return random.randrange(self._action_dim())
        try:
            # Prepare tensors
            obs_tensor = torch.as_tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
            # rnn_states & masks even if not used (shape compatibility)
            rnn_states = torch.zeros((1, self.args.recurrent_N, self.args.hidden_size), dtype=torch.float32)
            masks = torch.ones((1, 1), dtype=torch.float32)
            with torch.no_grad():
                actions, _log_probs, _rnn = self.actor(obs_tensor, rnn_states, masks, available_actions=None, deterministic=False)
            # actions shape (1,1) for Discrete
            return int(actions.view(-1)[0].item())
        except Exception as e:
            print(f"Inference error ({e}); returning random action.")
            return random.randrange(self._action_dim())

    def _action_dim(self) -> int:
        # Infer from last linear layer out features if loaded, else attempt env action space length
        try:
            return int(self.actor.act.action_out.linear.out_features)
        except Exception:
            return 1


# ------------------------- GUI/Display thread -------------------------

def start_gui_windows(env: async_scheduling, refresh_hz: float = 5.0):
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Agent Tracking Window")

    # Selection
    tk.Label(root, text="Track truck:").grid(row=0, column=0, sticky="w")
    sel_var = tk.StringVar(value="truck_0")
    truck_ids = [f"truck_{i}" for i in range(env.truck_num)]
    sel_box = ttk.Combobox(root, textvariable=sel_var, values=truck_ids, state="readonly")
    sel_box.grid(row=0, column=1, sticky="ew")

    # Info labels
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
    summary.title("Summary Information Window")
    cols = ("Truck", "Status", "Distance(m)", "Destination", "Road")
    tree = ttk.Treeview(summary, columns=cols, show="headings", height=min(20, env.truck_num))
    for c in cols:
        tree.heading(c, text=c)
        tree.column(c, width=140, anchor="center")
    tree.pack(fill="both", expand=True)
    for i in range(env.truck_num):
        tree.insert("", "end", iid=f"truck_{i}", values=(f"truck_{i}", "-", "0.0", "-", "-"))

    # Camera follow
    def update_camera():
        if env.use_gui:
            try:
                views = list(traci.gui.getIDList())
                if views:
                    traci.gui.trackVehicle(views[0], sel_var.get())
            except Exception:
                pass

    sel_box.bind("<<ComboboxSelected>>", lambda e: update_camera())

    # Periodic refresh
    period_ms = int(1000.0 / max(1.0, refresh_hz))

    def refresh():
        tid = sel_var.get()
        # Update tracking panel
        try:
            idx = int(tid.split("_")[-1])
            truck = env.truck_agents[idx]
            status_var.set(truck.state)
            dist_var.set(f"{truck.total_distance:.1f} m")
            dest_var.set(truck.destination)
            rul_var.set(f"{getattr(truck, 'rul', 0):.1f}")
            cargo_var.set(f"{truck.product}, {truck.weight:.1f}")
            try:
                road_id = traci.vehicle.getRoadID(tid)
            except Exception:
                road_id = "-"
            road_var.set(road_id)
        except Exception:
            pass
        # Update summary table
        for i in range(env.truck_num):
            vid = f"truck_{i}"
            t = env.truck_agents[i]
            try:
                road_id = traci.vehicle.getRoadID(vid)
            except Exception:
                road_id = "-"
            tree.item(vid, values=(vid, t.state, f"{t.total_distance:.1f}", t.destination, road_id))
        root.after(period_ms, refresh)

    update_camera()
    refresh()
    root.mainloop()


# ---------------------------- Demo runner ----------------------------

def simulation_thread(stop_evt: threading.Event, step_hz: float = 20.0):
    """Deprecated: avoid running libsumo steps from multiple threads.
    We keep the function for reference but do not use it to prevent crashes.
    """
    interval = 1.0 / max(1.0, step_hz)
    while not stop_evt.is_set():
        time.sleep(interval)


def run_demo_debug(sumo_cfg: str, actor_dir: Optional[str], env_args: EnvArgs, max_steps: int, debug: bool,
                   use_recurrent_policy: bool, recurrent_N: int):
    # Start SUMO inside env init; avoid extra close/start here to prevent libsumo instability
    # binary = "sumo-gui" if (mode == "gui") else "sumo"
    # cmd = [binary, "-c", sumo_cfg, "--no-warnings", "true"]
    # print(f"Starting {binary} with cfg: {sumo_cfg}")
    # traci.start(cmd)

    # Build env (debug/headless mode enforced)
    env_args.use_gui = False
    env = async_scheduling(env_args)
    obs = env.reset()

    # Action space size: factory_num(+1 for maintain if not use_rul_agent)
    action_dim = env.factory_num if env.use_rul_agent else env.factory_num + 1
    # Instantiate trained policy (R_MAPPO actor architecture)
    policy = TrainedPolicy(actor_dir=actor_dir,
                           env_obs_space=env.observation_space[0],
                           env_share_obs_space=env.share_observation_space[0],
                           env_act_space=env.action_space[0],
                           use_recurrent_policy=use_recurrent_policy,
                           recurrent_N=recurrent_N)

    steps = 0
    try:
        while steps < max_steps:
            # Build action dict for currently operable agents (keys in obs)
            actions: Dict[int, int] = {}
            for aid, o in obs.items():
                act = policy.act(np.asarray(o, dtype=np.float32))
                actions[int(aid)] = int(act)
            # Debug: show new RL decisions before environment step
            if debug:
                dec_parts = []
                for aid, act in actions.items():
                    if env.use_rul_agent:
                        target_label = f"Factory{act}"
                    else:
                        target_label = "MAINTAIN" if act == env.factory_num else f"Factory{act}"
                    dec_parts.append(f"agent{aid}->{target_label}")
                if dec_parts:
                    print("decisions: " + ", ".join(dec_parts))
            # Step environment logical time; SUMO calls are within env
            obs, rew, done, _info = env.step(actions)
            steps += 1
            if debug:
                parts = [f"step={steps}"]
                try:
                    sim_t = traci.simulation.getTime()
                    parts.append(f"t={sim_t:.1f}s")
                except Exception:
                    pass
                for i in range(env.truck_num):
                    vid = f"truck_{i}"
                    t = env.truck_agents[i]
                    try:
                        spd = traci.vehicle.getSpeed(vid)
                        road = traci.vehicle.getRoadID(vid)
                    except Exception:
                        spd, road = 0.0, "-"
                    parts.append(f"{vid}:{t.state}@{t.destination} d={t.total_distance:.1f} v={spd:.2f} road={road}")
                print(" | ".join(parts))
            if bool(np.all(done)):
                print("All trucks done; exiting.")
                break
    finally:
        try:
            traci.close()
        except Exception:
            pass


def run_demo_gui(sumo_cfg: str, actor_dir: Optional[str], env_args: EnvArgs, max_steps: int, debug: bool,
                 use_recurrent_policy: bool, recurrent_N: int, step_interval_ms: int = 100):
    """Run the GUI mode with Tkinter mainloop on the main thread to avoid segmentation faults.

    We integrate the environment stepping inside Tk's after() callback so that all traci/libsumo
    calls occur from the main thread. This prevents thread-safety issues seen when using a
    background while-loop plus a Tkinter thread.
    """
    import tkinter as tk
    from tkinter import ttk

    env_args.use_gui = True
    env = async_scheduling(env_args)
    obs = env.reset()

    policy = TrainedPolicy(actor_dir=actor_dir,
                           env_obs_space=env.observation_space[0],
                           env_share_obs_space=env.share_observation_space[0],
                           env_act_space=env.action_space[0],
                           use_recurrent_policy=use_recurrent_policy,
                           recurrent_N=recurrent_N)

    root = tk.Tk()
    root.title("SUMO RL Scheduling Demo")

    # Selection Combo
    tk.Label(root, text="Track truck:").grid(row=0, column=0, sticky="w")
    sel_var = tk.StringVar(value="truck_0")
    truck_ids = [f"truck_{i}" for i in range(env.truck_num)]
    sel_box = ttk.Combobox(root, textvariable=sel_var, values=truck_ids, state="readonly")
    sel_box.grid(row=0, column=1, sticky="ew")

    # Camera follow toggle
    cam_follow_var = tk.BooleanVar(value=True)
    tk.Checkbutton(root, text="Camera follow", variable=cam_follow_var, command=lambda: update_camera()).grid(row=0, column=2, padx=8, sticky="w")

    # Info vars
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

    # Summary Tree
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
    state = {"waiting": False}  # GUI two-phase loop state

    # Only refresh GUI info/camera every N seconds of SUMO simulation time
    GUI_UPDATE_INTERVAL_S = 50.0
    next_gui_update_time = 0.0

    def refresh_info_panels():
        """Update the detailed panel and summary table once."""
        try:
            tid = sel_var.get()
            idx = int(tid.split("_")[-1])
            truck = env.truck_agents[idx]
            status_var.set(truck.state)
            dist_var.set(f"{truck.total_distance:.1f} m")
            dest_var.set(truck.destination)
            rul_var.set(f"{getattr(truck, 'rul', 0):.1f}")
            cargo_var.set(f"{truck.product}, {truck.weight:.1f}")
            try:
                road_id = traci.vehicle.getRoadID(tid)
            except Exception:
                road_id = "-"
            road_var.set(road_id)
        except Exception:
            pass
        for i in range(env.truck_num):
            vid = f"truck_{i}"
            t = env.truck_agents[i]
            try:
                road_id = traci.vehicle.getRoadID(vid)
            except Exception:
                road_id = "-"
            tree.item(vid, values=(vid, t.state, f"{t.total_distance:.1f}", t.destination, road_id))

    def step_loop():
        nonlocal obs, steps, next_gui_update_time
        if steps >= max_steps:
            finish("Max steps reached")
            return
        # Phase 1: issue actions when trucks are operable (obs non-empty) and not waiting
        if not state["waiting"]:
            if obs:  # Only act when there are operable trucks
                actions: Dict[int, int] = {}
                for aid, o in obs.items():
                    act = policy.act(np.asarray(o, dtype=np.float32))
                    actions[int(aid)] = int(act)
                if debug and actions:
                    dec_parts = []
                    for aid, act in actions.items():
                        if env.use_rul_agent:
                            target_label = f"Factory{act}"
                        else:
                            if act == env.factory_num:
                                target_label = "MAINTAIN"
                            else:
                                target_label = f"Factory{act}"
                        dec_parts.append(f"agent{aid}->{target_label}")
                    if dec_parts:
                        print("decisions: " + ", ".join(dec_parts))
                try:
                    env.gui_prepare_actions(actions)
                except Exception as e:
                    print(f"Env prepare error: {e}")
                    finish("Env prepare error")
                    return
                state["waiting"] = True
        else:
            # Phase 2: advance simulation incrementally until next decision point
            try:
                    # Advance up to a larger chunk of SUMO time (e.g., 30s) for faster progression.
                    ready = env.gui_tick_until_operable(max_sim_seconds=30.0)
            except Exception as e:
                print(f"Env tick error: {e}")
                finish("Env tick error")
                return
            if ready:
                try:
                    obs, rew, done = env.gui_finalize_step()
                except Exception as e:
                    print(f"Env finalize error: {e}")
                    finish("Env finalize error")
                    return
                steps += 1
                state["waiting"] = False
                if debug:
                    parts = [f"step={steps}"]
                    try:
                        sim_t_dbg = traci.simulation.getTime()
                        parts.append(f"t={sim_t_dbg:.1f}s")
                    except Exception:
                        pass
                    for i in range(env.truck_num):
                        vid = f"truck_{i}"
                        t_agent = env.truck_agents[i]
                        try:
                            spd = traci.vehicle.getSpeed(vid)
                            road = traci.vehicle.getRoadID(vid)
                        except Exception:
                            spd, road = 0.0, "-"
                        parts.append(f"{vid}:{t_agent.state}@{t_agent.destination} d={t_agent.total_distance:.1f} v={spd:.2f} road={road}")
                    print(" | ".join(parts))
                if bool(np.all(done)):
                    finish("All trucks done")
                    return
        # Periodic GUI refresh (every 50s SUMO time)
        try:
            sim_t = traci.simulation.getTime()
        except Exception:
            sim_t = 0.0
        if next_gui_update_time <= 0.0:
            next_gui_update_time = ((sim_t // GUI_UPDATE_INTERVAL_S) + 1) * GUI_UPDATE_INTERVAL_S
        if sim_t >= next_gui_update_time:
            refresh_info_panels()
            update_camera()
            next_gui_update_time += GUI_UPDATE_INTERVAL_S
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

    # Kick off loop
    refresh_info_panels()
    update_camera()
    root.after(step_interval_ms, step_loop)
    try:
        root.mainloop()
    finally:
        # Ensure SUMO closed if GUI window closed early
        try:
            traci.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Run SUMO RL Scheduling Demo (async_scheduling)")
    parser.add_argument("--mode", choices=["debug", "gui"], default="gui")
    parser.add_argument("--sumo-cfg", default=os.environ.get("SUMO_CFG", "/home/lwh/Documents/Code/RL-Scheduling/map/sg_map/osm.sumocfg"))
    parser.add_argument("--actor-dir", default="/home/lwh/Documents/Code/results/async_schedule/rul_schedule/mappo/threshold_7/wandb/run-20250503_002045-r5psc472/files")
    parser.add_argument("--num-agents", type=int, default=12)
    parser.add_argument("--use-rul-agent", action="store_true", default=True)
    parser.add_argument("--rul-threshold", type=float, default=7.0)
    parser.add_argument("--rul-state",default=True, action="store_true")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--debug",default=True, action="store_true")
    parser.add_argument("--use-recurrent-policy", action="store_true", default=True)
    parser.add_argument("--recurrent_N", type=int, default=6)
    args = parser.parse_args()

    env_args = EnvArgs(
        num_agents=args.num_agents,
        use_rul_agent=args.use_rul_agent,
        rul_threshold=args.rul_threshold,
        rul_state=args.rul_state,
        use_gui=(args.mode == "gui"),
    )

    if args.mode == "gui":
        run_demo_gui(
            sumo_cfg=args.sumo_cfg,
            actor_dir=args.actor_dir,
            env_args=env_args,
            max_steps=args.max_steps,
            debug=args.debug,
            use_recurrent_policy=args.use_recurrent_policy,
            recurrent_N=args.recurrent_N
        )
    else:
        run_demo_debug(
            sumo_cfg=args.sumo_cfg,
            actor_dir=args.actor_dir,
            env_args=env_args,
            max_steps=args.max_steps,
            debug=args.debug or True,
            use_recurrent_policy=args.use_recurrent_policy,
            recurrent_N=args.recurrent_N
        )


if __name__ == "__main__":
    main()
