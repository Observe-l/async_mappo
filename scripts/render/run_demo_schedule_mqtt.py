#!/usr/bin/env python3
"""
SUMO RL Truck Scheduling MQTT Demo with MySQL logging.

Adds per-run MySQL database (sumo_YYYYMMDD_HHMMSS) containing one table per truck.
Each row captures: sim_time (UNIX seconds = 2025-11-01 00:00:00 + simulation seconds),
rul, driving_distance_km, state, destination, loaded_goods, weight, total_transported.
Rows inserted on every RL/pickup decision and every 200 seconds of SUMO time.
"""
from __future__ import annotations

import argparse
import random
import time
import datetime
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
try:
    from onpolicy.utils.mysql_logger import MySQLRunLogger  # type: ignore
    _MYSQL_IMPORT_OK = True
except Exception as _e:
    MySQLRunLogger = None  # type: ignore
    _MYSQL_IMPORT_OK = False
    _MYSQL_IMPORT_ERR = str(_e)


@dataclass
class EnvArgs:
    num_agents: int = 12
    use_rul_agent: bool = True
    rul_threshold: float = 7.0
    rul_state: bool = False  # Edge predicts RUL
    use_gui: bool = True
    exp_type: str = "rul_schedule_demo_mqtt"


class DistanceTracker:
    """Tracks per-vehicle total distance from SUMO, robust to vehicle delete/recreate resets.

    Uses traci.vehicle.getDistance(vehID) which may reset to 0 on recreate.
    We maintain an offset per vehicle so that total remains monotonic and cumulative.
    """
    def __init__(self):
        self._prev: Dict[str, float] = {}
        self._offset: Dict[str, float] = {}

    def total_m(self, veh_id: str) -> float:
        """Return robust cumulative distance in meters.

        Rules:
        - If current reading is invalid (negative or absurd), ignore update and return last total.
        - If current drops below previous but remains valid (>=0), treat as reset and add previous to offset.
        """
        try:
            cur = float(traci.vehicle.getDistance(veh_id))
        except Exception:
            cur = None
        prev = self._prev.get(veh_id)
        offset = self._offset.get(veh_id, 0.0)
        # Treat negative or sentinel values as invalid (SUMO may return ~-1.073741824e9)
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


def _collect_truck_row(truck_id: str, truck, decision: bool, dist_tracker: DistanceTracker):
    try:
        sim_t = traci.simulation.getTime()
    except Exception:
        sim_t = 0.0
    # Format simulation time as base date 2025-11-01 plus elapsed seconds -> 'YYYY-MM-DD HH:MM:SS'
    base_dt = datetime.datetime(2025, 11, 1, 0, 0, 0)
    ts_dt = base_dt + datetime.timedelta(seconds=float(sim_t))
    sim_time_str = ts_dt.strftime('%Y-%m-%d %H:%M:%S')
    total_dist_km = float(dist_tracker.total_m(truck_id) / 1000.0)
    return {
        'sim_time': sim_time_str,
        'rul': float(getattr(truck, 'rul', 0.0)),
        'driving_distance_km': total_dist_km,
        'state': str(getattr(truck, 'state', 'unknown')),
        'destination': str(getattr(truck, 'destination', '-')),
        'loaded_goods': str(getattr(truck, 'product', '')),
        'weight': float(getattr(truck, 'weight', 0.0)),
        'total_transported': float(getattr(truck, 'total_transported', 0.0)),
    }


class RemotePolicy:
    def __init__(self, bridge: MqttBridge, action_dim: int, device_map: Optional[Dict[str, str]] = None):
        self.bridge = bridge
        self.action_dim = action_dim
        self._step_id = 0
        self.device_map = device_map or {}
        self._last_action: Dict[str, int] = {}
        self._last_rul: Dict[str, float] = {}

    def next_step(self):
        self._step_id += 1

    def act(self, agent_label: str, obs_vec: np.ndarray, sensor_features) -> int:
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
        # timeout fallback
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
        candidates = candidates if candidates is not None else list(range(action_dim))
        try:
            if agent_label in self.device_map:
                _rul, action, ok, _dev = self.bridge.request_to(device_id=self.device_map[agent_label], step_id=self._step_id,
                                                                agent_id=agent_label, env_obs=[], sensor=[], action_dim=action_dim,
                                                                decision_type='pickup', pickup_candidates=candidates)
            else:
                _rul, action, ok, _dev = self.bridge.request(step_id=self._step_id, agent_id=agent_label, env_obs=[], sensor=[],
                                                             action_dim=action_dim, decision_type='pickup', pickup_candidates=candidates)
        except Exception as e:
            print(f"[SERVER][ERROR] pickup request failed step={self._step_id} agent={agent_label}: {e}")
            ok = False
            action = None
            _rul = None
        if ok and isinstance(action, int):
            act = int(action) % action_dim
            self._last_action[agent_label] = act
            try:
                self._last_rul[agent_label] = float(_rul)
            except Exception:
                pass
            print(f"[SERVER] remote pickup step={self._step_id} agent={agent_label} factory={act}")
            return act
        act = random.randrange(action_dim)
        self._last_action[agent_label] = act
        print(f"[SERVER][TIMEOUT] pickup step={self._step_id} agent={agent_label}; random_fallback={act}")
        return act


def run_demo_debug(env_args: EnvArgs, max_steps: int, bridge_cfg: BridgeConfig, mysql_cfg: Dict[str,str]):
    env_args.use_gui = False
    env = async_scheduling(env_args)
    obs = env.reset()
    action_dim = env.factory_num if env.use_rul_agent else env.factory_num + 1
    bridge = MqttBridge(bridge_cfg)
    # Fixed device mapping: truck_i -> devices[i]
    try:
        device_ids = list(bridge_cfg.devices)
    except Exception:
        device_ids = []
    fixed_map = {f'truck_{i}': device_ids[i] for i in range(min(env.truck_num, len(device_ids)))}
    if fixed_map:
        print("[MQTT] Using fixed device map:", ", ".join([f"truck_{k.split('_')[-1]}->{v}" for k,v in fixed_map.items()]))
    policy = RemotePolicy(bridge, action_dim, device_map=fixed_map)
    dist_tracker = DistanceTracker()
    # DB logger
    if MySQLRunLogger is not None and mysql_cfg.get('enable'):
        try:
            logger = MySQLRunLogger(user=mysql_cfg['user'], password=mysql_cfg['password'], host=mysql_cfg['host'], port=int(mysql_cfg['port']), prefix='sumo-')
            logger.ensure_tables([f'truck_{i}' for i in range(env.truck_num)])
            print(f"[DB] logging enabled database={logger.get_database_name()}")
        except Exception as e:
            print(f"[DB] init failed (debug): {e}")
            logger = None
    else:
        logger = None
    last_periodic_log_t = 0.0
    print_dist = mysql_cfg.get('print_dist', False)
    steps = 0
    try:
        while steps < max_steps:
            actions: Dict[int, int] = {}
            policy.next_step()
            for aid, o in obs.items():
                try:
                    truck = env.truck_agents[aid]
                except Exception:
                    truck = None
                if truck is not None and getattr(truck, 'needs_pickup', False):
                    act = policy.pickup(f'truck_{aid}', candidates=list(range(45)), action_dim=45)
                else:
                    sensor_features = getattr(truck, 'eng_obs', []) if truck is not None else []
                    act = policy.act(f'truck_{aid}', np.asarray(o, dtype=np.float32), sensor_features)
                actions[int(aid)] = int(act)
                if logger is not None:
                    try:
                        row = _collect_truck_row(f'truck_{aid}', env.truck_agents[aid], decision=True, dist_tracker=dist_tracker)
                        logger.insert(f'truck_{aid}', row)
                    except Exception as e:
                        print(f"[DB] decision log error truck_{aid}: {e}")
            if actions:
                print("decisions: " + ", ".join([f"agent{aid}->Factory{act}" for aid, act in actions.items()]))
            obs, _rew, done, _info = env.step(actions)
            # optional per-step distance debug
            if print_dist:
                try:
                    labels = []
                    for tidx in range(env.truck_num):
                        vid = f'truck_{tidx}'
                        try:
                            raw = float(traci.vehicle.getDistance(vid))
                        except Exception:
                            raw = float('nan')
                        cum = dist_tracker.total_m(vid)
                        labels.append(f"{vid}: raw={raw:.1f}m cum={cum:.1f}m")
                    print("[DIST] " + " | ".join(labels))
                except Exception as e:
                    print(f"[DIST] error: {e}")
            # periodic log
            if logger is not None:
                try:
                    cur_t = traci.simulation.getTime()
                except Exception:
                    cur_t = 0.0
                if (cur_t - last_periodic_log_t) >= 200.0:
                    for tidx in range(env.truck_num):
                        try:
                            row = _collect_truck_row(f'truck_{tidx}', env.truck_agents[tidx], decision=False, dist_tracker=dist_tracker)
                            logger.insert(f'truck_{tidx}', row)
                        except Exception as e:
                            print(f"[DB] periodic log error truck_{tidx}: {e}")
                    last_periodic_log_t = cur_t
            steps += 1
            try:
                sim_t = traci.simulation.getTime()
                print(f"step={steps} | t={sim_t:.1f}s")
            except Exception:
                print(f"step={steps}")
            if bool(np.all(done)):
                print("All trucks done; exiting.")
                break
    finally:
        try:
            traci.close()
        except Exception:
            pass
        if logger is not None:
            try:
                logger.close()
            except Exception:
                pass


def run_demo_gui(env_args: EnvArgs, max_steps: int, bridge_cfg: BridgeConfig, mysql_cfg: Dict[str,str], step_interval_ms: int = 100):
    import tkinter as tk
    from tkinter import ttk
    env_args.use_gui = True
    env = async_scheduling(env_args)
    obs = env.reset()
    action_dim = env.factory_num if env.use_rul_agent else env.factory_num + 1
    bridge = MqttBridge(bridge_cfg)
    # Fixed device mapping: truck_i -> devices[i]
    try:
        device_ids = list(bridge_cfg.devices)
    except Exception:
        device_ids = []
    fixed_map = {f'truck_{i}': device_ids[i] for i in range(min(env.truck_num, len(device_ids)))}
    if fixed_map:
        print("[MQTT] Using fixed device map:", ", ".join([f"truck_{k.split('_')[-1]}->{v}" for k,v in fixed_map.items()]))
    policy = RemotePolicy(bridge, action_dim, device_map=fixed_map)
    dist_tracker = DistanceTracker()
    if MySQLRunLogger is not None and mysql_cfg.get('enable'):
        try:
            logger = MySQLRunLogger(user=mysql_cfg['user'], password=mysql_cfg['password'], host=mysql_cfg['host'], port=int(mysql_cfg['port']), prefix='sumo-')
            logger.ensure_tables([f'truck_{i}' for i in range(env.truck_num)])
            print(f"[DB] logging enabled database={logger.get_database_name()}")
        except Exception as e:
            print(f"[DB] init failed (gui): {e}")
            logger = None
    else:
        logger = None
    last_periodic_log_t = 0.0
    print_dist = mysql_cfg.get('print_dist', False)
    root = tk.Tk()
    root.title("SUMO RL Scheduling MQTT Demo")
    tk.Label(root, text="Track truck:").grid(row=0, column=0, sticky='w')
    sel_var = tk.StringVar(value='truck_0')
    truck_ids = [f'truck_{i}' for i in range(env.truck_num)]
    sel_box = ttk.Combobox(root, textvariable=sel_var, values=truck_ids, state='readonly')
    sel_box.grid(row=0, column=1, sticky='ew')
    cam_follow_var = tk.BooleanVar(value=True)
    tk.Checkbutton(root, text="Camera follow", variable=cam_follow_var).grid(row=0, column=2, padx=8, sticky='w')
    status_var = tk.StringVar(value='-')
    dist_var = tk.StringVar(value='0.0 m')
    dest_var = tk.StringVar(value='-')
    road_var = tk.StringVar(value='-')
    rul_var = tk.StringVar(value='-')
    cargo_var = tk.StringVar(value='-')
    def make_row(r,label,var):
        tk.Label(root,text=label).grid(row=r,column=0,sticky='w')
        tk.Label(root,textvariable=var).grid(row=r,column=1,sticky='w')
    make_row(1,'Status:',status_var)
    make_row(2,'Total distance:',dist_var)
    make_row(3,'Destination:',dest_var)
    make_row(4,'Road:',road_var)
    make_row(5,'RUL:',rul_var)
    make_row(6,'Cargo (prod, weight):',cargo_var)
    summary = tk.Toplevel(root)
    summary.title('Summary Information')
    cols = ('Truck','Status','Distance(m)','Destination','Road')
    tree = ttk.Treeview(summary, columns=cols, show='headings', height=min(20, env.truck_num))
    for c in cols:
        tree.heading(c, text=c)
        tree.column(c, width=140, anchor='center')
    tree.pack(fill='both', expand=True)
    for i in range(env.truck_num):
        tree.insert('', 'end', iid=f'truck_{i}', values=(f'truck_{i}','-','0.0','-','-'))
    def update_camera(_=None):
        if env.use_gui and cam_follow_var.get():
            try:
                views = list(traci.gui.getIDList())
                if views:
                    traci.gui.trackVehicle(views[0], sel_var.get())
            except Exception:
                pass
    sel_box.bind('<<ComboboxSelected>>', update_camera)
    state = {'waiting': False}
    last_gui_update = {'t': 0.0}
    def refresh_panels(force=False):
        now = time.time()
        if not force and (now - last_gui_update['t']) < 0.5:
            return
        last_gui_update['t'] = now
        try:
            tid = sel_var.get()
            idx = int(tid.split('_')[-1])
            t = env.truck_agents[idx]
            status_var.set(t.state)
            # Total distance from SUMO via tracker (meters)
            try:
                dist_m = dist_tracker.total_m(tid)
            except Exception:
                dist_m = getattr(t, 'total_distance', 0.0)
            dist_var.set(f"{dist_m:.1f} m")
            dest_var.set(t.destination)
            rul_var.set(f"{getattr(t,'rul',0.0):.1f}")
            cargo_var.set(f"{t.product}, {t.weight:.1f}")
            try:
                road_id = traci.vehicle.getRoadID(tid)
            except Exception:
                road_id='-'
            road_var.set(road_id)
            for i in range(env.truck_num):
                vid=f'truck_{i}'
                ti=env.truck_agents[i]
                try:
                    rid = traci.vehicle.getRoadID(vid)
                except Exception:
                    rid='-'
                try:
                    vdist_m = dist_tracker.total_m(vid)
                except Exception:
                    vdist_m = getattr(ti, 'total_distance', 0.0)
                tree.item(vid, values=(vid, ti.state, f"{vdist_m:.1f}", ti.destination, rid))
        except Exception:
            pass
    def step_loop():
        nonlocal obs, last_periodic_log_t
        if state['waiting'] is False:
            if obs:
                actions: Dict[int,int] = {}
                policy.next_step()
                for aid,o in obs.items():
                    truck = env.truck_agents[aid]
                    if getattr(truck,'needs_pickup',False):
                        act = policy.pickup(f'truck_{aid}', candidates=list(range(45)), action_dim=45)
                    else:
                        sensor_features = getattr(truck,'eng_obs',[])
                        act = policy.act(f'truck_{aid}', np.asarray(o,dtype=np.float32), sensor_features)
                    actions[int(aid)] = int(act)
                    if logger is not None:
                        try:
                            row = _collect_truck_row(f'truck_{aid}', env.truck_agents[aid], decision=True, dist_tracker=dist_tracker)
                            logger.insert(f'truck_{aid}', row)
                        except Exception as e:
                            print(f"[DB] decision log error truck_{aid}: {e}")
                if actions:
                    print("decisions: "+", ".join([f"agent{aid}->Factory{act}" for aid,act in actions.items()]))
                # write back RUL for GUI display
                for aid in actions.keys():
                    rv = policy.get_last_rul(f'truck_{aid}')
                    if rv is not None:
                        env.truck_agents[aid].rul = float(rv)
                try:
                    env.gui_prepare_actions(actions)
                except Exception as e:
                    print(f"Env prepare error: {e}")
                    finish("Env prepare error")
                    return
                state['waiting'] = True
            else:
                try:
                    env.producer.produce_step(); traci.simulationStep()
                except Exception:
                    pass
                refresh_panels()
        else:
            try:
                ready = env.gui_tick_until_operable(max_sim_seconds=30.0)
            except Exception as e:
                print(f"Env tick error: {e}"); finish("Env tick error"); return
            if ready:
                try:
                    new_obs, _rew, done = env.gui_finalize_step()
                except Exception as e:
                    print(f"Env finalize error: {e}"); finish("Env finalize error"); return
                obs = new_obs
                state['waiting'] = False
                refresh_panels(force=True)
                try:
                    sim_t = traci.simulation.getTime(); print(f"t={sim_t:.1f}s")
                except Exception:
                    pass
                if print_dist:
                    try:
                        labels = []
                        for tidx in range(env.truck_num):
                            vid = f'truck_{tidx}'
                            try:
                                raw = float(traci.vehicle.getDistance(vid))
                            except Exception:
                                raw = float('nan')
                            cum = dist_tracker.total_m(vid)
                            labels.append(f"{vid}: raw={raw:.1f}m cum={cum:.1f}m")
                        print("[DIST] " + " | ".join(labels))
                    except Exception as e:
                        print(f"[DIST] error: {e}")
                if logger is not None:
                    try:
                        cur_t = traci.simulation.getTime()
                    except Exception:
                        cur_t = 0.0
                    if (cur_t - last_periodic_log_t) >= 200.0:
                        for tidx in range(env.truck_num):
                            try:
                                row = _collect_truck_row(f'truck_{tidx}', env.truck_agents[tidx], decision=False, dist_tracker=dist_tracker)
                                logger.insert(f'truck_{tidx}', row)
                            except Exception as e:
                                print(f"[DB] periodic log error truck_{tidx}: {e}")
                        last_periodic_log_t = cur_t
                if bool(np.all(done)):
                    finish("All trucks done"); return
        refresh_panels()
        root.after(step_interval_ms, step_loop)
    def finish(msg: str):
        print(msg+"; closing.")
        try: traci.close()
        except Exception: pass
        try: root.destroy()
        except Exception: pass
        if logger is not None:
            try: logger.close()
            except Exception: pass
    update_camera(); refresh_panels(force=True); root.after(step_interval_ms, step_loop)
    try:
        root.mainloop()
    finally:
        try: traci.close()
        except Exception: pass
        if logger is not None:
            try: logger.close()
            except Exception: pass


def main():
    p = argparse.ArgumentParser(description='Run SUMO RL Scheduling MQTT Demo (async_scheduling)')
    p.add_argument('--mode', choices=['debug','gui'], default='debug')
    p.add_argument('--num-agents', type=int, default=4)
    p.add_argument('--use-rul-agent', action='store_true', default=True)
    p.add_argument('--rul-threshold', type=float, default=7.0)
    p.add_argument('--rul-state', action='store_true', default=False)
    p.add_argument('--max-steps', type=int, default=1000)
    p.add_argument('--mqtt-host', default='127.0.0.1')
    p.add_argument('--mqtt-port', type=int, default=1883)
    p.add_argument('--mqtt-devices', default='edge-00,edge-01,edge-02,edge-03')
    p.add_argument('--mqtt-timeout-ms', type=int, default=1000)
    # MySQL logging options
    p.add_argument('--mysql-enable', action='store_true', help='Enable MySQL logging')
    p.add_argument('--mysql-host', default='127.0.0.1')
    p.add_argument('--mysql-port', default='3306')
    p.add_argument('--mysql-user', default='lwh')
    p.add_argument('--mysql-password', default='666888')
    # Debug print options
    p.add_argument('--print-distance-debug', action='store_true', help='Print raw and cumulative SUMO distance each step')
    args = p.parse_args()
    env_args = EnvArgs(num_agents=args.num_agents, use_rul_agent=args.use_rul_agent,
                       rul_threshold=args.rul_threshold, rul_state=args.rul_state,
                       use_gui=(args.mode=='gui'))
    devices = tuple([d.strip() for d in args.mqtt_devices.split(',') if d.strip()])
    bridge_cfg = BridgeConfig(host=args.mqtt_host, port=args.mqtt_port, timeout_ms=args.mqtt_timeout_ms,
                              devices=devices, mode='mqtt')
    if args.mysql_enable and MySQLRunLogger is None:
        print(f"[DB] mysql logging requested but import failed: {_MYSQL_IMPORT_ERR}")
    mysql_cfg = {
        'enable': bool(args.mysql_enable),
        'host': args.mysql_host,
        'port': args.mysql_port,
        'user': args.mysql_user,
        'password': args.mysql_password,
        'print_dist': bool(args.print_distance_debug),
    }
    if args.mode == 'gui':
        run_demo_gui(env_args, args.max_steps, bridge_cfg, mysql_cfg)
    else:
        run_demo_debug(env_args, args.max_steps, bridge_cfg, mysql_cfg)

if __name__ == '__main__':
    main()
