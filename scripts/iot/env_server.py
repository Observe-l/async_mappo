#!/usr/bin/env python3
"""
Environment server for local MQTT distributed inference demo.

- Starts 4-agent scheduling environment (async_scheduling) in headless mode.
- Dispatches each operable agent observation to a fixed device (edge-01..edge-04) round-robin.
- Waits (timeout) for response with RUL + action; applies action.
- Advances simulation similar to debug mode (non-GUI).

Note: Relies on existing onpolicy.envs.rul_schedule.demo_schedule.async_scheduling.
"""
from __future__ import annotations
import time
import json
import argparse
from pathlib import Path
import numpy as np
import sys
from typing import Dict

# Ensure root path
REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from onpolicy.envs.rul_schedule.demo_schedule import async_scheduling
from onpolicy.iot.mqtt_bridge import MqttBridge, BridgeConfig

DEVICE_IDS = ("edge-01", "edge-02", "edge-03", "edge-04")


def build_env(num_agents: int = 4):
    class Args:
        pass
    a = Args()
    a.num_agents = num_agents
    a.use_rul_agent = True
    a.rul_threshold = 7.0
    a.rul_state = False  # Do not include RUL in obs; edge clients compute it
    a.use_gui = False
    a.exp_type = "mqtt_local_demo"
    return async_scheduling(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--timeout-ms', type=int, default=150)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=1883)
    ap.add_argument('--devices', default=','.join(DEVICE_IDS))
    args = ap.parse_args()

    devices = tuple([d.strip() for d in args.devices.split(',') if d.strip()])
    bridge = MqttBridge(BridgeConfig(host=args.host, port=args.port, timeout_ms=args.timeout_ms, devices=devices, mode='mqtt'))

    env = build_env(num_agents=4)
    obs = env.reset()
    action_dim = env.factory_num if env.use_rul_agent else env.factory_num + 1

    step = 0
    while step < args.steps:
        # Collect actions for current operable agents
        actions: Dict[int, int] = {}
        for aid, o in obs.items():
            vec = np.asarray(o, dtype=np.float32)
            device_id = devices[aid % len(devices)]  # fixed mapping per agent id
            rul, act, ok, dev = bridge.request_to(device_id=device_id, step_id=step+1, agent_id=f"truck_{aid}", env_obs=vec, sensor=[], action_dim=action_dim)
            if not ok:
                print(f"[WARN] timeout a{aid} device={device_id}; fallback act={act}")
            actions[int(aid)] = int(act)
        if actions:
            print("decisions: " + ", ".join([f"agent{aid}->Factory{act}" for aid, act in actions.items()]))
        # Step env
        obs, rew, done, _ = env.step(actions)
        step += 1
        # Debug summary line
        try:
            import libsumo as traci
        except Exception:
            import traci  # type: ignore
        try:
            sim_t = traci.simulation.getTime()
        except Exception:
            sim_t = -1
        print(f"step={step} t={sim_t:.1f}s operable={len(obs)} done={bool(np.all(done))}")
        if bool(np.all(done)):
            break
    print("Simulation complete.")

if __name__ == '__main__':
    main()
