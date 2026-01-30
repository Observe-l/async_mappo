#!/usr/bin/env python3
"""Train async-MAPPO with the same logic as train_schedule.py, but with MQTT-based RUL.

Goal
- Keep actor/critic training code-path identical to train_schedule.py (Runner/Policy/Trainer/Buffer).
- Preserve the async schedule execution pattern (AsyncControl) and update once per episode.
- Replace ONLY the environment's RUL source with a remote MQTT RUL provider.

Notes
- This script does NOT move gradient updates to edge devices.
  Exact parity with train_schedule.py requires running the original training code (R_MAPPO trainer)
  in one process; distributing actor/critic training across machines would require distributed
  optimizers/gradient synchronization and would not be "exactly identical" without extra systems.
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from onpolicy.config import get_config


# Ensure repo root is on sys.path so local imports work no matter where invoked.
REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _build_namespaced_bridge(topic_prefix: str, host: str, port: int, timeout_ms: int,
                             devices: Tuple[str, ...], mqtt_auth: bool,
                             mqtt_username: str, mqtt_password: str):
    """Build a topic-namespaced MQTT bridge (gcp/*) compatible with existing edge_client.py."""

    from onpolicy.iot.mqtt_bridge import BridgeConfig, MqttBridge

    class NamespacedBridge(MqttBridge):
        def __init__(self, cfg: BridgeConfig, topic_prefix: str):
            self._topic_prefix = (topic_prefix or "demo").strip("/")
            super().__init__(cfg)

        def _topic_obs(self, device_id: str) -> str:
            return f"{self._topic_prefix}/{device_id}/obs"

        def _topic_act(self, device_id: str) -> str:
            return f"{self._topic_prefix}/{device_id}/act"

        def _on_connect(self, client, userdata, flags, rc):
            for dev in self.cfg.devices:
                client.subscribe(self._topic_act(dev), qos=self.cfg.qos)

        def request_to(self, *, device_id: str, step_id: int, agent_id: str, env_obs, sensor,
                       action_dim: Optional[int] = None,
                       decision_type: str = "rl", pickup_candidates: Optional[list] = None):
            # Fork of MqttBridge.request_to with namespaced topics.
            import json
            import threading
            import time as _time

            seq = self.new_seq()
            key = (step_id, agent_id, seq)

            if self.cfg.mode == "mock":
                # Not used here; keep behavior consistent.
                rul = self.default_rul()
                action = self.default_action(env_obs, action_dim)
                return (rul, action, None, True, device_id)

            assert self._client is not None, "MQTT client not initialized"

            def to_plain(x):
                try:
                    import numpy as _np
                except Exception:
                    _np = None
                if _np is not None and isinstance(x, _np.ndarray):
                    try:
                        return x.astype(float).tolist()
                    except Exception:
                        return x.tolist()
                if isinstance(x, (list, tuple)):
                    return [to_plain(v) for v in x]
                if isinstance(x, dict):
                    return {k: to_plain(v) for k, v in x.items()}
                try:
                    import numbers as _numbers

                    if isinstance(x, _numbers.Number):
                        return float(x)
                except Exception:
                    pass
                return x

            payload = {
                "step_id": step_id,
                "agent_id": agent_id,
                "seq": seq,
                "timestamp": _time.time(),
                "env_obs": {"vector": to_plain(env_obs)},
                "sensor": {"feature_vector": to_plain(sensor)},
                "meta": {
                    "action_dim": int(action_dim) if action_dim is not None else None,
                    "decision_type": decision_type,
                    "pickup_candidates": to_plain(pickup_candidates) if pickup_candidates is not None else None,
                },
            }

            ev = threading.Event()
            with self._lock:
                self._pending[key] = (ev, {})

            self._client.publish(self._topic_obs(device_id), json.dumps(payload), qos=self.cfg.qos, retain=False)
            ok = ev.wait(self.cfg.timeout_ms / 1000.0)

            with self._lock:
                _ev2, res = self._pending.pop(key, (None, {}))

            if not ok or not res:
                return (self.default_rul(), None, None, False, device_id)

            try:
                rul = float(res.get("rul", self.default_rul()))
                action_val = res.get("action", None)
                if isinstance(action_val, list):
                    action = int(action_val[-1])
                else:
                    action = int(action_val) if action_val is not None else None
                log_prob_val = res.get("log_prob", None)
                log_prob = float(log_prob_val) if log_prob_val is not None else None
                dev_id = str(res.get("device_id", device_id))
            except Exception:
                rul, action, log_prob, dev_id = self.default_rul(), None, None, device_id

            return (rul, action, log_prob, True, dev_id)

    cfg = BridgeConfig(
        host=str(host),
        port=int(port),
        timeout_ms=int(timeout_ms),
        devices=tuple(devices),
        enable_auth=bool(mqtt_auth),
        username=str(mqtt_username),
        password=str(mqtt_password),
    )
    return NamespacedBridge(cfg, topic_prefix=str(topic_prefix))


def make_train_env_mqtt(all_args, rul_provider):
    from onpolicy.envs.env_wrappers import ScheduleEnv

    def get_env_fn(rank):
        def init_env():
            # Use distributed env variant that supports injected RUL provider.
            from onpolicy.envs.gcp.schedule import async_scheduling

            env = async_scheduling(all_args, rul_provider=rul_provider)
            return env

        return init_env

    return ScheduleEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env_mqtt(all_args, rul_provider):
    from onpolicy.envs.env_wrappers import ScheduleEnv

    def get_env_fn(rank):
        def init_env():
            from onpolicy.envs.gcp.schedule import async_scheduling

            env = async_scheduling(all_args, rul_provider=rul_provider)
            return env

        return init_env

    return ScheduleEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def main(cli_args):
    # Reuse the exact argument set from train_schedule.py, and extend with MQTT options.
    import train_schedule as ts

    parser = get_config()

    # MQTT options (extra)
    parser.add_argument("--mqtt-host", default="127.0.0.1")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--topic-prefix", default="gcp")
    parser.add_argument("--timeout-ms", type=int, default=300)
    parser.add_argument("--devices", default="edge-00,edge-01,edge-02,edge-03")
    parser.add_argument("--mqtt-auth", action="store_true", default=False)
    parser.add_argument("--mqtt-username", default="admin")
    parser.add_argument("--mqtt-password", default="mailstrup123456")

    all_args = ts.parse_args(cli_args, parser)

    # Make intent explicit in logs/dirs.
    all_args.scenario_name = "gcp_mqtt"

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir (match train_schedule.py layout)
    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # episode length matches max_steps in this codebase
    all_args.episode_length = all_args.max_steps

    devices = tuple([d.strip() for d in str(all_args.devices).split(",") if d.strip()])
    bridge = _build_namespaced_bridge(
        topic_prefix=str(all_args.topic_prefix),
        host=str(all_args.mqtt_host),
        port=int(all_args.mqtt_port),
        timeout_ms=int(all_args.timeout_ms),
        devices=devices,
        mqtt_auth=bool(all_args.mqtt_auth),
        mqtt_username=str(all_args.mqtt_username),
        mqtt_password=str(all_args.mqtt_password),
    )

    # Remote RUL provider used by the env.
    # We keep a local cache for timeouts.
    req_counter = {"i": 0}
    last_rul: Dict[int, float] = {}

    def rul_provider(agent_id: int, eng_obs) -> float:
        req_counter["i"] += 1
        step_id = int(req_counter["i"])
        device_id = devices[int(agent_id) % len(devices)]

        rul, _act, _lp, ok, _dev = bridge.request_to(
            device_id=device_id,
            step_id=step_id,
            agent_id=f"truck_{int(agent_id)}",
            env_obs=[],
            sensor=eng_obs,
            action_dim=0,
            decision_type="rl",
        )
        if ok:
            last_rul[int(agent_id)] = float(rul)
        return float(last_rul.get(int(agent_id), 125.0))

    # env init
    envs = make_train_env_mqtt(all_args, rul_provider=rul_provider)
    eval_envs = make_eval_env_mqtt(all_args, rul_provider=rul_provider) if all_args.use_eval else None

    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # runner selection matches train_schedule.py
    if all_args.share_policy:
        from onpolicy.runner.shared.schedule_runner import ScheduleRunner as Runner
        print("share policy")
    else:
        from onpolicy.runner.separated.schedule_runner import ScheduleRunner as Runner
        print("separated policy")

    runner = Runner(config)
    runner.run()

    envs.close()
    if all_args.use_eval and eval_envs is not envs and eval_envs is not None:
        eval_envs.close()

    if not all_args.use_wandb:
        runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
