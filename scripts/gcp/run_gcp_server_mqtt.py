#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    import torch
except Exception as e:  # pragma: no cover
    raise RuntimeError("This script requires torch") from e

# Make sure repo root import works no matter where invoked
import sys
REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from onpolicy.config import get_config as get_cfg
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Critic
from onpolicy.iot.mqtt_bridge import BridgeConfig, MqttBridge


@dataclass
class ServerArgs:
    host: str = "0.0.0.0"
    port: int = 1883
    qos: int = 1
    timeout_ms: int = 3000
    topic_prefix: str = "gcp"  # isolates topics from demo/*

    num_agents: int = 4
    max_steps: int = 200
    episode_length: int = 800 * 4

    use_rul_agent: bool = True
    rul_threshold: float = 7.0
    exp_type: str = "gcp_mqtt"

    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95


class NamespacedBridge(MqttBridge):
    """Reuse MqttBridge but with configurable topic prefix."""

    def __init__(self, cfg: BridgeConfig, topic_prefix: str):
        self._topic_prefix = topic_prefix.strip("/")
        super().__init__(cfg)

    def _topic_obs(self, device_id: str) -> str:
        return f"{self._topic_prefix}/{device_id}/obs"

    def _topic_act(self, device_id: str) -> str:
        return f"{self._topic_prefix}/{device_id}/act"

    def _topic_train(self, device_id: str) -> str:
        return f"{self._topic_prefix}/{device_id}/train"

    # Override callbacks and publish functions to use namespaced topics
    def _on_connect(self, client, userdata, flags, rc):
        for dev in self.cfg.devices:
            client.subscribe(self._topic_act(dev), qos=self.cfg.qos)

    def request_to(self, *, device_id: str, step_id: int, agent_id: str, env_obs, sensor, action_dim: Optional[int] = None,
                   decision_type: str = "rl", pickup_candidates: Optional[list] = None):
        # Copy original but route publish/subscribe via namespaced topics.
        # We call parent implementation but temporarily patch topic names by shadowing publish.
        # Easiest: inline a minimal fork of parent's request_to.
        import json
        import random
        import threading
        import time as _time

        seq = self.new_seq()
        key = (step_id, agent_id, seq)

        if self.cfg.mode == "mock":
            if decision_type == "pickup":
                cands = pickup_candidates if pickup_candidates else list(range(int(action_dim or 45)))
                action = int(random.choice(cands))
                rul = self.default_rul()
                log_prob = None
            else:
                rul = float(self._mock_rul(sensor))
                action = self._mock_action(env_obs, action_dim)
                log_prob = None
            return (rul, action, log_prob, True, device_id)

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

        # Optional debug: surface edge-side predictor status when diagnosing stuck RUL.
        try:
            if getattr(self, "_debug_rul", False) and decision_type == "rul":
                dbg = res.get("rul_debug", None)
                print(
                    f"[SERVER][RUL_DEBUG] step={step_id} agent={agent_id} seq={seq} from={dev_id} rul={rul} dbg={dbg}",
                    flush=True,
                )
        except Exception:
            pass

        return (rul, action, log_prob, True, dev_id)

    def publish_train(self, device_id: str, payload: dict):
        import json
        try:
            if self._client is None:
                return False
            self._client.publish(self._topic_train(device_id), json.dumps(payload), qos=self.cfg.qos, retain=False)
            return True
        except Exception:
            return False


def build_critic(obs_dim: int, episode_length: int, lr: float):
    parser = get_cfg()
    args_tmp = parser.parse_args([])
    args_tmp.use_recurrent_policy = True
    args_tmp.use_naive_recurrent_policy = False
    args_tmp.recurrent_N = 6
    args_tmp.grid_goal = False
    args_tmp.episode_length = int(episode_length)
    args_tmp.n_rollout_threads = 1
    args_tmp.gamma = 0.99
    args_tmp.gae_lambda = 0.95
    args_tmp.use_gae = True
    args_tmp.use_popart = False
    args_tmp.use_valuenorm = True
    args_tmp.use_proper_time_limits = False
    args_tmp.asynch = False

    try:
        from gymnasium.spaces import Box, Dict
    except Exception:
        from gym.spaces import Box, Dict  # type: ignore

    share_obs_space = Dict({"global_obs": Box(low=-1, high=30000, shape=(obs_dim,))})
    critic = R_Critic(args_tmp, share_obs_space, device=torch.device("cpu"))
    opt = torch.optim.Adam(critic.parameters(), lr=float(lr))
    return critic, opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--topic-prefix", default="gcp")
    ap.add_argument("--timeout-ms", type=int, default=3000)
    ap.add_argument("--mqtt-auth", action="store_true", default=False, help="Enable MQTT username/password auth")
    ap.add_argument("--mqtt-username", default="admin")
    ap.add_argument("--mqtt-password", default="mailstrup123456")
    ap.add_argument("--num-agents", type=int, default=4)
    ap.add_argument("--episode-length", type=int, default=10000)
    ap.add_argument("--num-episodes", type=int, default=5000, help="Train for this many episodes (default: 5000)")
    ap.add_argument("--max-steps", type=int, default=None, help="Optional hard cap on total env steps (overrides num-episodes * episode-length)")
    ap.add_argument("--use-rul-agent", action="store_true", default=True)
    ap.add_argument("--rul-threshold", type=float, default=7.0)
    ap.add_argument("--exp-type", default="gcp_mqtt")
    ap.add_argument("--devices", default="edge-00,edge-01,edge-02,edge-03")
    ap.add_argument("--debug-rul", action="store_true", default=False, help="Print edge-provided RUL debug metadata")
    args = ap.parse_args()

    # Server env: same logic as train_schedule.py but using gcp env (no local predictor)
    from onpolicy.envs.gcp.schedule import async_scheduling, GcpScheduleArgs

    env_args = GcpScheduleArgs(
        num_agents=int(args.num_agents),
        use_rul_agent=bool(args.use_rul_agent),
        rul_threshold=float(args.rul_threshold),
        rul_state=False,
        exp_type=str(args.exp_type),
    )

    devices = tuple([d.strip() for d in str(args.devices).split(",") if d.strip()])
    bridge_cfg = BridgeConfig(
        host=args.host,
        port=int(args.port),
        timeout_ms=int(args.timeout_ms),
        devices=devices,
        enable_auth=bool(args.mqtt_auth),
        username=str(args.mqtt_username),
        password=str(args.mqtt_password),
    )
    bridge = NamespacedBridge(bridge_cfg, topic_prefix=str(args.topic_prefix))
    # Enable extra debug prints inside the bridge if requested.
    try:
        bridge._debug_rul = bool(args.debug_rul)  # type: ignore[attr-defined]
    except Exception:
        pass

    last_rul: Dict[int, float] = {}
    rul_req_counter = [0]
    def rul_provider(agent_id: int, eng_obs) -> float:
        # Ask the same device that serves this agent (fixed mapping by index)
        device_id = devices[agent_id % len(devices)]
        # Use a monotonic counter for (step_id, agent_id, seq) correlation.
        rul_req_counter[0] += 1
        # Edge computes RUL regardless; action_dim=0 means action is irrelevant.
        rul, _act, _lp, ok, _dev = bridge.request_to(
            device_id=device_id,
            step_id=int(rul_req_counter[0]),
            agent_id=f"truck_{agent_id}",
            env_obs=[],
            sensor=eng_obs,
            action_dim=0,
            decision_type="rul",
        )
        if ok:
            last_rul[agent_id] = float(rul)
        return float(last_rul.get(agent_id, 125.0))

    env = async_scheduling(env_args, rul_provider=rul_provider)

    obs = env.reset()
    sample_obs = next(iter(obs.values()))
    env_obs_dim = len(sample_obs)
    obs_dim = env_obs_dim + 1
    action_dim = env.factory_num if env.use_rul_agent else env.factory_num + 1

    # Case-1 distributed training:
    # - Edge does actor inference + actor training.
    # - Server trains critic and computes advantages (GAE).
    critic, critic_opt = build_critic(obs_dim=obs_dim, episode_length=int(args.episode_length), lr=1e-3)

    # Per-agent episode trajectories (only for decision points / operable trucks).
    traj: Dict[int, dict] = {
        int(a): {"obs": [], "actions": [], "log_probs": [], "rewards": [], "dones": [], "values": [], "next_values": []}
        for a in range(int(args.num_agents))
    }

    def critic_value(obs_vec: np.ndarray) -> float:
        x = torch.as_tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
        rs = torch.zeros((1, critic._recurrent_N, critic.hidden_size), dtype=torch.float32)
        masks = torch.ones((1, 1), dtype=torch.float32)
        with torch.no_grad():
            v, _rs2 = critic(x, rs, masks)
        return float(v.view(-1)[0].item())

    def compute_gae(rews: list, dones_list: list, vals: list, next_vals: list, gamma: float, lam: float):
        T = len(rews)
        adv = [0.0] * T
        gae = 0.0
        for t in reversed(range(T)):
            done = bool(dones_list[t])
            delta = float(rews[t]) + gamma * float(next_vals[t]) * (0.0 if done else 1.0) - float(vals[t])
            gae = delta + gamma * lam * (0.0 if done else 1.0) * gae
            adv[t] = gae
        returns = [float(adv[i]) + float(vals[i]) for i in range(T)]
        return adv, returns

    total_steps_cap = int(args.max_steps) if args.max_steps is not None else int(args.num_episodes) * int(args.episode_length)
    print(
        f"[SERVER] started host={args.host}:{args.port} prefix={args.topic_prefix} agents={env.truck_num} action_dim={action_dim} "
        f"episode_length={int(args.episode_length)} num_episodes={int(args.num_episodes)} total_steps_cap={int(total_steps_cap)}"
    )

    global_step = 0
    episode_step = 0
    episode_idx = 0

    while episode_idx < int(args.num_episodes) and global_step < int(total_steps_cap):
        step_id = int(global_step)

        actions: Dict[int, int] = {}
        pending: Dict[int, dict] = {}

        # Request decisions for operable trucks only.
        for aid, o in obs.items():
            aid = int(aid)
            device_id = devices[aid % len(devices)]

            # Ensure env has a recent RUL cached for this truck.
            rul_val = float(last_rul.get(aid, 125.0))
            if aid not in last_rul:
                try:
                    rul_val = float(rul_provider(aid, env.truck_agents[aid].eng_obs))
                except Exception:
                    rul_val = 125.0
            combined = np.asarray([rul_val] + list(np.asarray(o, dtype=np.float32)), dtype=np.float32)

            v = critic_value(combined)
            rul, act, logp, ok, _dev = bridge.request_to(
                device_id=device_id,
                step_id=step_id,
                agent_id=f"truck_{aid}",
                env_obs=np.asarray(o, dtype=np.float32),
                sensor=env.truck_agents[aid].eng_obs,
                action_dim=int(action_dim),
            )
            if act is None:
                act = int(np.random.randint(0, action_dim))
            actions[aid] = int(act)

            pending[aid] = {
                "device_id": device_id,
                "obs": combined,
                "action": int(act),
                "log_prob": float(logp) if logp is not None else 0.0,
                "value": float(v),
            }

        obs_next, rewards, dones, _info = env.step(actions)

        # Record transitions for those agents that acted.
        for aid, data in pending.items():
            r = float(rewards[aid][0]) if not isinstance(rewards, dict) else float(rewards.get(aid, 0.0))
            done = bool(dones[aid]) if not isinstance(dones, bool) else bool(dones)

            nxt_env = obs_next.get(aid)
            if nxt_env is None:
                nv = 0.0
            else:
                nxt_rul = float(last_rul.get(aid, 125.0))
                nxt_combined = np.asarray([nxt_rul] + list(np.asarray(nxt_env, dtype=np.float32)), dtype=np.float32)
                nv = critic_value(nxt_combined)

            t = traj[aid]
            t["obs"].append(data["obs"].tolist())
            t["actions"].append(int(data["action"]))
            t["log_probs"].append(float(data["log_prob"]))
            t["rewards"].append(float(r))
            t["dones"].append(bool(done))
            t["values"].append(float(data["value"]))
            t["next_values"].append(float(nv))

        obs = obs_next
        global_step += 1
        episode_step += 1

        if step_id % 10 == 0:
            print(f"[SERVER] step={step_id} active={len(pending)} ep_step={episode_step}")

        done_flag = bool(np.all(dones))
        if done_flag or episode_step >= int(args.episode_length):
            episode_idx += 1
            print(f"[SERVER] episode_end idx={episode_idx} steps={episode_step} (done={done_flag}) -> train critic + send advantages")

            # Compute per-agent advantages/returns and train critic on all gathered transitions.
            all_obs = []
            all_returns = []
            per_agent_adv: Dict[int, list] = {}

            for aid, t in traj.items():
                if len(t["rewards"]) == 0:
                    continue
                adv, rets = compute_gae(
                    t["rewards"],
                    t["dones"],
                    t["values"],
                    t["next_values"],
                    gamma=0.99,
                    lam=0.95,
                )
                per_agent_adv[aid] = adv
                all_obs.extend(t["obs"])
                all_returns.extend(rets)

            # Critic update (single epoch on full batch).
            if len(all_obs) > 0:
                obs_batch = torch.as_tensor(np.asarray(all_obs, dtype=np.float32), dtype=torch.float32)
                ret_batch = torch.as_tensor(np.asarray(all_returns, dtype=np.float32), dtype=torch.float32).unsqueeze(-1)
                rs = torch.zeros((obs_batch.shape[0], critic._recurrent_N, critic.hidden_size), dtype=torch.float32)
                masks = torch.ones((obs_batch.shape[0], 1), dtype=torch.float32)
                v_pred, _rs2 = critic(obs_batch, rs, masks)
                v_loss = torch.nn.functional.mse_loss(v_pred, ret_batch)
                critic_opt.zero_grad()
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
                critic_opt.step()
                print(f"[SERVER][CRITIC] batch={len(all_obs)} v_loss={float(v_loss.item()):.6f}")

            # Send per-agent train batches to their mapped edge device.
            for aid, t in traj.items():
                if len(t["rewards"]) == 0:
                    continue
                device_id = devices[int(aid) % len(devices)]
                payload = {
                    "type": "train_batch",
                    "episode": int(episode_idx),
                    "agent_id": f"truck_{aid}",
                    "obs": t["obs"],
                    "actions": t["actions"],
                    "old_log_probs": t["log_probs"],
                    "advantages": per_agent_adv.get(aid, []),
                    "action_dim": int(action_dim),
                }
                bridge.publish_train(device_id, payload)

            # Reset episode
            for aid in traj.keys():
                traj[aid] = {"obs": [], "actions": [], "log_probs": [], "rewards": [], "dones": [], "values": [], "next_values": []}
            obs = env.reset()
            episode_step = 0

            if episode_idx >= int(args.num_episodes):
                break

        time.sleep(0.005)


if __name__ == "__main__":
    main()
