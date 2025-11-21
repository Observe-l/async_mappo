#!/usr/bin/env python3
"""
Simple IoT edge client for SUMO-RL demo.

- Subscribes to demo/obs/{device_id}
- Performs dummy inference (random) or loads TorchScript models if provided
- Publishes results to demo/act/{device_id}

Usage:
  python3 scripts/iot/edge_client.py --device-id edge-01 [--host 127.0.0.1 --port 1883]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
import sys
# Support both Gymnasium and classic Gym
try:
    from gymnasium.spaces import Discrete, Box, Dict
except Exception:
    from gym.spaces import Discrete, Box, Dict
from typing import Optional
import numpy as np

# Ensure project root is on sys.path so 'onpolicy' and other local packages are importable
try:
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
except Exception:
    pass

try:
    import paho.mqtt.client as mqtt  # type: ignore
except Exception as e:
    print("paho-mqtt is required: pip install paho-mqtt")
    raise

try:
    import torch  # type: ignore
except Exception:
    torch = None

# Optional TensorFlow RUL predictor (fallback when TorchScript RUL model not provided)
try:
    from onpolicy.envs.rul_schedule.rul_gen import predictor as RulPredictor  # type: ignore
except Exception:
    RulPredictor = None

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
            return random.randrange(self._action_dim())
        try:
            obs_tensor = torch.as_tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
            rnn_states = torch.zeros((1, self.args.recurrent_N, self.args.hidden_size), dtype=torch.float32)
            masks = torch.ones((1, 1), dtype=torch.float32)
            with torch.no_grad():
                actions, log_probs, _rnn = self.actor(obs_tensor, rnn_states, masks, available_actions=None, deterministic=False)
            act_int = int(actions.view(-1)[0].item())
            self._last_log_prob = float(log_probs.view(-1)[0].item()) if log_probs is not None else None
            return act_int
        except Exception as e:
            print(f"Inference error ({e}); returning random action.")
            self._last_log_prob = None
            return random.randrange(self._action_dim())

    def _action_dim(self) -> int:
        # Infer from last linear layer out features if loaded, else attempt env action space length
        try:
            return int(self.actor.act.action_out.linear.out_features)
        except Exception:
            return 1

def load_torchscript_model(path: str):
    if not path:
        return None
    if torch is None:
        print("Torch not available; running dummy inference.")
        return None
    p = Path(path)
    if not p.exists():
        print(f"Model not found: {path}; using dummy inference")
        return None
    try:
        m = torch.jit.load(str(p))
        try:
            print(f"Loaded TorchScript model: {p}")
        except Exception:
            pass
        return m
    except Exception as e:
        print(f"Failed to load TorchScript model {path}: {e}")
        return None


class EdgeClient:
    def __init__(self, device_id: str, host: str, port: int, rul_model: str = '', actor_model: str = '', actor_dir: str = '', fresh_actor: bool = False):
        print(f"[CLIENT {device_id}] init starting host={host} port={port} fresh_actor={fresh_actor}", flush=True)
        self.device_id = device_id
        self.host = host
        self.port = port
        self.gcpatr = load_torchscript_model(rul_model)
        self.actor = load_torchscript_model(actor_model)
        self.actor_dir = actor_dir
        self._actor_built = False
        # Instantiate TF predictor if TorchScript RUL model not supplied
        self.rul_predictor = None
        if self.gcpatr is None and RulPredictor is not None:
            try:
                self.rul_predictor = RulPredictor()
                print(f"[CLIENT {self.device_id}] loaded TF RUL predictor (lookback={getattr(self.rul_predictor,'lw','?')})")
            except Exception as e:
                print(f"[CLIENT {self.device_id}] failed to init TF RUL predictor: {e}")
        self.client = mqtt.Client(client_id=f"edge-{device_id}-{random.randrange(1,1_000_000)}", clean_session=True)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        # Training state
        self.enable_training = True  # could expose flag later
        self.gamma = 0.99
        self.train_steps = 0
        self._last_obs = {}
        self._optimizer = None
        self.fresh_actor = bool(fresh_actor)
        self._horizon = 32
        self._lambda = 0.95
        self._clip = 0.2
        self._entropy_coef = 0.01
        self._batch_buffer = {  # per agent temporary storage
            'obs': {}, 'actions': {}, 'rewards': {}, 'values': {}, 'dones': {}, 'next_values': {}
        }
        self._last_action_dim = 0

        # Immediate attempt to build pretrained RL actor BEFORE connecting (ensures startup log appears first)
        if not self.fresh_actor and self.actor is None and self.actor_dir:
            try:
                inferred_obs, inferred_act = self._infer_actor_dims(self.actor_dir)
                obs_space = Dict({"global_obs": Box(low=-1, high=30000, shape=(inferred_obs,))})
                act_space = Discrete(inferred_act)
                self.actor = TrainedPolicy(actor_dir=self.actor_dir,
                                    env_obs_space=obs_space,
                                    env_share_obs_space=obs_space,
                                    env_act_space=act_space,
                                    use_recurrent_policy=True,
                                    recurrent_N=6)
                self._actor_built = True
            except Exception as e:
                print(f"[CLIENT {self.device_id}] initial actor load failed: {e}")

        # Connect to broker after local init
        try:
            self.client.connect(host, port, keepalive=30)
            print(f"[CLIENT {self.device_id}] connect initiated", flush=True)
        except Exception as e:
            print(f"[CLIENT {self.device_id}] connect failed: {e}", flush=True)

    def start(self):
        print(f"[CLIENT {self.device_id}] entering loop", flush=True)
        self.client.loop_forever()

    def on_connect(self, client, userdata, flags, rc):
        topic_obs = f"demo/obs/{self.device_id}"
        topic_train = f"demo/train/{self.device_id}"
        client.subscribe(topic_obs, qos=1)
        client.subscribe(topic_train, qos=1)
        print(f"[CLIENT {self.device_id}] connected rc={rc}, subscribed {topic_obs} & {topic_train}")

    def on_message(self, client, userdata, msg):
        is_train = msg.topic.startswith("demo/train/")
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
        except Exception:
            return
        if is_train:
            self._handle_train(payload)
            return
        step_id = payload.get('step_id')
        agent_id = payload.get('agent_id')
        seq = payload.get('seq')
        env_vec = payload.get('env_obs', {}).get('vector', [])
        sensor_vec = payload.get('sensor', {}).get('feature_vector', [])
        meta = payload.get('meta', {}) or {}
        action_dim = int(meta.get('action_dim') or 0)
        decision_type = meta.get('decision_type', 'rl') or 'rl'
        pickup_candidates = meta.get('pickup_candidates') or None
        try:
            if decision_type == 'pickup':
                print(f"[CLIENT {self.device_id}] recv PICKUP step={step_id} agent={agent_id} seq={seq} candidates={len(pickup_candidates) if pickup_candidates else 'NA'}")
            else:
                print(f"[CLIENT {self.device_id}] recv RL step={step_id} agent={agent_id} seq={seq} action_dim={action_dim}")
            if action_dim > 0:
                self._last_action_dim = action_dim
        except Exception:
            pass

        # Inference timings (simple)
        t0 = time.time()
        rul = self.infer_rul(sensor_vec)
        t1 = time.time()
        if decision_type == 'pickup':
            action = self._infer_pickup(pickup_candidates, action_dim)
            log_prob = None
        else:
            action, log_prob = self.infer_action(env_vec, rul, action_dim)
        t2 = time.time()

        reply = {
            'step_id': step_id,
            'agent_id': agent_id,
            'seq': seq,
            'rul': float(rul),
            'action': int(action),
            'log_prob': float(log_prob) if log_prob is not None else None,
            'inference_ms': float((t2 - t0) * 1000.0),
            'device_id': self.device_id,
            'decision_type': decision_type,
        }
        client.publish(f"demo/act/{self.device_id}", json.dumps(reply), qos=1, retain=False)
        try:
            if decision_type == 'pickup':
                print(f"[CLIENT {self.device_id}] send PICKUP step={step_id} agent={agent_id} seq={seq} action={int(action)} -> pick up goods at Factory{int(action)} rul={float(rul):.2f}")
            else:
                print(f"[CLIENT {self.device_id}] send RL step={step_id} agent={agent_id} seq={seq} action={int(action)} -> delivery goods to Factory{int(action)} rul={float(rul):.2f}")
        except Exception:
            pass

    def _init_optimizer_if_needed(self):
        if self._optimizer is None and self.enable_training and torch is not None and isinstance(self.actor, TrainedPolicy) and self.actor.actor is not None:
            try:
                self._optimizer = torch.optim.Adam(self.actor.actor.parameters(), lr=1e-4)
                print(f"[CLIENT {self.device_id}] optimizer initialized for actor training.")
            except Exception as e:
                print(f"[CLIENT {self.device_id}] optimizer init failed: {e}")

    def _handle_train(self, payload: dict):
        if not self.enable_training or torch is None:
            return
        # Fresh actor flag control: if payload indicates fresh training start
        msg_type = payload.get('type','train')
        if msg_type == 'config':
            self.fresh_actor = bool(payload.get('fresh_actor', False))
            return
        if msg_type == 'train':
            # Step transition accumulation
            agent_id = payload.get('agent_id')
            if agent_id is None:
                return
            try:
                obs = payload.get('obs', [])
                action = int(payload.get('action'))
                reward = float(payload.get('reward'))
                done = bool(payload.get('done'))
                value = float(payload.get('value', 0.0))
                next_value = float(payload.get('next_value', 0.0))
            except Exception:
                return
            b = self._batch_buffer
            for key,data in [('obs',obs),('actions',action),('rewards',reward),('values',value),('dones',done),('next_values',next_value)]:
                arr = b[key].setdefault(agent_id, [])
                arr.append(data)
            # Build actor lazily if fresh and not built yet
            if self.actor is None or (isinstance(self.actor, TrainedPolicy) and self.actor.actor is None):
                inferred_action_dim = int(payload.get('action_dim') or self._last_action_dim or 0)
                self._lazy_build_actor(len(obs), inferred_action_dim)
            horizon_reached = len(b['obs'][agent_id]) >= self._horizon or done
            if not horizon_reached:
                return
            self._init_optimizer_if_needed()
            self._update_actor_batch(agent_id)
            # reset buffers for agent
            for key in b.keys():
                b[key][agent_id] = []
            return
        if msg_type == 'train_batch':
            # Future extension for server-sent batch updates
            return

    def _lazy_build_actor(self, obs_len: int, action_dim: int):
        if torch is None or obs_len <= 0 or action_dim <= 0:
            return
        try:
            from onpolicy.config import get_config
            from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
            parser = get_config()
            args_local = parser.parse_args([])
            args_local.algorithm_name = 'mappo'
            args_local.use_recurrent_policy = True
            args_local.use_naive_recurrent_policy = False
            args_local.recurrent_N = 6
            args_local.grid_goal = False
            from gymnasium.spaces import Discrete, Box, Dict
            obs_space = Dict({"global_obs": Box(low=-1, high=30000, shape=(obs_len,))})
            act_space = Discrete(action_dim)
            self.actor = TrainedPolicy(actor_dir=None, env_obs_space=obs_space, env_share_obs_space=obs_space, env_act_space=act_space, use_recurrent_policy=True, recurrent_N=6)
            print(f"[CLIENT {self.device_id}] fresh actor initialized obs_dim={obs_len} action_dim={action_dim}")
        except Exception as e:
            print(f"[CLIENT {self.device_id}] fresh actor init failed: {e}")

    def _update_actor_batch(self, agent_id: str):
        if torch is None or not isinstance(self.actor, TrainedPolicy) or self.actor.actor is None or self._optimizer is None:
            return
        b = self._batch_buffer
        obs_seq = b['obs'][agent_id]
        act_seq = b['actions'][agent_id]
        rew_seq = b['rewards'][agent_id]
        val_seq = b['values'][agent_id]
        next_val_seq = b['next_values'][agent_id]
        done_seq = b['dones'][agent_id]
        T = len(obs_seq)
        if T == 0:
            return
        # Compute GAE
        advantages = [0.0]*T
        gae = 0.0
        for t in reversed(range(T)):
            next_value = next_val_seq[t]
            delta = rew_seq[t] + self.gamma * next_value * (0.0 if done_seq[t] else 1.0) - val_seq[t]
            gae = delta + self.gamma * self._lambda * (0.0 if done_seq[t] else 1.0) * gae
            advantages[t] = gae
        returns = [advantages[i] + val_seq[i] for i in range(T)]
        # Prepare tensors
        obs_tensor = torch.as_tensor(obs_seq, dtype=torch.float32)
        actions_tensor = torch.as_tensor(act_seq, dtype=torch.int64).unsqueeze(-1)
        adv_tensor = torch.as_tensor(advantages, dtype=torch.float32)
        # Normalize advantages
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
        rnn_states = torch.zeros((T, self.actor.args.recurrent_N, self.actor.args.hidden_size), dtype=torch.float32)
        masks = torch.ones((T,1), dtype=torch.float32)
        # Evaluate actions
        log_probs_list = []
        entropy_list = []
        # Process each step individually due to recurrent state
        rs = rnn_states[0:1]
        for t in range(T):
            obs_step = obs_tensor[t:t+1]
            act_step = actions_tensor[t:t+1]
            lp, ent, _vals = self.actor.actor.evaluate_actions(obs_step, rs, act_step, masks[t:t+1])
            log_probs_list.append(lp.view(-1)[0])
            entropy_list.append(ent.view(-1)[0])
        log_probs = torch.stack(log_probs_list)
        entropy = torch.stack(entropy_list).mean()
        # PPO surrogate (no old log probs stored; treat current as baseline -> ratio=1, so simple policy gradient)
        loss_pg = -(log_probs * adv_tensor).mean()
        loss_entropy = -self._entropy_coef * entropy
        loss_total = loss_pg + loss_entropy
        self._optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.actor.parameters(), 5.0)
        self._optimizer.step()
        self.train_steps += 1
        print(f"[CLIENT {self.device_id}] actor batch update #{self.train_steps} steps={T} loss={loss_total.item():.4f} pg={loss_pg.item():.4f} ent={entropy.item():.4f}")

    def infer_rul(self, sensor_vec):
        # TorchScript model path provided
        if self.gcpatr is not None:
            try:
                import torch
                x = torch.tensor(sensor_vec, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    y = self.gcpatr(x)
                return float(y.view(-1)[0].item())
            except Exception:
                pass
        # Fallback to TF predictor if available and sensor_vec is a sequence of engine observations
        if self.rul_predictor is not None:
            try:
                return float(self.rul_predictor.predict(sensor_vec))
            except Exception:
                pass
        # Final random fallback
        return 500.0 + random.random() * 100.0

    def infer_action(self, env_vec, rul, action_dim: int = 0):
        if self.actor is not None:
            try:
                act = int(self.actor.act([float(rul)] + list(env_vec)))
                lp = getattr(self.actor, '_last_log_prob', None)
                return act, lp
            except Exception as e:
                print(f"[CLIENT {self.device_id}] actor inference error: {e}")
        if action_dim and action_dim > 0:
            return random.randrange(0, int(action_dim)), None
        return random.randrange(0, 5), None

    def _infer_pickup(self, pickup_candidates, action_dim: int):
        """Randomly select a pickup factory among candidates (0-44). RL actor not used here."""
        if pickup_candidates and isinstance(pickup_candidates, list) and len(pickup_candidates) > 0:
            try:
                return int(random.choice(pickup_candidates))
            except Exception:
                pass
        # fallback to uniform range
        upper = action_dim if action_dim and action_dim > 0 else 45
        try:
            return int(random.randrange(0, int(upper)))
        except Exception:
            return 0

    def _infer_actor_dims(self, actor_dir: str):
        """Infer observation and action dimensions from actor.pt state dict without needing env spaces.
        Returns (obs_dim, action_dim). Falls back to heuristics if direct keys not found."""
        if torch is None:
            raise RuntimeError("Torch unavailable for dimension inference")
        state_path = Path(actor_dir) / 'actor.pt'
        if not state_path.exists():
            raise FileNotFoundError(f"actor.pt not found at {state_path}")
        sd = torch.load(str(state_path), map_location='cpu')
        obs_dim = None
        act_dim = None
        # Try common linear layer keys for MLPBase first layer
        for k, v in sd.items():
            if isinstance(v, torch.Tensor) and v.ndim == 2:
                # action head
                if 'act.action_out' in k and 'weight' in k and act_dim is None:
                    act_dim = int(v.shape[0])
                # input layer weight may have shape (hidden, obs_dim)
                if ('base.mlp' in k or 'base' in k) and 'weight' in k and obs_dim is None:
                    if v.shape[1] > 8:  # heuristic threshold
                        obs_dim = int(v.shape[1])
            if obs_dim is not None and act_dim is not None:
                break
        if obs_dim is None:
            # Fallback: search max second dimension among 2D weights
            candidates = [v.shape[1] for v in sd.values() if isinstance(v, torch.Tensor) and v.ndim == 2]
            if candidates:
                obs_dim = int(max(candidates))
        if act_dim is None:
            # Fallback: search min first dimension among action-like layers
            candidates = [v.shape[0] for v in sd.values() if isinstance(v, torch.Tensor) and v.ndim == 2]
            if candidates:
                act_dim = int(min(candidates))
        if obs_dim is None or act_dim is None:
            raise RuntimeError("Could not infer obs/action dims from state dict")
        print(f"[CLIENT {self.device_id}] inferred dims from state dict: obs_dim={obs_dim} action_dim={act_dim}")
        return obs_dim, act_dim


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--device-id', required=True)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=1883)
    ap.add_argument('--rul-model', default='')
    ap.add_argument('--actor-model', default='')
    ap.add_argument('--actor-dir', default='')
    ap.add_argument('--fresh-actor', action='store_true', help='Start training with a freshly initialized actor (ignore any actor.pt)')
    args = ap.parse_args()

    client = EdgeClient(args.device_id, args.host, args.port, args.rul_model, args.actor_model, args.actor_dir, fresh_actor=args.fresh_actor)
    client.start()
