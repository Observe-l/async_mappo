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
from gymnasium.spaces import Discrete, Box, Dict
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
    def __init__(self, device_id: str, host: str, port: int, rul_model: str = '', actor_model: str = '', actor_dir: str = ''):
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

        # Immediate attempt to build pretrained RL actor BEFORE connecting (ensures startup log appears first)
        if self.actor is None and self.actor_dir:
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
        self.client.connect(host, port, keepalive=30)

    def start(self):
        self.client.loop_forever()

    def on_connect(self, client, userdata, flags, rc):
        topic = f"demo/obs/{self.device_id}"
        client.subscribe(topic, qos=1)
        print(f"[CLIENT {self.device_id}] connected rc={rc}, subscribed {topic}")

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
        except Exception:
            return
        step_id = payload.get('step_id')
        agent_id = payload.get('agent_id')
        seq = payload.get('seq')
        env_vec = payload.get('env_obs', {}).get('vector', [])
        sensor_vec = payload.get('sensor', {}).get('feature_vector', [])
        meta = payload.get('meta', {}) or {}
        action_dim = int(meta.get('action_dim') or 0)
        try:
            print(f"[CLIENT {self.device_id}] recv step={step_id} agent={agent_id} seq={seq} action_dim={action_dim}")
        except Exception:
            pass

        # Inference timings (simple)
        t0 = time.time()
        rul = self.infer_rul(sensor_vec)
        t1 = time.time()
        action = self.infer_action(env_vec, rul, action_dim)
        t2 = time.time()

        reply = {
            'step_id': step_id,
            'agent_id': agent_id,
            'seq': seq,
            'rul': float(rul),
            'action': int(action),
            'inference_ms': float((t2 - t0) * 1000.0),
            'device_id': self.device_id,
        }
        client.publish(f"demo/act/{self.device_id}", json.dumps(reply), qos=1, retain=False)
        try:
            print(f"[CLIENT {self.device_id}] send step={step_id} agent={agent_id} seq={seq} action={int(action)} rul={float(rul):.2f}")
        except Exception:
            pass

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
        # PyTorch R_Actor path (built from actor_dir)
        if self.actor is not None:
            try:
                return int(self.actor.act([float(rul)] + list(env_vec)))
            except Exception as e:
                print(f"[CLIENT {self.device_id}] actor inference error: {e}")
                # fall through to other modes
        # Random fallback spanning full factory range
        if action_dim and action_dim > 0:
            return random.randrange(0, int(action_dim))
        return random.randrange(0, 5)

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
    args = ap.parse_args()

    client = EdgeClient(args.device_id, args.host, args.port, args.rul_model, args.actor_model, args.actor_dir)
    client.start()
