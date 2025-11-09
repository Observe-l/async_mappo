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
                self._build_pretrained_actor(obs_len=inferred_obs, action_dim=inferred_act)
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

        # If actor not yet built (could not infer dims), attempt build now using current obs/action sizes
        if (not self._actor_built) and (self.actor is None) and self.actor_dir and action_dim > 0:
            try:
                self._build_pretrained_actor(obs_len=len(env_vec)+1, action_dim=action_dim)
                self._actor_built = True
            except Exception as e:
                print(f"[CLIENT {self.device_id}] deferred actor build failed: {e}")

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
        if self.actor is not None and hasattr(self, '_is_r_actor') and getattr(self, '_is_r_actor'):
            try:
                import torch
                v = list(env_vec) + [float(rul)]
                x = torch.tensor(v, dtype=torch.float32).unsqueeze(0)
                rnn_states = torch.zeros((1, getattr(self, '_recurrent_N', 6), getattr(self, '_hidden_size', 64)), dtype=torch.float32)
                masks = torch.ones((1, 1), dtype=torch.float32)
                with torch.no_grad():
                    actions, _log_probs, _rnn = self.actor(x, rnn_states, masks, available_actions=None, deterministic=False)
                return int(actions.view(-1)[0].item()) % max(1, int(action_dim or 1))
            except Exception as e:
                print(f"[CLIENT {self.device_id}] actor inference error: {e}")
                # fall through to other modes
        # TorchScript actor path
        if self.actor is not None and not getattr(self, '_is_r_actor', False):
            try:
                import torch
                v = list(env_vec) + [float(rul)]
                x = torch.tensor(v, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    y = self.actor(x)
                return int(torch.argmax(y, dim=-1).view(-1)[0].item()) % max(1, int(action_dim or 1))
            except Exception:
                pass
        # Random fallback spanning full factory range
        if action_dim and action_dim > 0:
            return random.randrange(0, int(action_dim))
        return random.randrange(0, 5)

    def _build_pretrained_actor(self, obs_len: int, action_dim: int):
        """Build the R_MAPPO R_Actor like in run_demo_schedule.py and load actor.pt from actor_dir.
        obs_len should include RUL (env_vec + [rul])."""
        if torch is None:
            raise RuntimeError("Torch not available for pretrained actor.")
        from onpolicy.config import get_config  # type: ignore
        from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor  # type: ignore
        # Minimal gym-like spaces with Dict support (R_Actor expects Dict with 'global_obs')
        try:
            from gym.spaces import Box, Discrete, Dict as DictSpace  # type: ignore
        except Exception:
            # Fallback stubs mimicking gym.spaces
            class Box:  # type: ignore
                def __init__(self, low=None, high=None, shape=None, dtype=None):
                    self.shape = shape
            class Discrete:  # type: ignore
                def __init__(self, n):
                    self.n = n
            class DictSpace(dict):  # type: ignore
                @property
                def spaces(self):
                    return self
                # Ensure util.get_shape_from_obs_space sees 'Dict'
                def __repr__(self):
                    return f"Dict({dict.__repr__(self)})"
        parser = get_config()
        args = parser.parse_args([])
        # Provide additional defaults required by R_Actor/MLPBase initialization
        if not hasattr(args, 'gain'):
            args.gain = 0.01
        if not hasattr(args, 'use_policy_active_masks'):
            args.use_policy_active_masks = True
        if not hasattr(args, 'use_policy_vhead'):
            args.use_policy_vhead = False
        if not hasattr(args, 'activation_id'):
            args.activation_id = 1
        if not hasattr(args, 'use_orthogonal'):
            args.use_orthogonal = True
        if not hasattr(args, 'hidden_size'):
            args.hidden_size = 64
        if not hasattr(args, 'layer_N'):
            args.layer_N = 2
        # Provide overrides mirroring TrainedPolicy in run_demo_schedule.py
        if not hasattr(args, 'grid_goal'):
            args.grid_goal = False
        if not hasattr(args, 'goal_grid_size'):
            args.goal_grid_size = 4
        args.algorithm_name = 'mappo'
        args.use_recurrent_policy = True
        args.use_naive_recurrent_policy = False
        args.recurrent_N = 6  # match training recurrent_N
        # Ensure attention flags exist (expected by MLPBase)
        if not hasattr(args, 'use_attn_internal'):
            args.use_attn_internal = True
        if not hasattr(args, 'use_cat_self'):
            args.use_cat_self = True
        # Preserve for rnn state shapes
        self._recurrent_N = int(getattr(args, 'recurrent_N', 6))
        self._hidden_size = int(getattr(args, 'hidden_size', 64))
        # Build spaces
        try:
            import numpy as _np
        except Exception:
            _np = None
        if _np is not None and 'Box' in str(Box):
            low = -1e9
            high = 1e9
            obs_space = DictSpace({
                'global_obs': Box(low=low, high=high, shape=(int(obs_len),), dtype=_np.float32)
            })
        else:
            # Fallback without numpy types
            obs_space = DictSpace({
                'global_obs': Box(shape=(int(obs_len),))
            })
        act_space = Discrete(int(action_dim or 1))
        # Debug: show obs_space class name before constructing actor
        try:
            print(f"[CLIENT {self.device_id}] constructing R_Actor with obs_space_cls={obs_space.__class__.__name__} action_space_cls={act_space.__class__.__name__}")
        except Exception:
            pass
        # Create actor and load weights (verbose diagnostics)
        try:
            actor = R_Actor(args, obs_space, act_space, device=torch.device('cpu'))
        except Exception as e:
            print(f"[CLIENT {self.device_id}] R_Actor ctor failed: {type(e).__name__}: {e}")
            raise
        state_path = Path(self.actor_dir) / 'actor.pt'
        if not state_path.exists():
            raise FileNotFoundError(f"actor.pt not found at {state_path}")
        try:
            sd = torch.load(str(state_path), map_location='cpu')
            missing, unexpected = actor.load_state_dict(sd, strict=False)
            self.actor = actor
            self._is_r_actor = True
            msg = (f"[CLIENT {self.device_id}] Pretrained RL actor loaded successfully from {state_path} "
                   f"obs_dim={obs_len} action_dim={action_dim} recurrent_N={self._recurrent_N} hidden_size={self._hidden_size}")
            if missing or unexpected:
                msg += f" (non-strict load: missing={len(missing)} unexpected={len(unexpected)})"
            print(msg)
        except Exception as e:
            print(f"[CLIENT {self.device_id}] failed to load pretrained RL actor state dict: {type(e).__name__}: {e}")

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
