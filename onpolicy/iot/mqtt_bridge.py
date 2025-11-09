"""
MQTT bridge for distributed inference (PC host side).

Implements a lightweight request/response layer over MQTT to offload
RUL prediction + RL actor inference to IoT devices.

Topics:
  - Publish requests:  demo/obs/{device_id}
  - Receive replies:   demo/act/{device_id}

Messages are JSON and carry identifiers: step_id, agent_id, seq.
QoS=1 is used for at-least-once delivery; duplicates are ignored via ids.

Also provides a "mock" mode that simulates remote responses locally for
testing without a broker or clients.
"""
from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import paho.mqtt.client as mqtt  # type: ignore
except Exception:
    mqtt = None  # Will be validated at runtime when mode == "mqtt"


@dataclass
class BridgeConfig:
    host: str = "127.0.0.1"
    port: int = 1883
    keepalive: int = 30
    qos: int = 1
    timeout_ms: int = 150
    devices: Tuple[str, ...] = ("edge-00", "edge-01", "edge-02", "edge-03")
    mode: str = "mqtt"  # "mqtt" or "mock"


class MqttBridge:
    def __init__(self, cfg: BridgeConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._seq = 0
        self._rr_idx = 0
        self._pending: Dict[Tuple[int, str, int], Tuple[threading.Event, dict]] = {}
        # Keep as object to avoid type resolution when paho is absent
        self._client = None  # type: Optional[object]

        if self.cfg.mode == "mqtt":
            if mqtt is None:
                raise RuntimeError("paho-mqtt not installed but mode='mqtt'")
            self._setup_client()

    # --------------- MQTT setup and callbacks ---------------
    def _setup_client(self):
        c = mqtt.Client(client_id=f"env-bridge-{random.randrange(1,1_000_000)}", clean_session=True)
        c.on_connect = self._on_connect
        c.on_message = self._on_message
        # No TLS/auth for LAN tests
        c.connect(self.cfg.host, self.cfg.port, keepalive=self.cfg.keepalive)
        # Start loop in background thread
        c.loop_start()
        self._client = c

    def _on_connect(self, client, userdata, flags, rc):
        # Subscribe to all action topics
        for dev in self.cfg.devices:
            topic = f"demo/act/{dev}"
            client.subscribe(topic, qos=self.cfg.qos)

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            step_id = int(payload.get("step_id"))
            agent_id = str(payload.get("agent_id"))
            seq = int(payload.get("seq"))
            key = (step_id, agent_id, seq)
        except Exception:
            return
        # Idempotent complete
        with self._lock:
            pending = self._pending.get(key)
            if not pending:
                return  # Unknown or already handled
            ev, _container = pending
            # Store the full payload
            self._pending[key] = (ev, payload)
            ev.set()
        try:
            dev = str(payload.get("device_id", "?"))
            act = payload.get("action")
            rul = payload.get("rul")
            print(f"[MQTT->ENV] recv step={step_id} agent={agent_id} seq={seq} from={dev} action={act} rul={rul}")
        except Exception:
            pass

    # --------------- Public API ---------------
    def next_device(self) -> str:
        with self._lock:
            dev = self.cfg.devices[self._rr_idx % len(self.cfg.devices)]
            self._rr_idx += 1
            return dev

    def new_seq(self) -> int:
        with self._lock:
            self._seq = (self._seq + 1) & 0x7FFFFFFF
            return self._seq

    def request(self, *, step_id: int, agent_id: str, env_obs, sensor, action_dim: Optional[int] = None,
                decision_type: str = "rl", pickup_candidates: Optional[list] = None) -> Tuple[float, Optional[int], bool, str]:
        """Send one observation to a device and wait for result.

        Returns (rul, action, ok, device_id). If timeout, ok=False and defaults provided.
        """
        device_id = self.next_device()
        return self.request_to(device_id=device_id, step_id=step_id, agent_id=agent_id, env_obs=env_obs, sensor=sensor,
                               action_dim=action_dim, decision_type=decision_type, pickup_candidates=pickup_candidates)

    def request_to(self, *, device_id: str, step_id: int, agent_id: str, env_obs, sensor, action_dim: Optional[int] = None,
                   decision_type: str = "rl", pickup_candidates: Optional[list] = None) -> Tuple[float, Optional[int], bool, str]:
        """Send request to a specific device id (no round robin)."""
        seq = self.new_seq()
        key = (step_id, agent_id, seq)

        if self.cfg.mode == "mock":
            # Simulate inference locally
            if decision_type == "pickup":
                cands = pickup_candidates if pickup_candidates else list(range(int(action_dim or 45)))
                action = int(random.choice(cands))
                rul = self.default_rul()
            else:
                rul = float(self._mock_rul(sensor))
                action = self._mock_action(env_obs, action_dim)
            return (rul, action, True, device_id)

        assert self._client is not None, "MQTT client not initialized"
        # Prepare payload (ensure lists not numpy); handle nested containers recursively
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
            # Basic scalars
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
            "timestamp": time.time(),
            "env_obs": {"vector": to_plain(env_obs)},
            "sensor": {"feature_vector": to_plain(sensor)},
            "meta": {
                "action_dim": int(action_dim) if action_dim is not None else None,
                "decision_type": decision_type,
                "pickup_candidates": to_plain(pickup_candidates) if pickup_candidates is not None else None
            },
        }
        ev = threading.Event()
        with self._lock:
            self._pending[key] = (ev, {})
        self._client.publish(f"demo/obs/{device_id}", json.dumps(payload), qos=self.cfg.qos, retain=False)
        try:
            print(f"[ENV->MQTT] send step={step_id} agent={agent_id} seq={seq} to={device_id} action_dim={action_dim}")
        except Exception:
            pass

        ok = ev.wait(self.cfg.timeout_ms / 1000.0)
        with self._lock:
            _ev2, res = self._pending.pop(key, (None, {}))
        if not ok or not res:
            # Timeout -> do NOT synthesize local decisions; let caller decide fallback strategy.
            try:
                print(f"[ENV] timeout step={step_id} agent={agent_id} seq={seq} device={device_id}")
            except Exception:
                pass
            return (self.default_rul(), None, False, device_id)
        try:
            rul = float(res.get("rul", self.default_rul()))
            action_val = res.get("action", None)
            if isinstance(action_val, list):
                action = int(action_val[-1])
            else:
                action = int(action_val) if action_val is not None else None
        except Exception:
            rul = self.default_rul()
            action = None
        try:
            dev_id = str(res.get("device_id", device_id))
            print(f"[ENV] got result step={step_id} agent={agent_id} seq={seq} device={dev_id} action={action} rul={rul}")
        except Exception:
            dev_id = str(res.get("device_id", device_id))
        return (rul, action, True, dev_id)

    # --------------- Defaults / mock ---------------
    def default_rul(self) -> float:
        return 9999.0

    def default_action(self, env_obs, action_dim: Optional[int] = None) -> int:
        # Try to infer discrete action space by length heuristics; fallback 0
        try:
            # If env_obs is a vector including first element as RUL, distances next; we cannot infer factory_num reliably.
            # Fallback to a small range.
            if action_dim is not None and action_dim > 0:
                import random as _rnd
                return int(_rnd.randrange(0, int(action_dim)))
            return 0
        except Exception:
            return 0

    def _mock_rul(self, sensor) -> float:
        return 500.0 + random.random() * 100.0

    def _mock_action(self, env_obs, action_dim: Optional[int]) -> int:
        # Random small integer; caller should clip to valid range if needed
        if action_dim is None or action_dim <= 0:
            action_dim = 5
        return int(random.randrange(0, int(action_dim)))
