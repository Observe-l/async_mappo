# SUMO RL Scheduling MQTT Demo Guide

This guide explains how to run the distributed inference demo using MQTT with a central server and multiple edge clients.

## Prerequisites

- OS: Linux (tested)
- Python environment: the repo's environment (e.g., conda env `default`) with project dependencies installed
- MQTT broker: a local broker on `127.0.0.1:1883` (e.g., Mosquitto)
- Trained actor weights: `actor.pt` from your training run
  - Example path: `/home/lwh/Documents/Code/results/async_schedule/rul_schedule/mappo/threshold_7/wandb/run-20250503_002045-r5psc472/files/actor.pt`

Recommended packages:
- `paho-mqtt` for MQTT clients
- `torch` (PyTorch)
- `tensorflow` (optional; used for the RUL predictor fallback on the edge)

Optional (for MySQL logging in this MQTT demo):
- A local MySQL server reachable at `127.0.0.1:3306`
- Python driver: `mysql-connector-python`
  - You can install it with: `python3 -m pip install --user mysql-connector-python`
  - The demo will gracefully disable DB logging if it cannot connect.

## Paths and environment

From the repo root, ensure the project is on `PYTHONPATH` when running scripts:

```bash
# Example: activate your environment
source /home/lwh/anaconda3/etc/profile.d/conda.sh
conda activate default

# (Optional) ensure repo is on sys.path; scripts already do this automatically
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

## Start the MQTT broker

If you have Mosquitto installed locally, start it (or verify it's running on port 1883). The demo assumes `127.0.0.1:1883`.

## Launch edge clients (4 terminals)

Open four terminals and run one client per device. Replace `ACTOR_DIR` with your trained run folder containing `actor.pt`.

```bash
# Terminal 1
python3 scripts/iot/edge_client.py --device-id edge-00 --host 192.168.1.204 --fresh-actor

# Terminal 2
python3 scripts/iot/edge_client.py --device-id edge-01 --host 192.168.1.204 --fresh-actor

# Terminal 3
python3 scripts/iot/edge_client.py --device-id edge-02 --host 192.168.1.204 --fresh-actor

# Terminal 4
python3 scripts/iot/edge_client.py --device-id edge-03 --host 192.168.1.204 --fresh-actor
```

You should see at startup, before connecting to MQTT:
- A line indicating inferred dimensions, e.g. `obs_dim=243 action_dim=50`
- A success line: `Pretrained RL actor loaded successfully from ... actor.pt ...`

If a client times out waiting for messages, that’s normal until the server starts.

## Launch the server (MQTT GUI or headless debug)

GUI mode (recommended to visualize SUMO and truck info):

```bash
python3 scripts/render/run_demo_schedule_mqtt.py --mode gui --mysql-enable\
  --num-agents 4 --max-steps 2000
  
```

Debug (headless) mode:

```bash
python3 scripts/render/run_demo_schedule_mqtt.py --mode debug --mysql-enable\
  --num-agents 4 --max-steps 2000 \
```

Notes:
- The server sends observations to edge clients via round‑robin.
- Server uses remote decisions only. On timeout, it repeats the last action for that agent; if none, it chooses a random fallback.
- GUI shows per‑truck Status, Distance, Destination, Road, and RUL (RUL is sourced from the edge client predictions).
- MySQL logging: when enabled, a new database named `sumo_YYYYMMDD_HHMMSS` is created per run with tables `truck_0`..`truck_N`.
  - Each row includes `sim_time`, `rul`, `driving_distance_km`, `state`, `destination`, `loaded_goods`, `weight`, `total_transported`.
  - `sim_time` is a formatted string `YYYY-MM-DD HH:MM:SS`, computed as base date `2025-11-01 00:00:00` plus the SUMO simulation time (in seconds).
  - Inserts happen on every RL/pickup decision and approximately every 200 simulation seconds.

## Expected logs

Server:
- `[ENV->MQTT] send step=... agent=truck_X seq=... to=edge-YY action_dim=...`
- `[MQTT->ENV] recv ... from=edge-YY action=... rul=...`
- `[SERVER] remote action step=... agent=truck_X device=edge-YY action=...`

Edge clients:
- `[CLIENT edge-YY] Pretrained RL actor loaded successfully from ... actor.pt obs_dim=243 action_dim=50 ...`
- `[CLIENT edge-YY] recv step=... agent=truck_X seq=... action_dim=...`
- `[CLIENT edge-YY] send step=... agent=truck_X seq=... action=... rul=...`

Note: The observation vector is always `[RUL] + env` of length 243. The server reconstructs this for training using the RUL returned by the edge, and the edge actor always consumes `[RUL] + env` for inference.

## Troubleshooting

- No data in GUI columns:
  - Ensure edge clients are running and connected.
  - The GUI now refreshes continuously; if blank initially, wait a moment as SUMO advances to operable states.
  - Check for `[ENV] timeout ...` messages; if frequent, verify broker availability and network.

- Edge actor fails to load:
  - Verify `actor.pt` path and that the file exists.
  - Ensure your Python env can import local packages (`onpolicy`). The client script adds repo root to `sys.path` automatically.

- SUMO issues:
  - GUI requires SUMO (libsumo/traci). Ensure it’s installed and the SUMO network used by the demo is available.

## Local non‑MQTT demo (reference)

To run the original local demo without MQTT (for comparison):

```bash
python3 scripts/render/run_demo_schedule.py --mode gui --actor-dir /path/to/your/run/files
```

This uses the same policy architecture but infers actions locally.

---

If you want, I can add a convenience script to start all 4 edge clients automatically in separate terminals or tmux sessions.