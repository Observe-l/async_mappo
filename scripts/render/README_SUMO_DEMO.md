# SUMO Evaluation Demo

This demo visualizes a trained async-MAPPO policy—originally trained against the offline `rul_schedule` environment—acting in a live SUMO simulation. The policy now drives decisions directly from a SUMO-backed evaluation environment while SUMO routes and animates trucks between ParkingAreas `Factory0..Factory49`.

## Prerequisites
- SUMO installed, with `libsumo` importable:
  - `python -c "import libsumo"`
- A SUMO network configured with ParkingAreas named `Factory0..Factory49` and routes like `FactoryX_to_FactoryX` for initialization. See `onpolicy/envs/rul_schedule/core.py` for reference.
- A trained model checkpoint directory containing `actor.pt` (e.g., from a `results/.../` or wandb run files directory).

## Quick start

GUI (recommended for visualization):

```bash
python scripts/render/render_sumo_schedule.py \
  --actor_dir /home/lwh/Documents/Code/results/async_schedule/rul_schedule/mappo/threshold_7/wandb/run-20250503_002045-r5psc472/files \
  --sumo_cfg /home/lwh/Documents/Code/RL-Scheduling/map/sg_map/osm.sumocfg \
  --num_agents 12 \
  --episodes 1 \
  --max_steps 1000
```

Headless + debug (faster, prints detailed logs and caps sim time to 10,000s):

```bash
python scripts/render/render_sumo_schedule.py \
  --actor_dir /home/lwh/Documents/Code/results/async_schedule/rul_schedule/mappo/threshold_7/wandb/run-20250503_002045-r5psc472/files \
  --sumo_cfg /home/lwh/Documents/Code/RL-Scheduling/map/sg_map/osm.sumocfg \
  --num_agents 4 \
  --episodes 1 \
  --max_steps 20 \
  --headless \
  --debug
```

## Useful options
- `--actor_dir`: Directory containing `actor.pt`.
- `--sumo_cfg`: Path to your SUMO `.sumocfg` file.
- `--num_agents`: Number of demo trucks to control/visualize.
- `--episodes`: Number of episodes to run (default 50).
- `--max_steps`: Max decision steps per episode (termination condition).
- `--headless`: Run SUMO without GUI.
- `--debug`: Verbose logs; also forces `--sim_time_limit` to 10,000 seconds.
- `--sim_time_limit`: Upper bound on SUMO simulation time per episode (default 7 days).
- `--rul_state`: Append RUL signal to observations. Leave unset to match checkpoints trained without the extra feature.

## Notes
- Maintenance action mapping: when a truck is in maintenance, it is routed to `Factory0` (workshop) and remains inactive until recovery.
- wandb is disabled for this demo to avoid external logging.
- The demo keeps SUMO and the policy env loosely synchronized; it does not mirror factory inventories in SUMO.
- If routing to a Factory fails, ensure your network defines ParkingAreas `Factory0..Factory49` in the additional file loaded by your `.sumocfg`.
