#!/usr/bin/env python3
"""
Local device client entry point that reuses EdgeClient implementation.
Matches the iot_demo.md naming.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Ensure project root on path so relative import works when run directly
ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.iot.edge_client import EdgeClient

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device-id', required=True)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=1883)
    ap.add_argument('--rul-model', default='')
    ap.add_argument('--actor-model', default='')
     # Directory containing actor.pt (pretrained RL actor); built lazily if provided
    ap.add_argument('--actor-dir', default='/home/lwh/Documents/Code/results/async_schedule/rul_schedule/mappo/threshold_7/wandb/run-20250503_002045-r5psc472/files')
    args = ap.parse_args()
    client = EdgeClient(args.device_id, args.host, args.port, args.rul_model, args.actor_model, args.actor_dir)
    client.start()

if __name__ == '__main__':
    main()
