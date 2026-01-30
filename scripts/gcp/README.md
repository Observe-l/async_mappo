# GCP (MQTT) distributed training

This folder contains a minimal distributed training scaffold:

- **Broker / server**: runs environment + critic update
- **Edge clients**: run RUL predictor + actor inference + local actor update (separate policy per device)

Topics are **namespaced** (default prefix `gcp`) to avoid colliding with the existing demo (`demo/*`).

## Prereqs

- MQTT broker (e.g. Mosquitto) reachable by both server and clients.
- Python deps:
  - `paho-mqtt`
  - `torch`
  - `tensorflow` (only required on edge clients if using TF RUL predictor)

## Run

Start 4 edge clients (on any machines):

```bash
python3 scripts/iot/edge_client.py --device-id edge-00 --host <BROKER_IP> --port 1883 --topic-prefix gcp
python3 scripts/iot/edge_client.py --device-id edge-01 --host <BROKER_IP> --port 1883 --topic-prefix gcp
python3 scripts/iot/edge_client.py --device-id edge-02 --host <BROKER_IP> --port 1883 --topic-prefix gcp
python3 scripts/iot/edge_client.py --device-id edge-03 --host <BROKER_IP> --port 1883 --topic-prefix gcp
```

Then start server (on broker machine or any machine):

```bash
python3 scripts/gcp/run_gcp_server_mqtt.py --host <BROKER_IP> --port 1883 --topic-prefix gcp --num-agents 4 --max-steps 200
```

## MQTT username/password (for deployment)

Local tests typically run without auth. For secured brokers, enable auth explicitly:

- Server: add `--mqtt-auth --mqtt-username admin --mqtt-password mailstrup123456`
- Client: add `--mqtt-auth --mqtt-username admin --mqtt-password mailstrup123456`

## Data/model locations

- RUL model: `models/model/`
- CMAPSS engine data: `models/cisco_engine_data/`
- Results: `results/<exp_type>/async_mappo/<timestamp>/`
