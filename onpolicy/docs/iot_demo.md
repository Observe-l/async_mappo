
# Local MQTT-Based Distributed Inference Framework for SUMO-RL Demo

## 1. System Overview

This document specifies the design, communication logic, and requirements for a **local MQTT-based distributed inference demo**.  
All components run on the **same computer (localhost, 127.0.0.1)**, allowing simulation and testing without external devices or network dependencies.

The system connects a **SUMO + RL simulation environment** with **multiple lightweight MQTT inference clients**.  
Each client simulates an independent inference worker that runs locally, performs RUL prediction and RL Actor inference, and returns results back to the SUMO environment asynchronously.

---

## 2. System Framework

### 2.1 Components

| Role | Process | Description |
|------|----------|-------------|
| **MQTT Server / Environment Host** | `env_server.py` | Hosts the Mosquitto MQTT broker and runs the SUMO simulation with the RL environment. Publishes environment observations and collects inference results from clients. |
| **MQTT Clients (×4)** | `device_client.py` | Simulate four independent inference agents. Each loads two local models:<br>① **RUL Predictor** (e.g., `gcpatr_traced.pt`)<br>② **RL Actor** (e.g., `actor_traced.pt`)<br>Upon receiving messages, each performs local inference and publishes results back to the environment. |

### 2.2 Logical Diagram

```

┌─────────────────────────────────────────────────────────────────────┐
│                       Localhost (127.0.0.1)                         │
│─────────────────────────────────────────────────────────────────────│
│   Mosquitto Broker  (port 1883, QoS=1)                              │
│                                                                     │
│   ┌────────────────────────────────────────────────────────────┐    │
│   │ Environment Process (SUMO + RL Bridge)                      │    │
│   │ - Publishes observations + sensor data                      │    │
│   │ - Waits for client results (rul + action)                   │    │
│   │ - Updates SUMO simulation                                   │    │
│   └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│   ┌────────────────────┬────────────────────┬────────────────────┬────────────────────┐
│   │ MQTT Client edge-01 │ MQTT Client edge-02 │ MQTT Client edge-03 │ MQTT Client edge-04 │
│   │ - Loads models      │ - Loads models      │ - Loads models      │ - Loads models      │
│   │ - Subscribes topic  │ - Subscribes topic  │ - Subscribes topic  │ - Subscribes topic  │
│   │ - Inference & reply │ - Inference & reply │ - Inference & reply │ - Inference & reply │
│   └────────────────────┴────────────────────┴────────────────────┴────────────────────┘
└─────────────────────────────────────────────────────────────────────┘

````

---

## 3. Communication Flow

### 3.1 Overview

All communication occurs through **MQTT over localhost**, with one broker (`127.0.0.1:1883`).  
The environment process acts as the **publisher of tasks** and the **subscriber for inference results**, while each client performs the opposite role.

### 3.2 Step-by-Step Message Flow per Simulation Step

| Step | Actor | Description | Topic | Blocking Behavior |
|------|--------|-------------|--------|-------------------|
| **1** | Environment | Publishes observation (SUMO state + sensor data) to one client in a round-robin pattern. | `demo/obs/{device_id}` | Non-blocking publish |
| **2** | Client | Receives message, performs inference:<br>• `rul = RUL_Predictor(sensor)`<br>• `action = RL_Actor([env_obs, rul])` | — | Local compute |
| **3** | Client | Publishes `{rul, action, step_id, agent_id, seq}` back to the environment. | `demo/act/{device_id}` | Non-blocking publish |
| **4** | Environment | Waits asynchronously for the response using `threading.Event()` with a short timeout (e.g., 150 ms). | Subscribed to all `demo/act/*` | Asynchronous wait |
| **5** | Environment | Once the event is triggered, integrates results into SUMO step. If timeout, uses fallback values. | — | Non-blocking continuation |

### 3.3 Synchronization and Non-Blocking Logic

- **Dedicated MQTT Thread:**  
  The environment bridge runs MQTT I/O (`client.loop_start()`) in a background thread to stay responsive while SUMO advances.
  
- **Pending Dictionary:**  
  Each request `(device_id, agent_id, step_id, seq)` has a corresponding entry in `PENDING` with a threading event.

- **Asynchronous Wait:**  
  SUMO loop calls `event.wait(timeout)` for each agent. When a matching message arrives, the event is set, and the step continues.

- **Timeout Fallback:**  
  If no response is received within the timeout, default values are applied:
  ```python
  default_rul(agent_id) = 1e6
  default_action(agent_id) = [0.0, 0.0]


The simulation continues without stalling.

---

## 4. MQTT Topics and Message Format

### 4.1 Topics

| Direction   | Topic Pattern          | Purpose                                      |
| ----------- | ---------------------- | -------------------------------------------- |
| PC → Client | `demo/obs/{device_id}` | Send environment + sensor data for inference |
| Client → PC | `demo/act/{device_id}` | Return predicted RUL and RL action           |

### 4.2 Message Schema

#### Observation Message (Environment → Client)

```json
{
  "step_id": 101,
  "agent_id": "truck_01",
  "seq": 10001,
  "timestamp": 1731050400.321,
  "env_obs": { "vector": [0.12, 0.45, 0.78] },
  "sensor": { "feature_vector": [0.2, 0.3, 0.5] }
}
```

#### Action Message (Client → Environment)

```json
{
  "step_id": 101,
  "agent_id": "truck_01",
  "seq": 10001,
  "rul": 850.5,
  "action": [0.12, 0.34],
  "inference_ms": 4.7,
  "device_id": "edge-02"
}
```

---

## 5. Implementation Requirements

### 5.1 Environment Bridge (Server)

* Runs SUMO simulation and RL environment.
* Maintains a round-robin dispatcher across four clients.
* Handles both publishing (to clients) and subscription (for results).
* Maintains `PENDING` dictionary for ongoing requests.
* Integrates responses into SUMO vehicle scheduling decisions.

### 5.2 Inference Clients

Each client:

1. Subscribes to its designated topic (`demo/obs/{device_id}`).
2. Loads local models for RUL and Actor inference.
3. Processes observation messages immediately on arrival.
4. Publishes results asynchronously to `demo/act/{device_id}`.

---

## 6. Communication Settings

| Parameter         | Value                      |
| ----------------- | -------------------------- |
| MQTT Broker       | Mosquitto                  |
| Host              | `127.0.0.1`                |
| Port              | `1883`                     |
| QoS               | 1 (At-least-once delivery) |
| Retain            | False                      |
| Authentication    | Disabled                   |
| Keepalive         | 30 s                       |
| Timeout per step  | 150 ms                     |
| Number of clients | 4                          |
| Broker launch     | `mosquitto -v`             |

---

## 7. Functional Requirements

| ID | Function               | Description                                                                                  |
| -- | ---------------------- | -------------------------------------------------------------------------------------------- |
| F1 | Publish observation    | The environment publishes a JSON message per agent per step.                                 |
| F2 | Local inference        | Each client executes local RUL + Actor inference upon receiving data.                        |
| F3 | Publish result         | Each client publishes its computed RUL + action result back to the environment.              |
| F4 | Asynchronous wait      | The environment waits for responses via threading events, ensuring SUMO loop is not blocked. |
| F5 | Timeout fallback       | Missing responses trigger default RUL/action values.                                         |
| F6 | Round-robin assignment | Tasks are evenly distributed across four clients.                                            |
| F7 | Logging                | All components log publish/receive events for debugging and profiling.                       |

---

## 8. Non-Functional Requirements

| Category            | Requirement                                                                         |
| ------------------- | ----------------------------------------------------------------------------------- |
| **Performance**     | End-to-end inference latency <150 ms per simulation step.                           |
| **Scalability**     | Support up to 8 local clients by extending topic list.                              |
| **Reliability**     | MQTT QoS=1 ensures message delivery; duplicates are ignored using `(step_id, seq)`. |
| **Usability**       | No TLS or credentials; all components run locally in the same process environment.  |
| **Reproducibility** | Each run produces consistent results with the same random seed.                     |
| **Maintainability** | Components (broker, server, clients) are modular and independently testable.        |

---

## 9. Expected Outcomes

| Aspect             | Expected Result                                                                 |
| ------------------ | ------------------------------------------------------------------------------- |
| Connectivity       | All clients connect to the broker on `127.0.0.1:1883`.                          |
| Message Flow       | Environment → Clients → Environment loop works reliably.                        |
| Inference          | Each client receives input, performs inference, and responds with RUL + action. |
| Parallel Execution | Multiple clients run concurrently without conflicts.                            |
| Synchronization    | Environment waits for all responses or timeouts, never blocking SUMO.           |
| Logging            | Each process prints clear, timestamped events for every request and response.   |

---

## 10. Test Plan

### 10.1 Test Setup

* **Host:** Single computer (localhost).
* **Processes:**

  * 1× `mosquitto` (broker)
  * 1× `env_server.py` (environment bridge)
  * 4× `device_client.py` (clients)
* **Environment:**

  * Python ≥3.9
  * `paho-mqtt` installed (`pip install paho-mqtt`)
  * Dummy inference logic (random values)

### 10.2 Test Cases

| ID | Scenario              | Procedure                                      | Expected Outcome                          |
| -- | --------------------- | ---------------------------------------------- | ----------------------------------------- |
| T1 | Broker initialization | Run `mosquitto -v`                             | Broker starts on port 1883                |
| T2 | Client connection     | Launch 4 clients with IDs `edge-01`..`edge-04` | All print “connected rc=0”                |
| T3 | Observation broadcast | Server sends one observation per agent         | Corresponding client prints received data |
| T4 | Action response       | Clients publish back valid result              | Server logs received `{rul, action}`      |
| T5 | Timeout handling      | Stop one client                                | Server applies fallback and continues     |
| T6 | Multi-step loop       | Run 100 SUMO steps                             | Simulation continues without freeze       |
| T7 | Response integrity    | Check `(step_id, seq)` mapping                 | Matches and no duplication                |
| T8 | Performance profiling | Measure average inference turnaround           | <150 ms average per step                  |

---

## 11. Future Extensions

| Area                        | Enhancement                                                               |
| --------------------------- | ------------------------------------------------------------------------- |
| **Multi-device simulation** | Add process-level isolation to simulate LAN-based distributed deployment. |
| **Security**                | Introduce authentication and TLS after local tests.                       |
| **Shared Subscriptions**    | Automate load balancing across clients.                                   |
| **Monitoring**              | Implement lightweight heartbeats for connection status.                   |
| **Model Optimization**      | Replace dummy random inference with real TorchScript models.              |
| **Batch Processing**        | Send batched multi-agent observations to reduce MQTT overhead.            |

---

## 12. Summary

This local MQTT-based demo establishes a **non-blocking, asynchronous inference architecture** for SUMO-RL experiments.
By simulating four independent inference clients on the same computer, it provides a reproducible test environment to validate:

* **Reliable message exchange** using MQTT
* **Event-based synchronization** without blocking SUMO
* **Distributed inference structure** that can later be extended to physical IoT devices

This framework ensures that future deployments (e.g., on Jetsons or remote devices) will seamlessly integrate without changing the communication logic.

---
