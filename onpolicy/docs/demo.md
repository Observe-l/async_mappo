


# SUMO Reinforcement Learning Truck Scheduling GUI Demo

## 1. Overview

This demo presents an interactive **SUMO-based traffic system GUI** integrated with a **Reinforcement Learning (RL) scheduling agent**.  
The RL agent has been **pretrained offline** to coordinate and schedule multiple trucks efficiently in a logistics transport environment.

The demo allows users to visualize agent decisions in real-time through the **SUMO GUI**, with support for both **debug (headless)** and **visual (GUI)** modes.

---

## 2. Project Structure

| Component | Path | Description |
|------------|------|-------------|
| Training script | `train_schedule.py` | Script used for pretraining the RL model. |
| Training environment | `onpolicy/envs/rul_schedule/schedule.py` | Environment used during offline training. |
| Pretrained model | `/home/lwh/Documents/Code/results/async_schedule/rul_schedule/mappo/threshold_7/wandb/run-20251029_111639-ueqk7va5/files` | Directory containing trained RL weights and configurations. |
| GUI environment | `onpolicy/envs/rul_schedule/demo_schedule.py` | GUI-based environment used for this demo. |

---

## 3. Objectives

1. **Visualize RL-controlled truck scheduling** within the SUMO GUI.
2. Provide a **real-time information panel** for tracking a selected RL agent (truck).
3. Allow users to **switch between agents** for camera following and status display.
4. Maintain a **summary dashboard** showing information of all trucks.
5. Support **debug mode (non-GUI)** to ensure logic and vehicle motion correctness.
6. Provide a **README** detailing setup and execution steps.

---

## 4. Functional Requirements

### 4.1 Agent Tracking and Visualization
- During GUI execution, display a **control window** allowing the user to:
  - Select a specific RL agent/truck for **camera tracking**.
  - Switch tracking to other trucks dynamically.
- The control window should display the **selected agent’s information**:
  - Current **status** (`waiting`, `driving`, `arrived`, etc.)
  - **Total driving distance** (cumulative)
  - **Current destination**
  - **Current road ID** or name

### 4.2 Summary Information Window
- A separate window lists all trucks’ statuses in a structured table:
  - Truck ID
  - Status
  - Distance traveled
  - Assigned destination
  - Road information
- The table should refresh periodically (e.g., every simulation step).

### 4.3 Debug Mode (Non-GUI)
- When GUI is disabled, the demo should:
  - Run SUMO in **non-GUI mode**.
  - Print all truck states (ID, position, speed, status, etc.) to the terminal.
  - Verify that trucks follow the RL decisions correctly and complete their trips.

### 4.4 GUI Mode
- When enabled, the demo should:
  - Launch the **SUMO GUI**.
  - Automatically load the environment and agent policies.
  - Initialize the **tracking and info windows**.
  - Visualize real-time truck motion and scheduling decisions.

---

## 5. Non-Functional Requirements

| Category | Requirement |
|-----------|-------------|
| **Performance** | Simulation should run in real-time or faster than real-time in debug mode. |
| **Compatibility** | Must run under the local `default` Conda Python environment. No `bash -lc` calls. |
| **Usability** | Windows and displays should update smoothly and respond to user input. |
| **Maintainability** | Code should be modular — separate GUI logic, simulation logic, and RL agent loading. |
| **Logging** | Optional logging of simulation states for debugging or post-analysis. |

---

## 6. Implementation Details

### 6.1 Key Modules
- `demo_schedule.py`: Main environment class managing SUMO simulation and agent interaction.
- RL model loader: Automatically loads the pretrained weights from the specified path.
- GUI controller: Implements camera tracking and UI panels using `traci` or `PyQt` (depending on implementation choice).

### 6.2 Expected Behavior
- Trucks spawn and move according to RL agent policy.
- Selecting a truck centers the camera and updates the side panel.
- The information summary updates automatically at each simulation step.

---

## 7. Testing Plan

### 7.1 Test Environment
- SUMO version ≥ 1.17.0  
- Python ≥ 3.8  
- Conda environment: `default`  
- Required Python packages:  
  ```bash
  conda activate default
  ```

### 7.2 Test Cases

| Test ID | Description                                | Expected Result                                                         |
| ------- | ------------------------------------------ | ----------------------------------------------------------------------- |
| T1      | Run in debug mode (`sumo`, not `sumo-gui`) | Trucks move correctly and print state logs in terminal.                 |
| T2      | Run in GUI mode                            | SUMO GUI opens, trucks visible.                                         |
| T3      | Switch tracked truck from control window   | Camera follows the selected truck immediately.                          |
| T4      | Display truck info panel                   | Real-time info (status, distance, destination, road) updates correctly. |
| T5      | Display summary info panel                 | All trucks’ statuses visible and refreshing.                            |
| T6      | Model loading                              | RL weights load without error.                                          |
| T7      | Simulation termination                     | SUMO closes cleanly, logs saved if applicable.                          |

---

## 8. Execution Instructions (README)

### 8.1 Setup

```bash
conda activate default
pip install traci sumolib matplotlib numpy torch
```

### 8.2 Debug Mode (Headless)

```bash
conda activate default
pip install traci sumolib
python scripts/render/run_demo_schedule.py --mode debug --num-agents 4 --max-steps 200 --debug
```

* Runs SUMO without GUI.
* Logs truck states to terminal (status, distance, destination, road, speed).
* Useful for verifying logic and debugging.

### 8.3 GUI Mode

```bash
conda activate default
pip install traci sumolib
python scripts/render/run_demo_schedule.py --mode gui --num-agents 12 --max-steps 1000
```

* Launches SUMO GUI.
* Opens two panels:
  * **Agent Tracking Window**: pick a truck, shows status, distance, destination, road, RUL, cargo.
  * **Summary Information Window**: live table of all trucks.
* The camera follows the selected truck.

Notes:
- You can pass `--sumo-cfg /path/to/your.osm.sumocfg` if the default path differs.
- To use a trained policy, add `--actor-dir /path/to/wandb_or_results_dir` (expects `actor.pt`).

---

## 9. Acceptance Criteria

* ✅ All required windows appear and update correctly in GUI mode.
* ✅ Camera follows selected RL agent dynamically.
* ✅ Summary panel lists all trucks’ statuses.
* ✅ Debug mode runs SUMO simulation headlessly and prints valid outputs.
* ✅ Code runs successfully in local Conda environment without `bash -lc`.

---

## 10. Future Enhancements (Optional)

* Add interactive controls (pause/resume simulation).
* Support multiple pre-trained policies for comparison.
* Implement data logging (per-step vehicle positions, rewards, etc.).
* Integrate video recording of the GUI session.

---