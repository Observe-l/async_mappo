# Project Specification: HUD Dashboard with Model Comparison

**Target File:** `script/render/run_demo_dashboard_v2.py`
**Base Logic:** `script/render/run_demo_schedule_mqtt.py`
**Framework:** Python 3.x, PyQt5, Matplotlib, Traci, Pandas

-----

## 1\. Overview

The objective is to upgrade the existing SUMO demo into a **Head-Up Display (HUD)** style dashboard. The key feature is a **Live vs. Pretrained Comparison**, visualizing how the currently running model performs against historical best data.

**Key Constraints:**

1.  **Aesthetics:** Semi-transparent, frameless window overlaying the SUMO map.
2.  **Legacy Support:** Must retain **MQTT data publishing** and **Camera Tracking** (auto-follow trucks).
3.  **Data Source:** Pretrained data must be loaded from specific CSV files.

-----

## 2\. Data Sources & Processing

### 2.1 Pretrained Data (The Baseline)

You must load data from: `/home/lwh/Documents/Code/RL-Scheduling/result/rul_threshold_7/async_mappo/2025-05-03-00-20/exp_hpAM/1000`

We need to merge three CSV files to reconstruct the timeline:

| File Name | Key Columns to Use | Processing Logic |
| :--- | :--- | :--- |
| **`product.csv`** | `time`, `total` | `Pretrained_Prod = total` |
| **`distance.csv`** | `total_truck_0` ... `total_truck_11` | `Pretrained_Dist = Sum(total_truck_0 ... total_truck_11)` |
| **`result.csv`** | `cumulate reward_truck_0` ... | `Pretrained_Reward = Sum(cumulate reward_truck_0 ...)` |

### 2.2 Live Data (The Simulation)

Calculated real-time during `traci.simulationStep()`:

| Metric | Calculation Logic |
| :--- | :--- |
| **Live Production** | Count of items arriving at factory sink nodes (same as current demo). |
| **Live Distance** | `sum([traci.vehicle.getDistance(id) for id in truck_ids])` |
| **Live Reward** | Sum of rewards from the loaded Policy Network for this step. |

### 2.3 The Profit Formula

The profit metric applies to both Pretrained and Live data:

$$\text{Profit} = (\text{Total Final Product} \times 10) - (\text{Total Driving Distance} \times 0.00001)$$

-----

## 3\. Visual Design (The "Ghost" Comparison)

The dashboard will contain **3 Charts**. Each chart must plot **two lines**:

1.  **Line A (Pretrained Model):**

      * **Style:** Grey, Dashed Line (`--`), Alpha 0.6.
      * **Legend Label:** "Pretrained Model"
      * *Note:* This line is static (or revealed progressively).

2.  **Line B (Current Training):**

      * **Style:** Bright Neon Color (Green/Gold/Cyan), Solid Line (`-`), Thick.
      * **Legend Label:** "Current Model"
      * *Note:* This line updates dynamically every step.

-----The problem still there. Please run the demo program, 

## 4\. Implementation Logic & Code Structure

Below is the required Python class structure. Copy this logic into your file.

### 4.1 Imports & Setup

```python
import sys
import pandas as pd
import numpy as np
import traci
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- Path Configuration ---
BASE_PATH = "/home/lwh/Documents/Code/RL-Scheduling/result/rul_threshold_7/async_mappo/2025-05-03-00-20/exp_hpAM/1000"
```

### 4.2 Data Loader Class (New)

This handles the complexity of your specific CSV columns.

```python
class BaselineLoader:
    def __init__(self, base_path):
        # 1. Load Files
        df_prod = pd.read_csv(f"{base_path}/product.csv")
        df_dist = pd.read_csv(f"{base_path}/distance.csv")
        # df_res = pd.read_csv(f"{base_path}/result.csv") # Optional if you need reward

        # 2. Align Data (Ensure same length/index)
        # Assuming 'step_length' or index aligns them. 
        # Let's clean column names just in case (strip spaces)
        df_dist.columns = df_dist.columns.str.strip()
        df_prod.columns = df_prod.columns.str.strip()

        # 3. Calculate Aggregates
        # Sum all 'total_truck_X' columns for total distance
        truck_cols = [c for c in df_dist.columns if 'total_truck' in c]
        self.total_distance = df_dist[truck_cols].sum(axis=1)
        
        self.total_product = df_prod['total']
        self.steps = df_prod.index * 10 # Assuming 10 is your step interval, adjust if needed
        
        # 4. Calculate Profit (The Formula)
        # Profit = (Prod * 10) - (Dist * 0.00001)
        self.profit = (self.total_product * 10) - (self.total_distance * 0.00001)

    def get_data_up_to(self, step_idx):
        """Returns sliced data for plotting"""
        # Ensure we don't go out of bounds
        idx = min(step_idx, len(self.steps)-1)
        return self.steps[:idx], self.total_product[:idx], self.profit[:idx]
```

### 4.3 The Main Dashboard Class

This integrates the legacy features (`camera_track`, `mqtt`) with the new UI.

```python
class HUD_Dashboard(QMainWindow):
    def __init__(self, sumocfg):
        super().__init__()
        
        # --- UI Setup: Translucent & Frameless ---
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Create semi-transparent background
        self.central_widget = QWidget()
        self.central_widget.setStyleSheet("""
            QWidget { background-color: rgba(20, 20, 20, 200); border-radius: 15px; border: 1px solid #555; }
            QLabel { color: white; background: transparent; }
        """)
        self.setCentralWidget(self.central_widget)
        
        # --- Data Initialization ---
        self.baseline = BaselineLoader(BASE_PATH)
        self.live_steps = []
        self.live_prod = []
        self.live_profit = []
        
        # Live Metric Accumulators
        self.current_prod = 0
        self.current_dist_sum = 0.0

        # --- Legacy Features Setup ---
        self.mqtt_client = self.init_mqtt() # Use your existing code here
        self.truck_ids = [] # Populated after traci starts
        
        # --- SUMO Start ---
        traci.start(["sumo-gui", "-c", sumocfg, "--start"])
        
        self.init_ui_elements() # Create Matplotlib canvases here
        
        # --- Game Loop ---
        self.timer = QTimer()
        self.timer.setInterval(50) 
        self.timer.timeout.connect(self.update_loop)
        self.timer.start()

    def update_loop(self):
        if traci.simulation.getMinExpectedNumber() <= 0:
            self.timer.stop(); return

        # 1. SUMO Step
        traci.simulationStep()
        current_sim_time = traci.simulation.getTime()

        # 2. Legacy: Camera Track (Keep your existing logic)
        self.camera_track_logic() 

        # 3. Legacy: MQTT (Keep your existing logic)
        # self.mqtt_publish(...)

        # 4. Calculate Live Metrics
        if not self.truck_ids:
            self.truck_ids = traci.vehicle.getIDList()
        
        # A. Distance
        step_dist = 0
        for tid in self.truck_ids:
            step_dist += traci.vehicle.getDistance(tid) # Note: getDistance returns TOTAL distance of vehicle
        self.current_dist_sum = step_dist
        
        # B. Product (Use your existing logic to detect arrivals)
        # self.current_prod = ... 

        # C. Profit Formula
        current_profit = (self.current_prod * 10) - (self.current_dist_sum * 0.00001)
        
        # 5. Store Data
        self.live_steps.append(current_sim_time)
        self.live_prod.append(self.current_prod)
        self.live_profit.append(current_profit)

        # 6. Refresh UI (Visual Comparison)
        if len(self.live_steps) % 5 == 0: # Update every 5 steps to save CPU
            self.redraw_charts()

    def redraw_charts(self):
        # Example for Profit Chart
        self.ax_profit.clear()
        
        # Plot Baseline (Grey Dashed)
        b_steps, _, b_profit = self.baseline.get_data_up_to(len(self.live_steps))
        self.ax_profit.plot(b_steps, b_profit, 
                            color='gray', linestyle='--', alpha=0.6, label='Pretrained Model')
        
        # Plot Live (Gold Solid)
        self.ax_profit.plot(self.live_steps, self.live_profit, 
                            color='#ffd700', linewidth=2, label='Current Model')
        
        # Formatting
        self.ax_profit.legend(facecolor='#2b2b2b', labelcolor='white')
        self.ax_profit.grid(True, color='#444', linestyle=':')
        self.canvas_profit.draw()

    # ... Include your existing camera_track and mqtt functions here ...
```

-----

## 5\. Summary of Required Changes

1.  **Pandas Integration:** Added `pandas` to read the complex CSV headers in `distance.csv`.
2.  **Profit Logic:** Implemented `Prod*10 - Dist*0.00001` in both the `BaselineLoader` and the live `update_loop`.
3.  **Visualization:**
      * Set transparency using `rgba(20,20,20,200)`.
      * Added double-line plotting (Pretrained vs Current) with a clear Legend.
4.  **Legacy Code:** You must paste your specific `camera_track` and `mqtt_client` setup code into the placeholders provided in the class above.

**Next Step:** Copy the code structure into `script/render/run_demo_dashboard_v2.py`, paste your existing camera/MQTT logic into the placeholders, and run.
