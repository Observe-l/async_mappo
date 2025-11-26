Here is the comprehensive design document and specification for your new dashboard. You can save this file as `DASHBOARD_SPEC.md` or directly follow it to implement your code.

I have designed this based on the **PyQt5 + Matplotlib** architecture, which will provide the "better display effect" and video recording quality you requested.

-----

# Project Specification: RL Logistics Real-time Dashboard

**Target File:** `script/render/run_dashboard.py`
**Base Logic:** Based on `script/render/run_demo_schedule_mqtt.py`
**Framework:** Python 3.x, PyQt5, Matplotlib, Traci

-----

## 1\. Overview

The goal is to create a standalone, high-performance Graphical User Interface (GUI) to visualize the performance of the Async-MAPPO RL agent in a SUMO traffic simulation.

The dashboard will run alongside the SUMO-GUI window. It will control the simulation clock and render real-time metrics, providing a "Command Center" aesthetic suitable for video demonstrations.

-----

## 2\. Visual Design & Layout

### 2.1 Theme

  * **Style:** Cyberpunk / Dark Mode (High contrast for video visibility).
  * **Background:** Dark Grey (`#2b2b2b` or `#1e1e1e`).
  * **Text:** White / Light Grey.
  * **Accent Colors:**
      * **Production:** Neon Green (Growth/Success).
      * **Profit:** Gold/Yellow (Money).
      * **Reward:** Cyan/Blue (AI/Intelligence).

### 2.2 Layout Grid (Single Window)

The window should be divided vertically or logically into specific zones:

  * **Zone A: Key Performance Indicators (KPI) - Top Row**

      * Three large "Digital Cards" displaying real-time numbers.
      * Card 1: **Total Production** (Integer).
      * Card 2: **Total Profit** (Currency format `$`).
      * Card 3: **Current Reward** (Float).

  * **Zone B: Time-Series Charts - Middle/Bottom Area**

      * **Chart 1 (Production Rate):** Line chart showing cumulative production over simulation steps.
      * **Chart 2 (Financial Health):** Line chart showing Profit vs. Cost over time.
      * **Chart 3 (Agent Intelligence):** Line chart showing the Instant Reward or Cumulative Reward per step.

  * **Zone C: Control & Status (Bottom Bar)**

      * Simulation Progress Bar (Step 0 to 3600).
      * Status Label: "Running", "Paused", "Completed".

-----

## 3\. Data Logic & Metrics

You need to extract or calculate the following variables inside your `traci` loop:

### 3.1 Total Production ($P_{total}$)

  * **Definition:** The count of finished products delivered to the final destination.
  * **Source:** Monitor the "arrived" cargo at the sink nodes/factories.
  * **Data Type:** `int` (Cumulative).

### 3.2 Total Profit ($Prof_{total}$)

Since "Profit" is not a native SUMO metric, you must define a formula. A common logistics formula is:
$$\text{Profit} = (\text{Revenue per Unit} \times P_{total}) - (\text{Fuel Cost} \times \text{Total Distance}) - (\text{Time Penalty})$$

  * **Recommendation for Demo:**
      * Income: +100 for every finished product.
      * Cost: -0.1 for every meter traveled by trucks.
      * **Source:** `traci.vehicle.getDistance(truck_id)` and your production counter.

### 3.3 Training Reward ($R_{t}$)

  * **Definition:** The value the RL agent tries to maximize.
  * **Source:** Your trained Async-MAPPO policy output.
      * *Option A (Real-time):* The sum of rewards received by all agents in the current step.
      * *Option B (Static):* If you want to show "Training Result", you can load a static CSV of your training loss curve and display it as a static image to prove convergence, while showing the *current* episode reward dynamically.
      * **Selected Approach:** Display **Cumulative Episode Reward** (Sum of rewards from Step 0 to Current Step).

-----

## 4\. Implementation Skeleton (Code Structure)

Create `script/render/run_dashboard.py` using this structure.

### 4.1 Imports

```python
import sys
import traci
import numpy as np
from collections import deque
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QLabel, QFrame)
from PyQt5.QtCore import QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Import your RL environment/policy loader here
# from envs.mappo_env import LogisticsEnv 
```

### 4.2 The Dashboard Class

```python
class CyberDashboard(QMainWindow):
    def __init__(self, sumocfg_path):
        super().__init__()
        self.init_ui()
        self.init_sumo(sumocfg_path)
        self.init_data_containers()
        
        # Main Loop Timer (drives the simulation)
        self.timer = QTimer()
        self.timer.setInterval(50) # 50ms = 20 FPS
        self.timer.timeout.connect(self.game_loop)
        self.timer.start()

    def init_ui(self):
        """Setup Layout, Style, and Matplotlib Figures"""
        self.setWindowTitle("Async-MAPPO Logistics Monitor")
        self.setStyleSheet("background-color: #1e1e1e; color: white;")
        
        # ... Setup 3 KPI Labels ...
        # ... Setup Matplotlib Subplots (dark background) ...
        
    def init_data_containers(self):
        """Use deques for efficient sliding window data"""
        self.history_len = 200
        self.steps = deque(maxlen=self.history_len)
        self.production_data = deque(maxlen=self.history_len)
        self.profit_data = deque(maxlen=self.history_len)
        self.reward_data = deque(maxlen=self.history_len)
        
        # Metric Accumulators
        self.total_prod = 0
        self.total_profit = 0.0
        self.cum_reward = 0.0

    def init_sumo(self, config_file):
        """Start Traci without blocking"""
        # Ensure you use sumo-gui
        traci.start(["sumo-gui", "-c", config_file, "--start"])
        # Optional: Position SUMO window to the left of this dashboard
        
    def game_loop(self):
        """
        The Core Function:
        1. Step SUMO
        2. Get RL Data
        3. Update UI
        """
        if traci.simulation.getMinExpectedNumber() <= 0:
            self.timer.stop()
            return

        # 1. Traci Step
        traci.simulationStep()
        
        # 2. Extract Metrics (You need to implement logic here)
        # current_step_reward = ...
        # current_prod_increase = ...
        # fuel_cost = ...
        
        # 3. Update Data Structures
        self.update_metrics(...)
        
        # 4. Refresh Plots (Efficiently)
        self.refresh_charts()
```

-----

## 5\. Key Implementation Details to Watch

1.  **Matplotlib Integration:**

      * Do **not** use `plt.plot()` or `plt.show()`. You must use the Object-Oriented API (`ax.plot()`) embedded in `FigureCanvasQTAgg`.
      * Set `ax.set_facecolor('#1e1e1e')` and `fig.patch.set_facecolor('#1e1e1e')` to match the dashboard theme.

2.  **Performance Optimization:**

      * Do not redraw the entire chart legend or title every frame.
      * Use `line_reference.set_data(x, y)` and then `canvas.draw()` for smooth updates.
      * If `traci` becomes too fast, increase the `timer.setInterval`.

3.  **Synchronization:**

      * Since `traci` runs in the same thread as the GUI in this design (via `QTimer`), the UI will remain responsive. This is simpler than threading for a demo.

-----

## 6\. Next Steps for You

1.  **Copy** the logic from your `run_demo_schedule_mqtt.py` regarding how you load the MAPPO policy and how you assign actions to trucks.
2.  **Paste** that logic into the `game_loop` method of the new Dashboard class.
3.  **Implement** the calculation for "Profit" based on the formula above.
4.  **Run** `python script/render/run_dashboard.py` and record your screen\!