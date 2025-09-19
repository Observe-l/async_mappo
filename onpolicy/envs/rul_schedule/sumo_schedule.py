import numpy as np
from gymnasium.spaces import Discrete, Box, Dict
from collections import defaultdict
from pathlib import Path
from csv import writer
import datetime
import random
import string
import threading
try:
    import tkinter as tk
    from tkinter import ttk
except Exception:
    tk = None
    ttk = None

# SUMO
import libsumo as traci

from onpolicy.envs.rul_schedule.factory import Factory, Producer
from onpolicy.envs.rul_schedule.rul_gen import predictor

class TelemetryGUI:
    """Simple Tkinter dashboard to display truck telemetry while SUMO runs."""
    def __init__(self, num_agents, get_data_cb, title='SUMO Demo Telemetry'):
        if tk is None or ttk is None:
            raise RuntimeError("Tkinter is not available in this environment.")
        self.num_agents = num_agents
        self.get_data_cb = get_data_cb
        self.root = tk.Tk()
        self.root.title(title)
        cols = ('Truck', 'RUL', 'Distance (km)', 'Destination', 'Maintaining')
        self.tree = ttk.Treeview(self.root, columns=cols, show='headings', height=min(20, num_agents))
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=140, anchor='center')
        self.tree.pack(fill=tk.BOTH, expand=True)
        for i in range(self.num_agents):
            self.tree.insert('', tk.END, values=(f'truck_{i}', '-', '-', '-', '-'))
        self._schedule_refresh()

    def _schedule_refresh(self):
        self._refresh()
        self.root.after(500, self._schedule_refresh)

    def _refresh(self):
        try:
            data = self.get_data_cb()
        except Exception:
            return
        items = self.tree.get_children()
        if len(items) < self.num_agents:
            for _ in range(self.num_agents - len(items)):
                self.tree.insert('', tk.END, values=("-","-","-","-","-"))
            items = self.tree.get_children()
        for i, row in enumerate(data):
            dist = row.get('distance_km')
            dist_str = f"{dist:.3f}" if isinstance(dist, (int, float)) else (dist if dist is not None else '-')
            vals = (
                f"truck_{i}",
                f"{row.get('rul','-')}",
                dist_str,
                row.get('destination','-'),
                'Yes' if row.get('maintaining', False) else 'No'
            )
            self.tree.item(items[i], values=vals)

    def run(self):
        self.root.mainloop()

class async_scheduling_sumo(object):
    """
    SUMO-backed evaluation environment.
    Differences from training env (schedule.py):
    - Uses SUMO vehicles instead of Python Truck class.
    - A truck becomes operable when it arrives at its destination ParkingArea (arrival > 0).
    - Observations remain from offline dataset (same structure as schedule.py).
    """
    def __init__(self, args):
        self.truck_num = args.num_agents
        self.factory_num = 50
        self.use_rul_agent = args.use_rul_agent
        self.rul_threshold = args.rul_threshold
        self.rul_state = getattr(args, 'rul_state', False)
        self.sumo_cfg = args.sumo_cfg
        # Maintenance timing (reuse values close to truck.py defaults)
        self.maintain_time = getattr(args, 'maintain_time', 6*3600)
        # GUI backend selection
        self.use_sumo_gui = getattr(args, 'use_sumo_gui', False)
        # If GUI requested, rebind traci to standard traci and use sumo-gui executable
        if self.use_sumo_gui:
            try:
                import traci as traci_gui
                globals()['traci'] = traci_gui
            except Exception as e:
                print("Failed to import traci for GUI, falling back to libsumo:", e)
                self.use_sumo_gui = False
        self._telemetry_started = False

        self._start_sumo(self.sumo_cfg)
        self._spawn_trucks()

        self.predictor = predictor()
        self.init_offline_factories()

        # Spaces: mimic schedule.py
        obs_space = {}
        share_obs_space = {}
        self.observation_space = []
        self.action_space = {}
        obs = self._get_obs()
        share_obs_dim = 0
        for agent_id, tmp_obs in obs.items():
            obs_dim = len(tmp_obs)
            share_obs_dim += obs_dim
            obs_space["global_obs"] = Box(low=-1, high=30000, shape=(obs_dim,))
            self.observation_space.append(Dict(obs_space))
            if self.use_rul_agent:
                self.action_space[agent_id] = Discrete(self.factory_num)
            else:
                self.action_space[agent_id] = Discrete(self.factory_num+1)
        share_obs_space["global_obs"] = Box(low=-1, high=30000, shape=(share_obs_dim,))
        self.share_observation_space = [Dict(share_obs_space) for _ in range(self.truck_num)]

        self.episode_num = 0
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
        current_date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        self.path = f"/home/lwh/Documents/Code/RL-Scheduling/result/sumo_demo/{current_date}/exp_{random_string}"
        Path(self.path).mkdir(parents=True, exist_ok=True)

    # ------------- SUMO integration -------------
    def _start_sumo(self, cfg_path: str):
        try:
            traci.close()
        except Exception:
            pass
        exe = "sumo-gui" if self.use_sumo_gui else "sumo"
        traci.start([exe, "-c", cfg_path, "--no-warnings", "True"])  # GUI shows map when enabled

    def _spawn_trucks(self):
        self.veh_ids = []
        self.destinations = {}
        # Per-truck maintenance state
        self.maintaining = {i: False for i in range(self.truck_num)}
        self.maintain_timer = {i: 0 for i in range(self.truck_num)}
        self.pending_maintain = {i: False for i in range(self.truck_num)}
        for i in range(self.truck_num):
            vid = f"truck_{i}"
            self.veh_ids.append(vid)
            init_idx = random.randint(0, self.factory_num - 1)
            init_pa = f"Factory{init_idx}"
            route_id = f"{init_pa}_to_{init_pa}"
            try:
                traci.vehicle.add(vehID=vid, routeID=route_id, typeID="truck")
            except Exception:
                try:
                    traci.vehicle.remove(vehID=vid)
                except Exception:
                    pass
                traci.vehicle.add(vehID=vid, routeID=route_id, typeID="truck")
            traci.vehicle.setParkingAreaStop(vehID=vid, stopID=init_pa)
            self.destinations[i] = init_pa
        for _ in range(10):
            traci.simulationStep()
        # Initialize GUI panel for telemetry if GUI is enabled (start once)
        if not hasattr(self, 'rul_values'):
            self.rul_values = {i: 125 for i in range(self.truck_num)}
        if (not self._telemetry_started) and self.use_sumo_gui and tk is not None:
            try:
                self._telemetry = TelemetryGUI(self.truck_num, self._collect_gui_data)
                threading.Thread(target=self._telemetry.run, daemon=True).start()
                self._telemetry_started = True
            except Exception as e:
                print("Failed to start Telemetry GUI:", e)

    def _set_destination(self, agent_id: int, action: int):
        # If in maintenance, ignore new actions
        if self.maintaining.get(agent_id, False):
            return
        # Maintain action: route to Factory0
        if (not self.use_rul_agent) and action >= self.factory_num:
            action = 0
            self.pending_maintain[agent_id] = True
        else:
            self.pending_maintain[agent_id] = False
        vid = f"truck_{agent_id}"
        dest_pa = f"Factory{int(action)}"
        self.destinations[agent_id] = dest_pa
        try:
            edge_id = traci.parkingarea.getRoadID(dest_pa)
            if edge_id:
                try:
                    traci.vehicle.changeTarget(vehID=vid, edgeID=edge_id)
                except Exception:
                    pass
            # If currently parked, resume
            try:
                stops = traci.vehicle.getStops(vehID=vid)
                if stops and stops[0].arrival > 0:
                    traci.vehicle.resume(vehID=vid)
            except Exception:
                pass
            traci.vehicle.setParkingAreaStop(vehID=vid, stopID=dest_pa)
        except Exception:
            pass

    def _arrived_operable(self, agent_id: int) -> bool:
        vid = f"truck_{agent_id}"
        target = self.destinations.get(agent_id)
        if not target:
            return False
        try:
            stops = traci.vehicle.getStops(vehID=vid)
            if not stops:
                return False
            latest = stops[-1]
            return (latest.arrival > 0) and (latest.stoppingPlaceID == target)
        except Exception:
            return False

    def _update_maintenance(self, ticks: int):
        """Advance maintenance timers and recover trucks when done.

        Behavior:
        - If a truck arrives at Factory0 due to a maintain action, it becomes non-operable
          and starts a maintenance timer. While maintaining, we ignore actions and do not
          expose observations for that truck. Once timer >= maintain_time, the truck
          recovers and becomes operable again (parked at Factory0).
        """
        for i in range(self.truck_num):
            if self.maintaining[i]:
                self.maintain_timer[i] += ticks
                if self.maintain_timer[i] >= self.maintain_time:
                    # Recover from maintenance
                    self.maintaining[i] = False
                    self.maintain_timer[i] = 0
            else:
                # If just arrived at Factory0 and action was maintenance, start timer
                if self._arrived_operable(i) and self.destinations.get(i) == 'Factory0' and self.pending_maintain.get(i, False):
                    self.maintaining[i] = True
                    self.maintain_timer[i] = 0
                    # Clear pending flag once maintenance starts
                    self.pending_maintain[i] = False

    # ------------- Offline logic (factories, producer, obs, reward) -------------
    def init_offline_factories(self):
        # Build same factory graph as schedule.py using offline logic
        self.factory = {f'Factory{i}': Factory(factory_id=f'Factory{i}', product=f'P{i}') for i in range(self.factory_num)}
        final_product = ['A', 'B', 'C', 'D', 'E']
        remaining_materials = [f'P{i}' for i in range(45)]
        self.transport_idx = {}
        for i, product in enumerate(final_product):
            tmp_factory_id = f'Factory{45 + i}'
            tmp_materials = [remaining_materials.pop() for _ in range(9)]
            tmp_factory = Factory(factory_id=tmp_factory_id, product=product, material=tmp_materials)
            self.factory[tmp_factory_id] = tmp_factory
            for transport_material in tmp_materials:
                self.transport_idx[transport_material] = tmp_factory_id
        # We do not use Truck class in Producer, so only use produce() calls
        self.producer = Producer(self.factory, truck=[], transport_idx=self.transport_idx)
        for _ in range(100):
            for f in self.factory.values():
                f.produce()

    def reset(self, seed=None, options=None):
        self.make_folder()
        # Reinitialize SUMO world to clear parked states
        self._start_sumo(self.sumo_cfg)
        self._spawn_trucks()
        # Reset maintenance flags
        self.maintaining = {i: False for i in range(self.truck_num)}
        self.maintain_timer = {i: 0 for i in range(self.truck_num)}
        # Reset offline factories
        self.init_offline_factories()
        self.episode_num += 1
        self.episode_len = 0
        self.done = np.array([False for _ in range(self.truck_num)])
        obs = self._get_obs()
        return obs

    def step(self, action_dict: dict):
        # Apply actions
        for agent_id, action in action_dict.items():
            self._set_destination(int(agent_id), int(action))
        # Advance SUMO until any non-maintaining truck has arrived
        operable = []
        guard = 0
        while len(operable) == 0:
            # offline produce once per SUMO step to emulate time passing
            for f in self.factory.values():
                f.produce()
            traci.simulationStep()
            # Update maintenance timers and states
            self._update_maintenance(1)
            operable = [i for i in range(self.truck_num) if self._arrived_operable(i) and not self.maintaining.get(i, False)]
            guard += 1
            if guard > 10000:
                break
        self.episode_len += guard

        obs = self._get_obs()
        rewards = self._get_reward()
        self.flag_reset()
        self.save_results(self.episode_len, guard, action_dict, rewards)
        # Episode termination condition (time-based), similar to schedule.py
        if self.episode_len >= 30 * 24 * 3600:
            self.done = np.array([True for _ in range(self.truck_num)])
        return obs, rewards, self.done, {}

    def _get_obs(self):
        observation = {}
        # Shared observation from offline factories
        fac_truck_num = defaultdict(int)
        for i in range(self.truck_num):
            # Count trucks by intended destination (as proxy)
            dest = self.destinations.get(i)
            if dest is not None:
                fac_truck_num[dest] += 1
        warehouse_storage = []
        com_truck_num = []
        for tmp_factory in self.factory.values():
            material_storage = list(tmp_factory.material_num.values())
            product_storage = tmp_factory.product_num
            warehouse_storage += material_storage + [product_storage]
            com_truck_num.append(fac_truck_num[tmp_factory.id])
        queue_obs = np.concatenate([warehouse_storage] + [com_truck_num])

        # Observation for operable trucks only: trucks that arrived and not under maintenance
        for i in range(self.truck_num):
            if self._arrived_operable(i) and not self.maintaining.get(i, False):
                # Distances from SUMO aren't used; keep same shape as schedule.py by faking with zeros
                # Maintain compatibility: [queue]+[distance to each factory?]+[position]+[weight]+[product]
                # We don’t track actual load here; use weight=0 and product=-1 as in "empty" state.
                dest = self.destinations.get(i, 'Factory0')
                position_idx = int(dest[7:]) if dest.startswith('Factory') else 0
                distance_stub = np.zeros(self.factory_num)  # placeholder
                weight = 0
                product = -1
                if self.rul_state:
                    # No engine series from SUMO, approximate with large RUL
                    observation[i] = np.concatenate([[125]] + [queue_obs] + [distance_stub] + [[position_idx]] + [[weight]] + [[product]])
                else:
                    observation[i] = np.concatenate([queue_obs] + [distance_stub] + [[position_idx]] + [[weight]] + [[product]])
        return observation

    def _get_reward(self):
        # Reuse schedule.py style: only short-term reward using final product increments
        rew = np.zeros((self.truck_num, 1))
        tmp_final_product = 0
        for k in ["Factory45", "Factory46", "Factory47", "Factory48", "Factory49"]:
            tmp_final_product += self.factory[k].total_final_product
        rew_final_product = 10 * tmp_final_product
        # Assign to all arrived agents equally this step (simple choice)
        arrived = [i for i in range(self.truck_num) if self._arrived_operable(i) and not self.maintaining.get(i, False)]
        for i in arrived:
            rew[i, 0] = rew_final_product / max(1, len(arrived))
        return rew

    def flag_reset(self):
        pass

    # ------------- Logging (reused) -------------
    def make_folder(self):
        folder_path = self.path + '/{}/'.format(self.episode_num)
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        self.action_file = folder_path + 'action.csv'
        self.product_file = folder_path + 'product.csv'
        self.agent_file = folder_path + 'result.csv'
        with open(self.product_file,'w') as f:
            f_csv = writer(f)
            result_list = ['time', 'step_length', 'total', 'A', 'B', 'C', 'D', 'E']
            f_csv.writerow(result_list)
        with open(self.agent_file, 'w') as f:
            f_csv = writer(f)
            result_list = ['time', 'step_length']
            for i in range(self.truck_num):
                agent_list = [f'action_truck_{i}', f'reward_truck_{i}']
                result_list += agent_list
            f_csv.writerow(result_list)

    def save_results(self, time, lenth, action_dict, rewards):
        current_time = round(time/3600,3)
        with open(self.product_file, 'a') as f:
            f_csv = writer(f)
            tmp_A = round(self.factory['Factory45'].total_final_product,3)
            tmp_B = round(self.factory['Factory46'].total_final_product,3)
            tmp_C = round(self.factory['Factory47'].total_final_product,3)
            tmp_D = round(self.factory['Factory48'].total_final_product,3)
            tmp_E = round(self.factory['Factory49'].total_final_product,3)
            total = tmp_A+tmp_B+tmp_C+tmp_D+tmp_E
            product_list = [current_time,lenth,total,tmp_A,tmp_B,tmp_C,tmp_D,tmp_E]
            f_csv.writerow(product_list)
        with open(self.agent_file, 'a') as f:
            f_csv = writer(f)
            agent_list = [current_time, lenth]
            for i in range(self.truck_num):
                if i in action_dict.keys():
                    agent_list += [action_dict[i], rewards[i,0]]
                else:
                    agent_list += ['NA', rewards[i,0]]
            f_csv.writerow(agent_list)

    def close(self):
        try:
            traci.close()
        except Exception:
            pass

    # --------- GUI data collection ---------
    def _collect_gui_data(self):
        rows = []
        for i in range(self.truck_num):
            try:
                dist_m = traci.vehicle.getDistance(f"truck_{i}")
                dist_km = (dist_m or 0.0) / 1000.0
            except Exception:
                dist_km = 0.0
            dest = self.destinations.get(i, 'Factory0')
            if self.maintaining.get(i, False) or self.pending_maintain.get(i, False):
                dest_label = 'Workshop'
            else:
                dest_label = dest
            rows.append({
                'rul': self.rul_values.get(i, 125),
                'distance_km': dist_km,
                'destination': dest_label,
                'maintaining': self.maintaining.get(i, False)
            })
        return rows
