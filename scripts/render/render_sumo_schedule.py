#!/usr/bin/env python3
"""
Demo: Evaluate a trained async-MAPPO policy in a live SUMO simulation.

This script reuses the offline-trained policy (trained on `onpolicy/envs/rul_schedule/schedule.py`)
for decision making, and mirrors those decisions to a SUMO world so you can visually
see trucks moving between factories. It does not attempt to perfectly synchronize
SUMO-side inventories; it's a visual/control demo driven by the trained policy.

Key behavior: Instead of using a fixed decision interval, the demo advances the
SUMO simulation until at least one truck has arrived at its destination parking area
(operable), similar to `refresh_state()` logic in `core.py`.

Requirements:
- SUMO/libsumo available in your Python environment
- A SUMO network with ParkingAreas named Factory0..Factory49 and routeIDs like
    "FactoryX_to_FactoryX" (see onpolicy/envs/rul_schedule/core.py and ray_env.py examples)
- A trained actor checkpoint directory containing `actor.pt`

Usage example (adjust paths):
python scripts/render/render_sumo_schedule.py \
        --actor_dir /path/to/checkpoint_dir \
        --sumo_cfg /path/to/your/osm.sumocfg \
        --num_agents 12 \
        --max_steps 2000
"""
from __future__ import annotations
import os
import sys
import time
import argparse
import random
from pathlib import Path
import threading
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import torch

# SUMO
# Use libsumo for consistency with the repo; start with 'sumo-gui' to show the map
import libsumo as traci

# Repo imports
"""
Disable Weights & Biases before any onpolicy imports that might import wandb.
This prevents unwanted logging during the demo.
"""
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")

# Ensure repo root is on sys.path for 'onpolicy' imports when run directly
REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import ScheduleEnv
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy


def make_envs(all_args):
    """Create the offline scheduling envs used for inference (policy expects this)."""
    def get_env_fn(rank):
        def init_env():
            if all_args.scenario_name == "rul_schedule":
                from onpolicy.envs.rul_schedule.schedule import async_scheduling
            elif all_args.scenario_name == "map_schedule":
                from onpolicy.envs.schedule.map_schedule import async_scheduling
            else:
                from onpolicy.envs.schedule.schedule import async_scheduling
            env = async_scheduling(all_args)
            return env
        return init_env

    return ScheduleEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


class SumoDemo:
    """Minimal SUMO controller that mirrors policy decisions as vehicle destinations."""
    def __init__(self, num_agents: int, sumo_cfg: str, factory_count: int = 50, headless: bool = False, debug: bool = False):
        self.num_agents = num_agents
        self.factory_count = factory_count
        self.debug = bool(debug)
        # Track current intended destination per agent_id (parking area name)
        self.destinations = {i: None for i in range(self.num_agents)}
        # Track last known parked parking area per agent for robust routing
        self.current_pa: Dict[int, Optional[str]] = {i: None for i in range(self.num_agents)}
        # Track whether SUMO has ended/connection closed
        self._ended = False
        # Cache last-known non-negative distances for stability in GUI
        self._last_distance_m = {}
        # Track last XY position and accumulated distance per vehicle for robust distance
        self._last_xy = {}
        self._cum_dist_m = {}
        # Track last SUMO-reported absolute distance for delta computation
        self._last_sumo_dist = {}
        # Track last time (sim sec) when vehicle advanced distance
        self._last_move_time = {}
        # GUI view id for highlighting/tracking (if available)
        self._view_id = None
        # Cache available parking areas to avoid invalid IDs
        self._pa_ids = []
        self._start_sumo(sumo_cfg, headless=headless)
        try:
            self._pa_ids = list(traci.parkingarea.getIDList())
        except Exception:
            self._pa_ids = []
        if self.debug:
            sample_pa = self._pa_ids[:10]
            print(f"[INIT] ParkingAreas found: count={len(self._pa_ids)} sample={sample_pa}")
        self._spawn_trucks()
        # After spawning, try to highlight and focus view if GUI
        self._init_gui_view()

    def _dump_vehicle_status(self, agent_id: int):
        """Print detailed status for a vehicle (debug helper)."""
        if not self.debug:
            return
        vid = f"truck_{agent_id}"
        try:
            t = traci.simulation.getTime()
        except Exception:
            t = '-'
        try:
            try:
                _ids = traci.vehicle.getIDList()
            except Exception:
                _ids = []
            if vid not in _ids:
                print(f"[VSTAT] t={t} agent={agent_id} veh missing")
                return
            spd = None
            road = None
            lane_pos = None
            rlen = None
            ridx = None
            stopped_pk = None
            last_stop = None
            driving_dist = None
            try:
                spd = float(traci.vehicle.getSpeed(vid))
            except Exception:
                pass
            try:
                road = traci.vehicle.getRoadID(vid)
            except Exception:
                pass
            try:
                lane_pos = float(traci.vehicle.getLanePosition(vid))
            except Exception:
                pass
            try:
                rr = traci.vehicle.getRoute(vid)
                rlen = len(rr) if rr is not None else None
            except Exception:
                rlen = None
            try:
                ridx = traci.vehicle.getRouteIndex(vid)
            except Exception:
                ridx = None
            try:
                stopped_pk = bool(traci.vehicle.isStoppedParking(vid))
            except Exception:
                stopped_pk = None
            try:
                stops = traci.vehicle.getStops(vid)
                last_stop = (stops[-1].stoppingPlaceID, stops[-1].arrival) if stops else None
            except Exception:
                last_stop = None
            # Estimate driving distance to destination edge if possible
            try:
                dest_pa = self.destinations.get(agent_id)
                if dest_pa:
                    dest_lane = traci.parkingarea.getLaneID(dest_pa)
                    dest_edge = traci.lane.getEdgeID(dest_lane)
                    driving_dist = traci.vehicle.getDrivingDistance(vid, dest_edge, 0)
            except Exception:
                driving_dist = None
            print(f"[VSTAT] t={t} agent={agent_id} spd={spd} road={road} lanePos={lane_pos} routeIdx={ridx} routeLen={rlen} parked={stopped_pk} lastStop={last_stop} driveDist={driving_dist}")
        except Exception as e:
            print(f"[VSTAT] t={t} agent={agent_id} status error: {e}")

    def _resolve_pa(self, index: int) -> Optional[str]:
        """Map Factory{index} to an existing ParkingArea ID; choose nearest available if missing."""
        target = f"Factory{int(index)}"
        if target in self._pa_ids:
            return target
        # Try nearest by numeric id
        try:
            nums = []
            for pid in self._pa_ids:
                if str(pid).startswith('Factory'):
                    try:
                        nums.append((abs(int(pid.replace('Factory','')) - int(index)), pid))
                    except Exception:
                        continue
            if nums:
                nums.sort(key=lambda x: x[0])
                return nums[0][1]
        except Exception:
            pass
        # Fallback to any available parking area
        return self._pa_ids[0] if self._pa_ids else None

    def _start_sumo(self, sumo_cfg: str, headless: bool = False):
        try:
            traci.close()
        except Exception:
            pass
        # Keep options in line with example files
        exe = "sumo" if headless else "sumo-gui"
        traci.start([
            exe, "-c", sumo_cfg,
            "--no-warnings", "True"
        ])

    def _spawn_trucks(self):
        self.veh_ids = []
        for i in range(self.num_agents):
            vid = f"truck_{i}"
            self.veh_ids.append(vid)
            init_idx = random.randint(0, self.factory_count - 1)
            init_pa = self._resolve_pa(init_idx) or f"Factory{init_idx}"
            # Prefer the predefined FactoryX_to_FactoryX routes like in core.py
            route_id = f"{init_pa}_to_{init_pa}"
            try:
                traci.vehicle.add(vehID=vid, routeID=route_id, typeID='truck')
            except Exception:
                try:
                    traci.vehicle.remove(vehID=vid)
                except Exception:
                    pass
                traci.vehicle.add(vehID=vid, routeID=route_id, typeID='truck')
            # Park at initial parking area
            traci.vehicle.setParkingAreaStop(vehID=vid, stopID=init_pa)
            # Configure speed mode and max speed to ensure vehicle can depart
            try:
                # Permissive mode helps vehicles leave parking promptly
                traci.vehicle.setSpeedMode(vid, 0)
            except Exception:
                pass
            try:
                traci.vehicle.setMaxSpeed(vid, 25.0)
            except Exception:
                pass
            # Enable rerouting device for dynamic changeTarget routing
            try:
                traci.vehicle.setParameter(vid, "device.rerouting.probability", "1")
            except Exception:
                pass
            # Do not mark a destination yet to avoid immediate 'operable' status
            self.destinations[i] = None
            # Remember initial parked area for routing
            self.current_pa[i] = init_pa
            # Ensure distance cache starts at 0
            self._last_distance_m[vid] = 0.0
            self._cum_dist_m[vid] = 0.0
            self._last_xy[vid] = None
            self._last_sumo_dist[vid] = None

        # Warm up a few steps to settle
        for _ in range(10):
            try:
                traci.simulationStep()
            except Exception:
                # If warmup fails, mark ended and stop
                self._ended = True
                break

    def _init_gui_view(self):
        """If running with GUI, pick a view and highlight our trucks distinctly."""
        try:
            views = list(traci.gui.getIDList())
            if views:
                self._view_id = views[0]
                # Zoom to a reasonable level and track first truck
                try:
                    traci.gui.setZoom(self._view_id, 1200)
                except Exception:
                    pass
                try:
                    traci.gui.trackVehicle(self._view_id, self.veh_ids[0])
                except Exception:
                    pass
            # Brightly color our demo trucks
            for vid in self.veh_ids:
                try:
                    traci.vehicle.setColor(vid, (255, 0, 255, 255))  # magenta with full alpha
                except Exception:
                    pass
        except Exception:
            pass

    def _ensure_vehicle(self, agent_id: int):
        """Ensure vehicle exists in SUMO; if missing, recreate parked at its last destination or Factory0."""
        vid = f"truck_{agent_id}"
        try:
            try:
                _ids = traci.vehicle.getIDList()
            except Exception:
                _ids = []
            if vid in _ids:
                return True
        except Exception:
            return False
        # Recreate vehicle
        try:
            target_pa = self.destinations.get(agent_id) or "Factory0"
            route_id = f"{target_pa}_to_{target_pa}"
            traci.vehicle.add(vehID=vid, routeID=route_id, typeID='truck')
            traci.vehicle.setParkingAreaStop(vehID=vid, stopID=target_pa)
            # Update current parked area since we recreated at target_pa
            try:
                self.current_pa[agent_id] = target_pa
            except Exception:
                pass
            # Reset distance deltas to avoid counting a teleport
            try:
                self._last_xy[vid] = None
                self._last_sumo_dist[vid] = None
            except Exception:
                pass
            # Reapply highlight color
            try:
                traci.vehicle.setColor(vid, (255, 0, 255, 255))
            except Exception:
                pass
            # Reapply permissive speed/rerouting
            try:
                traci.vehicle.setSpeedMode(vid, 0)
                traci.vehicle.setMaxSpeed(vid, 25.0)
                traci.vehicle.setParameter(vid, "device.rerouting.probability", "1")
            except Exception:
                pass
            return True
        except Exception:
            return False

    def set_destination(self, agent_id: int, dest_index: int):
        """Assign a new destination parking area to a truck and resume it.

        If dest_index >= factory_count, treat as maintenance and route to Factory0."""
        vid = f"truck_{agent_id}"
        if self._ended:
            return
        if not self._ensure_vehicle(agent_id):
            return
        mapped_maint = False
        if dest_index >= self.factory_count:
            dest_index = 0
            mapped_maint = True
        dest_pa = self._resolve_pa(dest_index) or f"Factory{dest_index}"
        self.destinations[agent_id] = dest_pa
        if self.debug:
            try:
                sim_t = traci.simulation.getTime()
            except Exception:
                sim_t = "-"
            print(f"[ROUTE] t={sim_t} agent={agent_id} veh={vid} set_destination -> idx={dest_index} pa={dest_pa} maint_map={mapped_maint}")
        # Determine destination edge
        try:
            dest_lane_id = traci.parkingarea.getLaneID(dest_pa)
            dest_edge_id = traci.lane.getEdgeID(dest_lane_id)
        except Exception:
            dest_edge_id = None
        if dest_edge_id:
            # Clear pending stops first
            try:
                traci.vehicle.clearPendingStops(vid)
            except Exception:
                pass
            # Ensure unparked before routing
            try:
                if traci.vehicle.isStoppedParking(vid):
                    traci.vehicle.resume(vid)
            except Exception:
                pass
            # Ask SUMO to route to destination edge (core.py behavior)
            try:
                traci.vehicle.changeTarget(vehID=vid, edgeID=dest_edge_id)
                if self.debug:
                    print(f"[ROUTE] agent={agent_id} changeTarget -> {dest_edge_id}")
            except Exception:
                pass
            # Optionally set explicit route for robustness
            src_edge_id = None
            try:
                last_pa = self.current_pa.get(agent_id)
                if last_pa:
                    last_lane = traci.parkingarea.getLaneID(last_pa)
                    src_edge_id = traci.lane.getEdgeID(last_lane)
            except Exception:
                src_edge_id = None
            if not src_edge_id:
                try:
                    src_edge_id = traci.vehicle.getRoadID(vid)
                    if src_edge_id.startswith(":"):
                        lane_id_curr = traci.vehicle.getLaneID(vid)
                        src_edge_id = traci.lane.getEdgeID(lane_id_curr)
                except Exception:
                    src_edge_id = None
            if self.debug:
                print(f"[ROUTE] agent={agent_id} src_edge={src_edge_id} -> dest_edge={dest_edge_id}")
            if src_edge_id:
                try:
                    route_obj = traci.simulation.findRoute(src_edge_id, dest_edge_id)
                    route_edges = route_obj.edges if hasattr(route_obj, 'edges') else []
                    if route_edges:
                        traci.vehicle.setRoute(vehID=vid, edgeList=route_edges)
                        if self.debug:
                            print(f"[ROUTE] agent={agent_id} route_len={len(route_edges)}")
                except Exception:
                    pass
        # Nudge to depart and schedule parking stop (resume before stop mirrors core.py)
        try:
            traci.vehicle.resume(vid)
        except Exception:
            pass
        # Try to schedule parking stop; if it fails, rebuild vehicle on dest_to_dest like core.py
        try:
            traci.vehicle.setParkingAreaStop(vehID=vid, stopID=dest_pa)
        except Exception:
            try:
                # Remove and recreate on a safe route, then set stop
                traci.vehicle.remove(vehID=vid)
            except Exception:
                pass
            try:
                safe_route = f"{dest_pa}_to_{dest_pa}"
                traci.vehicle.add(vehID=vid, routeID=safe_route, typeID='truck')
                traci.vehicle.setParkingAreaStop(vehID=vid, stopID=dest_pa)
                # Reapply visuals and params
                try:
                    traci.vehicle.setColor(vid, (255, 0, 255, 255))
                    traci.vehicle.setSpeedMode(vid, 0)
                    traci.vehicle.setMaxSpeed(vid, 25.0)
                    traci.vehicle.setParameter(vid, "device.rerouting.probability", "1")
                except Exception:
                    pass
                # Reset distance deltas to avoid counting a teleport
                try:
                    self._last_xy[vid] = None
                    self._last_sumo_dist[vid] = None
                except Exception:
                    pass
                if self.debug:
                    print(f"[ROUTE] agent={agent_id} fallback recreate on {safe_route} and set stop {dest_pa}")
            except Exception as e:
                if self.debug:
                    print(f"[ROUTE] agent={agent_id} fallback recreate failed: {e}")
        # Note: avoid adding an extra generic edge stop; rely solely on ParkingAreaStop like core.py
        try:
            traci.vehicle.setSpeedMode(vid, 0)
            traci.vehicle.setMaxSpeed(vid, 25.0)
            traci.vehicle.setSpeed(vid, -1)
        except Exception:
            pass
        # Ensure rerouting is active for this vehicle
        try:
            traci.vehicle.setParameter(vid, "device.rerouting.probability", "1")
        except Exception:
            pass
        if self.debug:
            try:
                stops = traci.vehicle.getStops(vehID=vid)
                road = traci.vehicle.getRoadID(vid)
                print(f"[ROUTE] agent={agent_id} stops_len={len(stops) if stops else 0} road={road} last_stop={(stops[-1].stoppingPlaceID if stops else None)}")
            except Exception as e:
                print(f"[ROUTE] debug getStops/road failed: {e}")
        # Leaving current parking area
        try:
            self.current_pa[agent_id] = None
        except Exception:
            pass

    def step(self, ticks: int = 1):
        if self._ended:
            return
        for _ in range(max(1, int(ticks))):
            try:
                traci.simulationStep()
                # Ensure our demo vehicles exist (recreate if SUMO removed them for any reason)
                for i in range(self.num_agents):
                    try:
                        self._ensure_vehicle(i)
                    except Exception:
                        pass
                # Update last-known distances for all demo trucks
                for vid in self.veh_ids:
                    try:
                        MAX_STEP_DIST_M = 200.0  # per-tick safety clamp
                        # Update accumulated distance from XY positions
                        pos = traci.vehicle.getPosition(vid)
                        inc_xy = None
                        if pos is not None:
                            last = self._last_xy.get(vid)
                            if last is not None:
                                dx = float(pos[0]) - float(last[0])
                                dy = float(pos[1]) - float(last[1])
                                inc = float(np.hypot(dx, dy))
                                if np.isfinite(inc) and inc >= 0:
                                    # Clamp unrealistically large per-step jumps (teleport/snap)
                                    if inc > MAX_STEP_DIST_M and self.debug:
                                        try:
                                            tnow = traci.simulation.getTime()
                                        except Exception:
                                            tnow = '-'
                                        print(f"[DIST] t={tnow} {vid} XY inc clamped {inc:.2f}m -> {MAX_STEP_DIST_M}m")
                                    inc_xy = min(inc, MAX_STEP_DIST_M)
                            self._last_xy[vid] = (float(pos[0]), float(pos[1]))
                        # Also read SUMO's internal absolute distance and convert to delta
                        inc_sumo = None
                        try:
                            d_abs = float(traci.vehicle.getDistance(vid))
                            if np.isfinite(d_abs) and d_abs >= 0:
                                d_last = self._last_sumo_dist.get(vid, None)
                                if d_last is not None:
                                    delta = d_abs - d_last
                                    if delta >= 0:
                                        if delta > MAX_STEP_DIST_M and self.debug:
                                            try:
                                                tnow = traci.simulation.getTime()
                                            except Exception:
                                                tnow = '-'
                                            print(f"[DIST] t={tnow} {vid} SUMO inc clamped {delta:.2f}m -> {MAX_STEP_DIST_M}m")
                                        inc_sumo = min(delta, MAX_STEP_DIST_M)
                                # Update last absolute regardless
                                self._last_sumo_dist[vid] = d_abs
                        except Exception:
                            pass
                        # Choose a safe delta to add this tick
                        add = 0.0
                        if inc_xy is not None and inc_sumo is not None:
                            add = max(0.0, min(inc_xy, inc_sumo))
                        elif inc_xy is not None:
                            add = max(0.0, inc_xy)
                        elif inc_sumo is not None:
                            add = max(0.0, inc_sumo)
                        if add:
                            self._cum_dist_m[vid] = self._cum_dist_m.get(vid, 0.0) + add
                        prev = self._last_distance_m.get(vid, 0.0)
                        curr = self._cum_dist_m.get(vid, prev)
                        self._last_distance_m[vid] = curr
                        # Update last move time if distance increased
                        try:
                            if (curr - prev) > 5.0:  # require >5m movement to count as progress
                                self._last_move_time[vid] = traci.simulation.getTime()
                        except Exception:
                            pass
                    except Exception:
                        pass
                # Stall recovery: if a vehicle hasn't moved for > 120s and not arrived, retry routing
                try:
                    now_t = traci.simulation.getTime()
                except Exception:
                    now_t = None
                if now_t is not None:
                    for i in range(self.num_agents):
                        vid_i = f"truck_{i}"
                        try:
                            last_t = self._last_move_time.get(vid_i, None)
                            if last_t is None:
                                # Initialize if missing
                                self._last_move_time[vid_i] = now_t
                                continue
                            if (now_t - last_t) > 120 and not self._has_arrived(i):
                                # Re-apply routing to current destination and resume
                                dest = self.destinations.get(i)
                                if dest is not None:
                                    try:
                                        # Recompute route from current edge
                                        dest_lane_id = traci.parkingarea.getLaneID(dest)
                                        dest_edge_id = traci.lane.getEdgeID(dest_lane_id)
                                        src_edge_id = traci.vehicle.getRoadID(vid_i)
                                        if src_edge_id.startswith(":"):
                                            lane_id_curr = traci.vehicle.getLaneID(vid_i)
                                            src_edge_id = traci.lane.getEdgeID(lane_id_curr)
                                        # Always set changeTarget in addition to route
                                        try:
                                            traci.vehicle.changeTarget(vehID=vid_i, edgeID=dest_edge_id)
                                        except Exception:
                                            pass
                                        route_obj = traci.simulation.findRoute(src_edge_id, dest_edge_id)
                                        edges = route_obj.edges if hasattr(route_obj, 'edges') else []
                                        if edges:
                                            traci.vehicle.setRoute(vid_i, edges)
                                        traci.vehicle.clearPendingStops(vid_i)
                                        traci.vehicle.setParkingAreaStop(vid_i, dest)
                                        traci.vehicle.resume(vid_i)
                                        if self.debug:
                                            print(f"[STALL] t={now_t} agent={i} reroute to {dest} edges={len(edges) if edges else 0}")
                                        # Reset last move timer to avoid repeated spam
                                        self._last_move_time[vid_i] = now_t
                                    except Exception:
                                        pass
                        except Exception:
                            pass
            except Exception:
                # SUMO likely ended; mark and stop further stepping
                self._ended = True
                break

    def _has_arrived(self, agent_id: int) -> bool:
        """Arrived if the latest stop is at the destination with arrival > 0 (core.py style)."""
        vid = f"truck_{agent_id}"
        dest_pa = self.destinations.get(agent_id)
        if not dest_pa:
            return False
        # If missing, recreate the vehicle parked at its destination (core.py pattern)
        try:
            try:
                _ids = traci.vehicle.getIDList()
            except Exception:
                _ids = []
            if vid not in _ids:
                if self._ensure_vehicle(agent_id):
                    # Treat as arrived since we recreated it parked at destination
                    self.current_pa[agent_id] = dest_pa
                    if self.debug:
                        try:
                            sim_t = traci.simulation.getTime()
                        except Exception:
                            sim_t = '-'
                        print(f"[ARRIVE] t={sim_t} agent={agent_id} veh={vid} recreated parked at {dest_pa}")
                    return True
        except Exception:
            pass
        try:
            try:
                _ids = traci.vehicle.getIDList()
            except Exception:
                _ids = []
            if vid not in _ids:
                return False
            # Primary check: explicit parking area stop arrival (core.py semantics)
            try:
                stops = traci.vehicle.getStops(vehID=vid)
            except Exception:
                stops = []
            if stops:
                last = stops[-1]
                if getattr(last, 'stoppingPlaceID', None) == dest_pa and getattr(last, 'arrival', -1) >= 0:
                    self.current_pa[agent_id] = dest_pa
                    if self.debug:
                        try:
                            sim_t = traci.simulation.getTime()
                        except Exception:
                            sim_t = '-'
                        print(f"[ARRIVE] t={sim_t} agent={agent_id} veh={vid} at {dest_pa} arrival={getattr(last,'arrival',-1)}")
                    return True
            # Fallback: on the destination edge within ParkingArea span or stopped parking there
            try:
                dest_lane_id = traci.parkingarea.getLaneID(dest_pa)
                dest_edge_id = traci.lane.getEdgeID(dest_lane_id)
                cur_edge = traci.vehicle.getRoadID(vid)
                lane_pos = float(traci.vehicle.getLanePosition(vid))
                start_pos = float(traci.parkingarea.getStartPos(dest_pa))
                end_pos = float(getattr(traci.parkingarea, 'getEndPos')(dest_pa)) if hasattr(traci.parkingarea, 'getEndPos') else start_pos + max(5.0, float(traci.parkingarea.getLength(dest_pa)))
                in_span = (cur_edge == dest_edge_id) and (lane_pos >= start_pos - 2.0) and (lane_pos <= end_pos + 2.0)
                stopped_pk = False
                try:
                    stopped_pk = bool(traci.vehicle.isStoppedParking(vid))
                except Exception:
                    stopped_pk = False
                if in_span or (stopped_pk and cur_edge == dest_edge_id):
                    self.current_pa[agent_id] = dest_pa
                    if self.debug:
                        try:
                            sim_t = traci.simulation.getTime()
                        except Exception:
                            sim_t = '-'
                        print(f"[ARRIVE] t={sim_t} agent={agent_id} veh={vid} reached {dest_pa} by span/stopped cur_edge={cur_edge} pos={lane_pos:.1f}")
                    return True
            except Exception:
                pass
            return False
        except Exception:
            return False

    def operable_agents(self) -> list[int]:
        """Return agent_ids whose vehicles have arrived and are parked at their destination."""
        return [i for i in range(self.num_agents) if self._has_arrived(i)]

    def close(self):
        try:
            traci.close()
        except Exception:
            pass
        self._ended = True

    def ended(self) -> bool:
        return bool(self._ended)

    def get_distance_km(self, agent_id: int) -> float:
        vid = f"truck_{agent_id}"
        # Ensure vehicle exists before querying distance
        try:
            self._ensure_vehicle(agent_id)
        except Exception:
            pass
        d = self._cum_dist_m.get(vid, self._last_distance_m.get(vid, 0.0))
        return max(0.0, float(d) / 1000.0)

    def focus_on(self, agent_id: int):
        """Focus the GUI camera on a given truck if GUI is available."""
        if self._view_id is None:
            return
        try:
            traci.gui.trackVehicle(self._view_id, f"truck_{agent_id}")
        except Exception:
            pass


@dataclass
class TruckStatus:
    truck_id: str
    rul: Optional[float]
    distance_km: float
    destination: str
    maintain: bool
    state: str


class SumoInfoGUI:
    """Simple Tkinter panel showing per-truck status alongside SUMO-GUI."""
    def __init__(self, num_agents: int):
        import tkinter as tk
        from tkinter import ttk

        self.num_agents = num_agents
        self._data: List[TruckStatus] = []
        self._lock = threading.Lock()
        self._stopped = False

        self.root = tk.Tk()
        self.root.title("SUMO Schedule Demo - Truck Status")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        cols = ("Truck", "RUL", "Distance (km)", "Destination", "Maintain", "State")
        self.tree = ttk.Treeview(self.root, columns=cols, show="headings", height=min(20, num_agents))
        for c in cols:
            self.tree.heading(c, text=c)
            width = 130 if c != "Destination" else 160
            self.tree.column(c, width=width, anchor="center")
        self.tree.pack(fill="both", expand=True)

        # Precreate rows
        self._rows = []
        for i in range(num_agents):
            row_id = self.tree.insert("", "end", values=(f"truck_{i}", "-", "-", "-", "-", "-"))
            self._rows.append(row_id)
        # Row click focuses the selected truck in SUMO
        self.tree.bind("<ButtonRelease-1>", self._on_row_click)
        # Start periodic refresh
        self.root.after(200, self._refresh)

    def _on_close(self):
        with self._lock:
            self._stopped = True
        self.root.destroy()

    def stopped(self) -> bool:
        with self._lock:
            return self._stopped

    def update_data(self, data: List[TruckStatus]):
        with self._lock:
            self._data = data

    def _on_row_click(self, event=None):
        # Determine selected item and ask SUMO to focus on it via a callback if hooked
        item = self.tree.focus()
        if not item:
            return
        vals = self.tree.item(item, 'values')
        if not vals:
            return
        name = vals[0]
        if hasattr(self, "on_focus") and callable(self.on_focus):
            try:
                idx = int(str(name).split('_')[-1])
            except Exception:
                return
            self.on_focus(idx)

    def _refresh(self):
        try:
            with self._lock:
                data_copy = list(self._data)
            for i, status in enumerate(data_copy):
                if i >= len(self._rows):
                    break
                rul_str = f"{status.rul:.1f}" if status.rul is not None else "-"
                dist_str = f"{status.distance_km:.2f}"
                dest_str = status.destination
                maint_str = "Yes" if status.maintain else "No"
                state_str = status.state
                self.tree.item(self._rows[i], values=(status.truck_id, rul_str, dist_str, dest_str, maint_str, state_str))
        finally:
            if not self.stopped():
                self.root.after(300, self._refresh)

    def run_mainloop(self):
        self.root.mainloop()


class SumoTrackerGUI:
    """A small Tkinter window to track a single truck with detailed info and camera follow/control."""
    def __init__(self, num_agents: int, parent_root=None):
        import tkinter as tk
        from tkinter import ttk

        self.num_agents = num_agents
        self._data: List[TruckStatus] = []
        self._lock = threading.Lock()
        self._stopped = False
        self._selected_idx = 0
        self._follow = tk.BooleanVar(value=True)

        # Build as Toplevel if parent provided, else own root
        if parent_root is not None:
            self.root = tk.Toplevel(parent_root)
        else:
            self.root = tk.Tk()
        self.root.title("SUMO Schedule Demo - Tracked Truck")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Selection bar
        sel_frame = ttk.Frame(self.root)
        sel_frame.pack(fill="x", padx=8, pady=6)
        ttk.Label(sel_frame, text="Track agent:").pack(side="left")
        self.sel_var = tk.StringVar(value=str(self._selected_idx))
        self.sel_combo = ttk.Combobox(sel_frame, textvariable=self.sel_var, values=[str(i) for i in range(num_agents)], width=6, state="readonly")
        self.sel_combo.pack(side="left", padx=6)
        self.sel_combo.bind("<<ComboboxSelected>>", self._on_select_change)
        ttk.Button(sel_frame, text="Prev", command=self._on_prev).pack(side="left", padx=2)
        ttk.Button(sel_frame, text="Next", command=self._on_next).pack(side="left", padx=2)
        ttk.Checkbutton(sel_frame, text="Follow camera", variable=self._follow, command=self._maybe_focus).pack(side="right")

        # Info grid
        grid = ttk.Frame(self.root)
        grid.pack(fill="both", expand=True, padx=8, pady=6)
        labels = [
            ("Truck", "-"),
            ("RUL", "-"),
            ("Distance (km)", "-"),
            ("Destination", "-"),
            ("Maintain", "-"),
            ("State", "-"),
            ("Speed (m/s)", "-"),
            ("Road", "-"),
            ("LanePos", "-"),
            ("Last Stop", "-"),
        ]
        self._value_vars = {}
        for r, (k, v) in enumerate(labels):
            ttk.Label(grid, text=k+":", width=16, anchor="e").grid(row=r, column=0, sticky="e", padx=4, pady=2)
            var = tk.StringVar(value=v)
            self._value_vars[k] = var
            ttk.Label(grid, textvariable=var, width=40, anchor="w").grid(row=r, column=1, sticky="w", padx=4, pady=2)

        # Expose focus callback like SumoInfoGUI
        self.on_focus = None
        # Periodic refresh
        self.root.after(200, self._refresh)

    def _on_close(self):
        with self._lock:
            self._stopped = True
        try:
            self.root.destroy()
        except Exception:
            pass

    def stopped(self) -> bool:
        with self._lock:
            return self._stopped

    def update_data(self, data: List[TruckStatus]):
        with self._lock:
            self._data = data

    def _on_prev(self):
        idx = (int(self.sel_var.get()) - 1) % self.num_agents
        self.sel_var.set(str(idx))
        self._on_select_change()

    def _on_next(self):
        idx = (int(self.sel_var.get()) + 1) % self.num_agents
        self.sel_var.set(str(idx))
        self._on_select_change()

    def _on_select_change(self, event=None):
        try:
            self._selected_idx = int(self.sel_var.get())
        except Exception:
            return
        self._maybe_focus()

    def _maybe_focus(self):
        if self._follow.get() and hasattr(self, "on_focus") and callable(self.on_focus):
            try:
                self.on_focus(self._selected_idx)
            except Exception:
                pass

    def _refresh(self):
        try:
            with self._lock:
                data_copy = list(self._data)
            # Pick status for selected idx
            s = None
            if 0 <= self._selected_idx < len(data_copy):
                s = data_copy[self._selected_idx]
            # Update basic fields from cached status
            if s is not None:
                self._value_vars["Truck"].set(s.truck_id)
                self._value_vars["RUL"].set("-" if s.rul is None else f"{s.rul:.1f}")
                self._value_vars["Distance (km)"].set(f"{s.distance_km:.2f}")
                self._value_vars["Destination"].set(s.destination)
                self._value_vars["Maintain"].set("Yes" if s.maintain else "No")
                self._value_vars["State"].set(s.state)
            # Live kinematics from SUMO
            try:
                vid = f"truck_{self._selected_idx}"
                spd = traci.vehicle.getSpeed(vid)
                road = traci.vehicle.getRoadID(vid)
                lane_pos = traci.vehicle.getLanePosition(vid)
                stops = traci.vehicle.getStops(vid)
                last_stop = None
                if stops:
                    last = stops[-1]
                    last_stop = f"{getattr(last,'stoppingPlaceID',None)} (arr={getattr(last,'arrival',None)})"
                self._value_vars["Speed (m/s)"].set(f"{float(spd):.2f}")
                self._value_vars["Road"].set(str(road))
                self._value_vars["LanePos"].set(f"{float(lane_pos):.1f}")
                self._value_vars["Last Stop"].set(last_stop or "-")
            except Exception:
                # Vehicle may be missing or SUMO headless
                pass
        finally:
            if not self.stopped():
                self.root.after(300, self._refresh)


def parse_demo_args():
    parser = argparse.ArgumentParser(description="SUMO demo evaluation for async-MAPPO policy")
    # Policy/env args
    parser.add_argument('--scenario_name', type=str, default='rul_schedule')
    parser.add_argument('--num_agents', type=int, default=12)
    parser.add_argument('--max_steps', type=int, default=800)
    parser.add_argument('--n_rollout_threads', type=int, default=1)
    parser.add_argument('--use_eval', action='store_false')  # not used, kept for config compatibility
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--cuda_deterministic', action='store_true', default=False)
    parser.add_argument('--n_training_threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--algorithm_name', type=str, default='mappo')
    parser.add_argument('--env_name', type=str, default='async_schedule')
    parser.add_argument('--experiment_name', type=str, default='sumo_demo')
    parser.add_argument('--use_rul_agent', action='store_true', default=True)
    parser.add_argument('--rul_threshold', type=float, default=7)
    parser.add_argument('--rul_state', action='store_false')  # policy trained without RUL in obs by default

    # Checkpoint and SUMO
    parser.add_argument('--actor_dir', type=str, required=True, help='Directory containing actor.pt')
    parser.add_argument('--sumo_cfg', type=str, required=True, help='Path to SUMO .sumocfg file')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--headless', action='store_true', help='Run SUMO headless (no GUI); useful for testing')
    parser.add_argument('--exp_type', type=str, default='rul', help='Experiment type for pathing in offline env')
    parser.add_argument('--sim_time_limit', type=int, default=int(7*24*3600), help='Upper limit of SUMO simulation time (seconds)')
    # Defaults required by lower-level components
    parser.add_argument('--asynch', action='store_true', default=True, help='Use asynchronous buffer/control')
    # Debugging
    parser.add_argument('--debug', action='store_true', help='Enable verbose debug logging')
    return parser


def main():
    demo_parser = parse_demo_args()
    # Merge with project config to stay compatible with runner
    base_parser = get_config()
    # parse_known_args twice to allow mixing
    demo_args, _ = demo_parser.parse_known_args()
    all_args = base_parser.parse_known_args([])[0]

    # Hydrate fields from demo args
    for k, v in vars(demo_args).items():
        setattr(all_args, k, v)
    # Force-disable wandb and rendering for demo
    all_args.use_wandb = False
    all_args.use_render = False
    # If debugging, cap sim time to 10000s as requested
    if getattr(all_args, 'debug', False):
        all_args.sim_time_limit = 10000

    # Infer model architecture from checkpoint (recurrent flags, layers, hidden size)
    try:
        ckpt = torch.load(os.path.join(demo_args.actor_dir, 'actor.pt'), map_location='cpu')
        if isinstance(ckpt, dict):
            keys = list(ckpt.keys())
            has_rnn = any(k.startswith('rnn.rnn.') for k in keys)
            if has_rnn:
                # Determine number of GRU layers
                layer_ids = []
                for k in keys:
                    if k.startswith('rnn.rnn.weight_ih_l'):
                        try:
                            layer_ids.append(int(k.split('weight_ih_l')[-1]))
                        except Exception:
                            pass
                if layer_ids:
                    recurrent_N = max(layer_ids) + 1
                    setattr(all_args, 'recurrent_N', recurrent_N)
                # Determine hidden size from LayerNorm weight or GRU weight
                hid = None
                for k in keys:
                    if k == 'rnn.norm.weight':
                        try:
                            hid = int(ckpt[k].numel())
                            break
                        except Exception:
                            pass
                if hid is None:
                    for k in keys:
                        if k.startswith('rnn.rnn.weight_hh_l0'):
                            try:
                                hid = int(ckpt[k].shape[1])
                                break
                            except Exception:
                                pass
                if hid is not None:
                    setattr(all_args, 'hidden_size', hid)
                setattr(all_args, 'use_recurrent_policy', True)
                setattr(all_args, 'use_naive_recurrent_policy', False)
    except Exception as e:
        # Non-fatal; fall back to parser defaults
        print(f"[Demo] Note: could not infer recurrent config from checkpoint: {e}")

    # Device
    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # Build envs (offline env used by policy)
    envs = make_envs(all_args)
    obs_space = envs.observation_space[0]
    share_obs_space = envs.share_observation_space[0]
    act_space = envs.action_space[0]

    # Ensure required policy flags have defaults
    for attr, default in [
        ("grid_goal", False),
        ("use_attn_internal", True),
        ("use_cat_self", True),
        ("use_policy_active_masks", True),
        ("use_naive_recurrent_policy", False),
        ("use_recurrent_policy", False),
        ("use_policy_vhead", False),
    ]:
        if not hasattr(all_args, attr):
            setattr(all_args, attr, default)

    # Create policy and load actor weights directly
    policy = R_MAPPOPolicy(all_args, obs_space, share_obs_space, act_space, device=device)
    actor_state = torch.load(os.path.join(all_args.actor_dir, 'actor.pt'), map_location=device)
    policy.actor.load_state_dict(actor_state)
    policy.actor.eval()
    print(f"Loaded actor from: {all_args.actor_dir}")

    # Start SUMO demo world (with GUI)
    sumo = SumoDemo(num_agents=all_args.num_agents, sumo_cfg=all_args.sumo_cfg, factory_count=50, headless=demo_args.headless, debug=getattr(all_args, 'debug', False))

    # Build status GUI and tracker GUI
    gui = None if demo_args.headless else SumoInfoGUI(num_agents=all_args.num_agents)
    tracker = None
    if gui is not None:
        # Connect GUI focus callback to SUMO camera
        setattr(gui, 'on_focus', lambda idx: sumo.focus_on(idx))
        # Secondary tracker window to follow a single agent
        tracker = SumoTrackerGUI(num_agents=all_args.num_agents, parent_root=gui.root)
        setattr(tracker, 'on_focus', lambda idx: sumo.focus_on(idx))

    # Access the underlying offline env to read RUL/state when available
    # ScheduleEnv holds a list of envs, we run only one here
    offline_env = envs.envs[0]

    # RNN state and masks per-agent
    recurrent_N = getattr(all_args, 'recurrent_N', 1)
    hidden_size = getattr(all_args, 'hidden_size', 64)
    rnn_states = np.zeros((all_args.num_agents, recurrent_N, hidden_size), dtype=np.float32)
    masks = np.ones((all_args.num_agents, 1), dtype=np.float32)

    stop_flag = {"stop": False}

    def _build_statuses() -> List[TruckStatus]:
        statuses: List[TruckStatus] = []
        for i in range(all_args.num_agents):
            vid = f"truck_{i}"
            # Ensure vehicle exists, then get distance via SUMO with clamping/caching
            try:
                sumo._ensure_vehicle(i)
            except Exception:
                pass
            dist_km = sumo.get_distance_km(i)
            # RUL from offline env for display only
            rul_val = None
            dest_display = None
            maintain = False
            state_str = "waiting"
            try:
                rul_val = float(offline_env.truck_agents[i].rul)
            except Exception:
                pass
            # Destination: prefer offline env intended destination; show Workshop if maintaining
            try:
                if hasattr(offline_env.truck_agents[i], 'state') and offline_env.truck_agents[i].state == 'maintain':
                    maintain = True
                    dest_display = 'Workshop'
                else:
                    dest_display = str(getattr(offline_env.truck_agents[i], 'destination', '-')) or sumo.destinations.get(i)
            except Exception:
                dest_display = str(sumo.destinations.get(i) or '-')
            # State: derive purely from SUMO kinematics so it changes every step
            try:
                try:
                    _ids = traci.vehicle.getIDList()
                except Exception:
                    _ids = []
                if vid in _ids:
                    if traci.vehicle.isStoppedParking(vid):
                        state_str = 'arrived' if sumo._has_arrived(i) else 'waiting'
                    else:
                        spd = float(traci.vehicle.getSpeed(vid))
                        state_str = 'driving' if spd > 0.1 else 'stopped'
                else:
                    state_str = 'missing'
            except Exception:
                state_str = state_str or '-'
            statuses.append(TruckStatus(
                truck_id=vid,
                rul=rul_val,
                distance_km=dist_km,
                destination=str(dest_display) if dest_display else "-",
                maintain=maintain,
                state=state_str
            ))
        return statuses

    def sim_loop():
        try:
            episodes = all_args.episodes
            for ep in range(episodes):
                if (gui is not None) and gui.stopped():
                    break
                if sumo.ended():
                    break
                print(f"[SUMO Demo] Episode {ep+1}/{episodes}")
                # Reset offline env and per-episode states
                dict_obs_list = envs.reset()
                # envs.reset returns array-like per env; use first
                dict_obs = dict_obs_list[0]
                rnn_states[:] = 0
                masks[:] = 1
                step_cnt = 0
                # Proactively dispatch every truck once to break initial waiting deadlock
                # Use a simple different factory from current position; maintenance -> Factory0
                try:
                    for i in range(all_args.num_agents):
                        try:
                            if offline_env.truck_agents[i].state == 'maintain':
                                sumo.set_destination(agent_id=i, dest_index=0)
                                continue
                        except Exception:
                            pass
                        try:
                            cur_pos = str(offline_env.truck_agents[i].position)
                            cur_idx = int(cur_pos.replace('Factory', ''))
                        except Exception:
                            # fallback to a pseudo-random different index
                            cur_idx = i % 50
                        # Choose a deterministic different destination initially
                        dest_idx = (cur_idx + 1) % 50
                        # Snap to an existing parking area if the target doesn't exist
                        try:
                            pa = sumo._resolve_pa(dest_idx)
                            if pa is None:
                                continue
                        except Exception:
                            pass
                        sumo.set_destination(agent_id=i, dest_index=dest_idx)
                        # Strong nudge to depart from initial parking
                        try:
                            vid = f"truck_{i}"
                            traci.vehicle.setSpeedMode(vid, 0)  # permissive to leave parking
                            traci.vehicle.resume(vid)
                            traci.vehicle.setSpeed(vid, -1)
                        except Exception:
                            pass
                except Exception:
                    pass
                # Helper: choose nearest different factory if action equals current pos
                def pick_moving_dest(agent_id: int, proposed_idx: int) -> int:
                    try:
                        cur_pos = offline_env.truck_agents[int(agent_id)].position
                        cur_idx = int(str(cur_pos).replace('Factory',''))
                        if int(proposed_idx) != cur_idx:
                            return int(proposed_idx)
                        # Find nearest different destination
                        md = offline_env.truck_agents[int(agent_id)].map_distance
                        candidates = []
                        for k, v in md.items():
                            if not str(k).startswith(cur_pos + '_to_'):
                                continue
                            try:
                                tgt = str(k).split('_to_')[1]
                                tgt_idx = int(tgt.replace('Factory',''))
                                if tgt_idx != cur_idx:
                                    candidates.append((float(v), tgt_idx))
                            except Exception:
                                continue
                        if candidates:
                            candidates.sort(key=lambda x: x[0])
                            return int(candidates[0][1])
                    except Exception:
                        pass
                    # default fallback
                    return (int(proposed_idx) + 1) % 50

                # If we have any operable agents at t=0, pick an initial destination
                if isinstance(dict_obs, dict) and len(dict_obs) > 0:
                    agent_ids = sorted(list(dict_obs.keys()))
                    obs_batch = np.stack([dict_obs[i] for i in agent_ids], axis=0)
                    rnn_batch = rnn_states[agent_ids]
                    mask_batch = masks[agent_ids]
                    if getattr(all_args, 'debug', False):
                        try:
                            print(f"[RL] t={traci.simulation.getTime()} initial decision for agents={agent_ids}")
                        except Exception:
                            print(f"[RL] initial decision for agents={agent_ids}")
                        print(f"[RL] obs.shape={obs_batch.shape} rnn.shape={rnn_batch.shape} mask.shape={mask_batch.shape}")
                    with torch.no_grad():
                        obs_t = torch.from_numpy(obs_batch).to(device=device, dtype=torch.float32)
                        rnn_t = torch.from_numpy(rnn_batch).to(device=device, dtype=torch.float32)
                        mask_t = torch.from_numpy(mask_batch).to(device=device, dtype=torch.float32)
                        actions_t, _, new_rnn_t = policy.actor(obs_t, rnn_t, mask_t, available_actions=None, deterministic=True)
                    new_rnn_np = new_rnn_t.detach().cpu().numpy()
                    actions_np = actions_t.detach().cpu().numpy().astype(np.int64).reshape(-1)
                    if getattr(all_args, 'debug', False):
                        print(f"[RL] actions={actions_np.tolist()}")
                    for idx, agent_id in enumerate(agent_ids):
                        rnn_states[agent_id] = new_rnn_np[idx]
                        dest_idx = pick_moving_dest(int(agent_id), int(actions_np[idx]))
                        # Maintenance maps to Factory0
                        if (not all_args.use_rul_agent) and dest_idx >= 50:
                            dest_idx = 0
                        if getattr(all_args, 'debug', False):
                            try:
                                cur_pos = str(offline_env.truck_agents[int(agent_id)].position)
                            except Exception:
                                cur_pos = "?"
                            print(f"[RL->SUMO] agent={int(agent_id)} action={int(actions_np[idx])} mapped_dest={dest_idx} cur_pos={cur_pos}")
                        sumo.set_destination(agent_id=int(agent_id), dest_index=dest_idx)
                # Kickstart: step SUMO a bit after initial dispatch so trucks actually depart
                # and GUI readings begin to change even before first operable arrival
                for _ in range(60):
                    if (gui is not None) and gui.stopped():
                        stop_flag["stop"] = True
                        break
                    if sumo.ended():
                        stop_flag["stop"] = True
                        break
                    try:
                        if traci.simulation.getTime() >= all_args.sim_time_limit:
                            if getattr(all_args, 'debug', False):
                                print(f"[LIM] Reached sim_time_limit={all_args.sim_time_limit} during kickstart")
                            stop_flag["stop"] = True
                            break
                    except Exception:
                        pass
                    sumo.step(1)
                    # Nudge any parked trucks to resume during kickstart
                    for i in range(all_args.num_agents):
                        try:
                            vid = f"truck_{i}"
                            if traci.vehicle.isStoppedParking(vid):
                                traci.vehicle.setSpeedMode(vid, 0)
                                traci.vehicle.resume(vid)
                        except Exception:
                            pass
                    if gui is not None:
                        _d = _build_statuses()
                        gui.update_data(_d)
                        try:
                            tracker.update_data(_d)
                        except Exception:
                            pass
                while True:
                    if (gui is not None) and gui.stopped():
                        stop_flag["stop"] = True
                        break
                    if sumo.ended():
                        stop_flag["stop"] = True
                        break
                    try:
                        if traci.simulation.getTime() >= all_args.sim_time_limit:
                            if getattr(all_args, 'debug', False):
                                print(f"[LIM] Reached sim_time_limit={all_args.sim_time_limit}; stopping episode")
                            stop_flag["stop"] = True
                            break
                    except Exception:
                        pass
                    # Always push live GUI updates
                    if gui is not None:
                        _d = _build_statuses()
                        gui.update_data(_d)
                        try:
                            tracker.update_data(_d)
                        except Exception:
                            pass
                    # Ensure dict_obs is a dict (when no one is operable it may be empty or None)
                    if not isinstance(dict_obs, dict):
                        dict_obs = {}
                    # Compute actions for currently operable agents
                    act_dict = {}
                    if isinstance(dict_obs, dict) and len(dict_obs) > 0:
                        agent_ids = sorted(list(dict_obs.keys()))
                        obs_batch = np.stack([dict_obs[i] for i in agent_ids], axis=0)
                        rnn_batch = rnn_states[agent_ids]
                        mask_batch = masks[agent_ids]
                        if getattr(all_args, 'debug', False):
                            try:
                                print(f"[RL] t={traci.simulation.getTime()} step decision for agents={agent_ids}")
                            except Exception:
                                print(f"[RL] step decision for agents={agent_ids}")
                            print(f"[RL] obs.shape={obs_batch.shape} rnn.shape={rnn_batch.shape} mask.shape={mask_batch.shape}")
                        with torch.no_grad():
                            obs_t = torch.from_numpy(obs_batch).to(device=device, dtype=torch.float32)
                            rnn_t = torch.from_numpy(rnn_batch).to(device=device, dtype=torch.float32)
                            mask_t = torch.from_numpy(mask_batch).to(device=device, dtype=torch.float32)
                            actions_t, _, new_rnn_t = policy.actor(obs_t, rnn_t, mask_t, available_actions=None, deterministic=True)
                        new_rnn_np = new_rnn_t.detach().cpu().numpy()
                        actions_np = actions_t.detach().cpu().numpy().astype(np.int64).reshape(-1)
                        if getattr(all_args, 'debug', False):
                            print(f"[RL] actions={actions_np.tolist()}")
                        for idx, agent_id in enumerate(agent_ids):
                            rnn_states[agent_id] = new_rnn_np[idx]
                            act_dict[int(agent_id)] = int(actions_np[idx])

                    # Mirror newly chosen actions to SUMO (if any) and detect wait actions
                    wait_detected = False
                    for agent_id, dest_idx in act_dict.items():
                        # If offline env truck is maintaining, always route to Factory0 and skip other moves
                        try:
                            if offline_env.truck_agents[agent_id].state == 'maintain':
                                if getattr(all_args, 'debug', False):
                                    print(f"[RL->SUMO] agent={agent_id} in maintain: route to Factory0")
                                sumo.set_destination(agent_id=agent_id, dest_index=0)
                                continue
                        except Exception:
                            pass
                        # If action equals current position, pick a nearby different factory to visualize movement
                        move_idx = pick_moving_dest(int(agent_id), int(dest_idx))
                        # Consider wait only if mapped move equals current position
                        try:
                            cur_pos = str(offline_env.truck_agents[int(agent_id)].position)
                            cur_idx = int(cur_pos.replace('Factory',''))
                            if int(move_idx) == cur_idx:
                                wait_detected = True
                        except Exception:
                            pass
                        if getattr(all_args, 'debug', False):
                            try:
                                cur_pos = str(offline_env.truck_agents[int(agent_id)].position)
                            except Exception:
                                cur_pos = "?"
                            print(f"[RL->SUMO] agent={agent_id} action={dest_idx} mapped_move={move_idx} cur_pos={cur_pos}")
                        sumo.set_destination(agent_id=agent_id, dest_index=move_idx)
                        # Strong nudge to depart if currently parked
                        try:
                            vid = f"truck_{agent_id}"
                            if traci.vehicle.isStoppedParking(vid):
                                traci.vehicle.setSpeedMode(vid, 0)
                                traci.vehicle.resume(vid)
                                traci.vehicle.setSpeed(vid, -1)
                        except Exception:
                            pass
                    # Advance SUMO until at least one truck has arrived (operable)
                    operable = []
                    safety_ticks = 0
                    # If we detected a wait/no-op, simulate 120s and proceed without waiting for arrival
                    if wait_detected:
                        if getattr(all_args, 'debug', False):
                            print(f"[SIM] wait/no-op detected; fast-forward 120 ticks")
                        for _ in range(120):
                            try:
                                if traci.simulation.getTime() >= all_args.sim_time_limit:
                                    if getattr(all_args, 'debug', False):
                                        print(f"[LIM] Reached sim_time_limit={all_args.sim_time_limit} during fast-forward")
                                    stop_flag["stop"] = True
                                    break
                            except Exception:
                                pass
                            sumo.step(1)
                            if sumo.ended():
                                stop_flag["stop"] = True
                                break
                            if gui is not None:
                                _d = _build_statuses()
                                gui.update_data(_d)
                                try:
                                    tracker.update_data(_d)
                                except Exception:
                                    pass
                    else:
                        while len(operable) == 0 and not ((gui is not None) and gui.stopped()):
                            try:
                                if traci.simulation.getTime() >= all_args.sim_time_limit:
                                    if getattr(all_args, 'debug', False):
                                        print(f"[LIM] Reached sim_time_limit={all_args.sim_time_limit} while waiting for arrival")
                                    stop_flag["stop"] = True
                                    break
                            except Exception:
                                pass
                            sumo.step(1)
                            if sumo.ended():
                                stop_flag["stop"] = True
                                break
                            # Nudge any parked trucks to resume
                            for i in range(all_args.num_agents):
                                try:
                                    vid = f"truck_{i}"
                                    if traci.vehicle.isStoppedParking(vid):
                                        traci.vehicle.setSpeedMode(vid, 0)
                                        traci.vehicle.resume(vid)
                                except Exception:
                                    pass
                            operable = sumo.operable_agents()
                            safety_ticks += 1
                            if getattr(all_args, 'debug', False) and (safety_ticks % 100 == 0):
                                try:
                                    t_now = traci.simulation.getTime()
                                except Exception:
                                    t_now = '-'
                                print(f"[SIM] t={t_now} waiting for arrival... ticks={safety_ticks}")
                            if getattr(all_args, 'debug', False) and (safety_ticks % 300 == 0):
                                # Dump detailed status for all trucks occasionally
                                for aid in range(all_args.num_agents):
                                    sumo._dump_vehicle_status(aid)
                            # Keep GUI live while waiting for first arrival
                            if gui is not None:
                                _d = _build_statuses()
                                gui.update_data(_d)
                                try:
                                    tracker.update_data(_d)
                                except Exception:
                                    pass
                        # Periodically push GUI updates while waiting
                        if gui is not None:
                            _d = _build_statuses()
                            gui.update_data(_d)
                            try:
                                tracker.update_data(_d)
                            except Exception:
                                pass

                        if getattr(all_args, 'debug', False):
                            try:
                                t_now = traci.simulation.getTime()
                            except Exception:
                                t_now = '-'
                            print(f"[SIM] t={t_now} first arrival after ticks={safety_ticks}; operable={operable}")

                        if safety_ticks > 10000:
                            break

                    # Step the offline policy env
                    if sumo.ended():
                        break
                    action_env = [act_dict]
                    dict_obs_list, rewards, dones, infos = envs.step(action_env)
                    dict_obs = dict_obs_list[0]
                    if getattr(all_args, 'debug', False):
                        try:
                            t_now = traci.simulation.getTime()
                        except Exception:
                            t_now = '-'
                        print(f"[ENV] t={t_now} env.step done; next_operable_keys={sorted(list(dict_obs.keys())) if isinstance(dict_obs, dict) else 'NA'}")
                        try:
                            # rewards is a list-like of dicts or arrays; print compact summary
                            print(f"[ENV] rewards_summary={type(rewards)} dones_summary={dones}")
                        except Exception:
                            pass
                    # Push an update after env step
                    if gui is not None:
                        _d = _build_statuses()
                        gui.update_data(_d)
                        try:
                            tracker.update_data(_d)
                        except Exception:
                            pass

                    # Termination conditions
                    done_flag = False
                    for env_done in dones:
                        if 'bool' in env_done.__class__.__name__:
                            if env_done:
                                done_flag = True
                        else:
                            if np.all(env_done):
                                done_flag = True
                    step_cnt += 1
                    if (step_cnt >= all_args.max_steps) or done_flag:
                        break

                print(f"[SUMO Demo] Episode {ep+1} finished after {step_cnt} steps")
                if stop_flag["stop"]:
                    break
        finally:
            try:
                sumo.close()
            except Exception:
                pass
            try:
                envs.close()
            except Exception:
                pass

    # Run simulation in a background thread and the GUI mainloop in the main thread
    if gui is None:
        # headless: just run the simulation loop synchronously
        sim_loop()
    else:
        t = threading.Thread(target=sim_loop, daemon=True)
        t.start()
        gui.run_mainloop()
        # Join thread briefly on exit
        try:
            t.join(timeout=1.0)
        except Exception:
            pass


if __name__ == "__main__":
    main()
