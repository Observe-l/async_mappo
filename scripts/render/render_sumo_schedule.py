#!/usr/bin/env python3
"""
Demo: Evaluate a trained async-MAPPO policy in a live SUMO simulation.

This script now drives policy decisions directly from a SUMO-backed environment
(`async_scheduling_sumo`) rather than the offline async_scheduling wrapper.
The policy observes the SUMO world via that environment and dispatches trucks
accordingly while optional Tkinter GUIs visualise their state.

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
import argparse
import random
from pathlib import Path
import threading
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import torch

"""
SUMO TraCI binding
- We do not start SUMO here; the env (async_scheduling_sumo) controls lifecycle.
- Import libsumo by default; the env will rebind to GUI traci when use_sumo_gui=True.
"""
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
from onpolicy.envs.rul_schedule.sumo_schedule import async_scheduling_sumo
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
class SumoDemo:
    """
    Deprecated: Controller formerly responsible for starting SUMO and direct TraCI control.
    Kept for backward compatibility but no longer starts SUMO; env.reset() manages lifecycle.
    """
    def __init__(self, *args, **kwargs):
        raise RuntimeError("SumoDemo is deprecated; the demo now uses async_scheduling_sumo directly.")

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

    def _start_sumo(self, *args, **kwargs):
        raise RuntimeError("SumoDemo no longer manages SUMO startup.")

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
            # Do not override colors here; env will set based on cargo
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
            # Do not override colors here; env will set based on cargo
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
            # Parking-aware routing to produce a route terminating at the ParkingArea
            try:
                traci.vehicle.rerouteParkingArea(vid, dest_pa)
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
                # Reapply params (do not override color; env handles visuals)
                try:
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

    def focus_on(self, agent_id: Optional[int]):
        """Control GUI camera following. If agent_id is None or <0, stop following."""
        if self._view_id is None:
            return
        try:
            if agent_id is None or int(agent_id) < 0:
                # Stop following the current vehicle (best-effort sequence)
                done = False
                try:
                    if hasattr(traci.gui, 'trackOff'):
                        traci.gui.trackOff(self._view_id)
                        done = True
                except Exception:
                    pass
                if not done:
                    for unfollow_arg in ("", None, "off", " ", "__none__"):
                        try:
                            traci.gui.trackVehicle(self._view_id, unfollow_arg)  # type: ignore[arg-type]
                            done = True
                            break
                        except Exception:
                            continue
                # As a last resort, tweak view without tracking
                try:
                    traci.gui.setZoom(self._view_id, 1200)
                except Exception:
                    pass
            else:
                traci.gui.trackVehicle(self._view_id, f"truck_{int(agent_id)}")
        except Exception:
            pass


class SumoMonitor:
    """Lightweight helper for visualisation and telemetry over a running SUMO sim."""

    def __init__(self, num_agents: int, headless: bool = False, debug: bool = False):
        self.num_agents = num_agents
        self.headless = bool(headless)
        self.debug = bool(debug)
        self.veh_ids = [f"truck_{i}" for i in range(self.num_agents)]
        self._view_id: Optional[str] = None
        self._focus_lock = threading.Lock()
        self._pending_focus: Optional[int] = None
        self._pending_unfollow = False
        self.refresh()

    def refresh(self):
        """Reapply basic rendering and tracking configuration."""
        self._setup_view()
        self._setup_vehicles()

    def _setup_view(self):
        if self.headless:
            return
        try:
            views = list(traci.gui.getIDList())
            if views:
                if self._view_id not in views:
                    self._view_id = views[0]
                if self._view_id:
                    try:
                        traci.gui.setZoom(self._view_id, 1200)
                    except Exception:
                        pass
                    try:
                        traci.gui.trackVehicle(self._view_id, self.veh_ids[0])
                    except Exception:
                        pass
        except Exception:
            self._view_id = None

    def _setup_vehicles(self):
        for vid in self.veh_ids:
            try:
                traci.vehicle.setColor(vid, (255, 0, 255, 255))
            except Exception:
                pass
            try:
                traci.vehicle.setSpeedMode(vid, 0)
                traci.vehicle.setMaxSpeed(vid, 25.0)
                traci.vehicle.setSpeed(vid, -1)
            except Exception:
                pass

    def get_distance_km(self, agent_id: int) -> float:
        vid = f"truck_{int(agent_id)}"
        try:
            dist = traci.vehicle.getDistance(vid)
            if dist is None:
                return 0.0
            return max(0.0, float(dist) / 1000.0)
        except Exception:
            return 0.0

    def focus_on(self, agent_id: Optional[int]):
        with self._focus_lock:
            if agent_id is None or int(agent_id) < 0:
                self._pending_focus = None
                self._pending_unfollow = True
            else:
                self._pending_focus = int(agent_id)
                self._pending_unfollow = False

    def apply_pending_focus(self):
        if self.headless:
            return
        with self._focus_lock:
            focus_idx = self._pending_focus
            unfollow = self._pending_unfollow and focus_idx is None
            self._pending_focus = None
            self._pending_unfollow = False
        if (focus_idx is None) and not unfollow:
            return
        if self._view_id is None:
            self._setup_view()
        if self._view_id is None:
            return
        try:
            if focus_idx is None and unfollow:
                if hasattr(traci.gui, 'trackOff'):
                    try:
                        traci.gui.trackOff(self._view_id)
                        return
                    except Exception:
                        pass
                for unfollow_arg in ("", None, "off", " "):
                    try:
                        traci.gui.trackVehicle(self._view_id, unfollow_arg)  # type: ignore[arg-type]
                        return
                    except Exception:
                        continue
            elif focus_idx is not None:
                traci.gui.trackVehicle(self._view_id, self.veh_ids[focus_idx])
        except Exception:
            pass

    def ended(self) -> bool:
        try:
            traci.simulation.getTime()
            return False
        except Exception:
            return True

    def close(self):
        try:
            traci.close()
        except Exception:
            pass


@dataclass
class TruckStatus:
    truck_id: str
    rul: Optional[float]
    distance_km: float
    destination: str
    cargo: str
    maintain: bool
    state: str
    speed_m_s: Optional[float]
    road_id: Optional[str]
    lane_pos_m: Optional[float]
    last_stop: Optional[str]


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

        cols = ("Truck", "RUL", "Distance (km)", "Destination", "Cargo", "Maintain", "State")
        self.tree = ttk.Treeview(self.root, columns=cols, show="headings", height=min(20, num_agents))
        for c in cols:
            self.tree.heading(c, text=c)
            width = 130 if c not in ("Destination", "Cargo") else 180
            self.tree.column(c, width=width, anchor="center")
        self.tree.pack(fill="both", expand=True)

        # Precreate rows
        self._rows = []
        for i in range(num_agents):
            row_id = self.tree.insert("", "end", values=(f"truck_{i}", "-", "-", "-", "-", "-", "-"))
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
            # If external tracker exposes a shared follow flag and it's off, unfocus
            try:
                follow_flag = getattr(self, "follow_flag", None)
                if follow_flag is not None and callable(getattr(follow_flag, 'get', None)):
                    if not bool(follow_flag.get()):
                        self.on_focus(None)
                        return
            except Exception:
                pass
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
                cargo_str = status.cargo
                maint_str = "Yes" if status.maintain else "No"
                state_str = status.state
                self.tree.item(self._rows[i], values=(status.truck_id, rul_str, dist_str, dest_str, cargo_str, maint_str, state_str))
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
        if hasattr(self, "on_focus") and callable(self.on_focus):
            try:
                if self._follow.get():
                    self.on_focus(self._selected_idx)
                else:
                    # Explicitly request unfollow
                    self.on_focus(None)
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
                if s.speed_m_s is None:
                    self._value_vars["Speed (m/s)"].set("-")
                else:
                    self._value_vars["Speed (m/s)"].set(f"{s.speed_m_s:.2f}")
                self._value_vars["Road"].set(s.road_id or "-")
                if s.lane_pos_m is None:
                    self._value_vars["LanePos"].set("-")
                else:
                    self._value_vars["LanePos"].set(f"{s.lane_pos_m:.1f}")
                self._value_vars["Last Stop"].set(s.last_stop or "-")
            else:
                # No status available yet; reset fields
                for key in ["Truck", "RUL", "Distance (km)", "Destination", "Maintain", "State",
                            "Speed (m/s)", "Road", "LanePos", "Last Stop"]:
                    self._value_vars[key].set("-")
                self._value_vars["Truck"].set(f"truck_{self._selected_idx}")
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
    parser.add_argument('--rul_state', action='store_true', default=False, help='Include RUL in observations (default: disabled)')

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

    # Do not bind GUI traci here; env will rebind to GUI when use_sumo_gui=True
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

    # Build SUMO-backed evaluation environment
    all_args.use_sumo_gui = not demo_args.headless
    env = async_scheduling_sumo(all_args)
    obs_space = env.observation_space[0]
    share_obs_space = env.share_observation_space[0]
    act_space = env.action_space[0]

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

    # If using GUI, start SUMO immediately (on main thread) so the SUMO window appears,
    # then rebind this module's traci to the GUI client so monitor/GUI can control it.
    initial_obs = None
    used_initial_reset = False
    if all_args.use_sumo_gui:
        try:
            initial_obs = env.reset()
            used_initial_reset = True
            # Rebind to GUI traci so SumoMonitor and GUIs talk to the same TraCI session
            try:
                import traci as traci_client  # type: ignore
                globals()['traci'] = traci_client
            except Exception as exc:
                print(f"[SUMO Demo] Warning: failed to bind GUI traci in render script: {exc}")
            # Wait briefly for SUMO-GUI view to become available
            try:
                import time as _time
                ready = False
                for _ in range(30):  # up to ~3s
                    try:
                        views = list(traci.gui.getIDList())
                        if views:
                            print(f"[SUMO Demo] SUMO GUI ready (views={views})")
                            ready = True
                            break
                    except Exception:
                        pass
                    _time.sleep(0.1)
                if not ready:
                    print("[SUMO Demo] Waiting for SUMO GUI view timed out; continuing")
            except Exception:
                pass
        except Exception as exc:
            print(f"[SUMO Demo] initial reset failed: {exc}")

    # Visualisation helper (only affects GUI-mode)
    monitor = SumoMonitor(num_agents=all_args.num_agents, headless=demo_args.headless, debug=getattr(all_args, 'debug', False))

    # Build status GUI and tracker GUI
    gui = None if demo_args.headless else SumoInfoGUI(num_agents=all_args.num_agents)
    tracker = None
    if gui is not None:
        # Connect GUI focus callback to SUMO camera
        setattr(gui, 'on_focus', lambda idx: monitor.focus_on(idx))
        # Secondary tracker window to follow a single agent
        tracker = SumoTrackerGUI(num_agents=all_args.num_agents, parent_root=gui.root)
        setattr(tracker, 'on_focus', lambda idx: monitor.focus_on(idx))
        tracker._maybe_focus()
        # Share the follow flag so list clicks respect toggle
        try:
            gui.follow_flag = tracker._follow
        except Exception:
            pass

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
            dist_km = monitor.get_distance_km(i)
            rul_val = None
            if hasattr(env, 'rul_values'):
                try:
                    val = env.rul_values.get(i)
                    if val is not None:
                        rul_val = float(val)
                except Exception:
                    pass
            dest_display = str(getattr(env, 'destinations', {}).get(i, '-'))
            maintain = bool(getattr(env, 'maintaining', {}).get(i, False))
            state_str = 'maintain' if maintain else 'waiting'
            speed_val = None
            road_id = None
            lane_pos = None
            last_stop_str = None
            try:
                speed_val = float(traci.vehicle.getSpeed(vid))
            except Exception:
                pass
            try:
                road_id = traci.vehicle.getRoadID(vid)
            except Exception:
                road_id = None
            try:
                lane_pos = float(traci.vehicle.getLanePosition(vid))
            except Exception:
                lane_pos = None
            try:
                stops = traci.vehicle.getStops(vid)
                if stops:
                    last = stops[-1]
                    last_stop_str = f"{getattr(last, 'stoppingPlaceID', None)} (arr={getattr(last, 'arrival', None)})"
            except Exception:
                last_stop_str = None
            try:
                if maintain:
                    state_str = 'maintain'
                elif getattr(env, '_arrived_operable')(i):
                    state_str = 'waiting'
                else:
                    if (speed_val is not None) and (speed_val > 0.1):
                        state_str = 'driving'
                    elif traci.vehicle.isStoppedParking(vid):
                        state_str = 'arrived'
                    else:
                        state_str = 'stopped'
            except Exception:
                pass
            statuses.append(TruckStatus(
                truck_id=vid,
                rul=rul_val,
                distance_km=dist_km,
                destination=str(dest_display) if dest_display else "-",
                cargo=cargo_str,
                maintain=maintain,
                state=state_str,
                speed_m_s=speed_val,
                road_id=road_id,
                lane_pos_m=lane_pos,
                last_stop=last_stop_str
            ))
        return statuses

    def sim_loop():
        nonlocal used_initial_reset, initial_obs
        try:
            episodes = all_args.episodes
            for ep in range(episodes):
                if (gui is not None) and gui.stopped():
                    break
                if monitor.ended():
                    break
                print(f"[SUMO Demo] Episode {ep+1}/{episodes}")
                # Use the observations from the initial main-thread reset once to avoid a second reload
                if used_initial_reset:
                    dict_obs = initial_obs if isinstance(initial_obs, dict) else {}
                    used_initial_reset = False
                else:
                    try:
                        dict_obs = env.reset()
                    except Exception as exc:
                        print(f"[SUMO Demo] reset failed: {exc}")
                        break
                monitor.refresh()
                monitor.apply_pending_focus()
                rnn_states.fill(0.0)
                masks.fill(1.0)
                step_cnt = 0
                done_flag = False

                while not done_flag and step_cnt < all_args.max_steps:
                    monitor.apply_pending_focus()
                    if (gui is not None):
                        _statuses = _build_statuses()
                        gui.update_data(_statuses)
                        try:
                            tracker.update_data(_statuses)
                        except Exception:
                            pass
                    if (gui is not None) and gui.stopped():
                        stop_flag["stop"] = True
                        break
                    if monitor.ended():
                        stop_flag["stop"] = True
                        break

                    try:
                        if traci.simulation.getTime() >= all_args.sim_time_limit:
                            if getattr(all_args, 'debug', False):
                                print(f"[SUMO Demo] reached sim_time_limit={all_args.sim_time_limit}; stopping episode")
                            stop_flag["stop"] = True
                            break
                    except Exception:
                        pass

                    if not isinstance(dict_obs, dict):
                        dict_obs = {}

                    operable_ids = sorted(dict_obs.keys())
                    masks[:, 0] = 0.0
                    for aid in operable_ids:
                        masks[aid] = 1.0
                    inactive_ids = [idx for idx in range(all_args.num_agents) if idx not in operable_ids]
                    for idx in inactive_ids:
                        rnn_states[idx] = 0.0

                    act_dict: Dict[int, int] = {}
                    if operable_ids:
                        obs_batch = np.stack([dict_obs[i] for i in operable_ids], axis=0)
                        rnn_batch = rnn_states[operable_ids]
                        mask_batch = masks[operable_ids]
                        if getattr(all_args, 'debug', False):
                            try:
                                print(f"[RL] t={traci.simulation.getTime()} decision for {operable_ids}")
                            except Exception:
                                print(f"[RL] decision for {operable_ids}")
                        with torch.no_grad():
                            obs_t = torch.from_numpy(obs_batch).to(device=device, dtype=torch.float32)
                            rnn_t = torch.from_numpy(rnn_batch).to(device=device, dtype=torch.float32)
                            mask_t = torch.from_numpy(mask_batch).to(device=device, dtype=torch.float32)
                            actions_t, _, new_rnn_t = policy.actor(obs_t, rnn_t, mask_t, available_actions=None, deterministic=True)
                        new_rnn_np = new_rnn_t.detach().cpu().numpy()
                        actions_np = actions_t.detach().cpu().numpy().astype(np.int64).reshape(-1)
                        for idx, agent_id in enumerate(operable_ids):
                            rnn_states[agent_id] = new_rnn_np[idx]
                            act_dict[int(agent_id)] = int(actions_np[idx])

                    try:
                        dict_obs, rewards, dones, _ = env.step(act_dict)
                    except Exception as exc:
                        print(f"[SUMO Demo] env.step error: {exc}")
                        stop_flag["stop"] = True
                        break

                    step_cnt += 1

                    if (gui is not None):
                        _statuses = _build_statuses()
                        gui.update_data(_statuses)
                        try:
                            tracker.update_data(_statuses)
                        except Exception:
                            pass

                    if isinstance(dones, np.ndarray):
                        done_flag = bool(np.all(dones))
                    else:
                        done_flag = bool(dones)
                    if (gui is not None) and gui.stopped():
                        stop_flag["stop"] = True
                        break

                print(f"[SUMO Demo] Episode {ep+1} finished after {step_cnt} steps")
                if stop_flag["stop"]:
                    break
        finally:
            monitor.close()

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
