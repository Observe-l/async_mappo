import numpy as np
# Prefer standard traci for stability with GUI
try:
    import traci
except Exception:
    try:
        import libsumo as traci
    except Exception:
        raise ImportError("Neither traci nor libsumo found")
import random

class Truck(object):
    def __init__(self, truck_id:str = 'truck_0', capacity:float = 5.0, weight:float = 0.0,\
                 state:str = 'waiting', product:str = 'P1', eng_time:int = 500, lw:int = 40, maintain_time:int = 6*3600, broken_time:int = 2*24*3600, map_data:dict = None, factory_edge:dict | None = None) -> None:
        self.id = truck_id
        self.capacity = capacity
        # Time delay of loading and unloading
        self.load_time = 600
        # Read data from sumo
        self.map_time = map_data['time']
        self.map_distance = map_data['distance']
        # Mapping from FactoryX -> edgeID for routing
        self.factory_edge = factory_edge
        self.final_factory = ['Factory45', 'Factory46', 'Factory47', 'Factory48', 'Factory49']
        # PredM parameter
        self.maintain_time = maintain_time
        self.broken_time = broken_time
        # Engine update time and slide window size
        self.eng_time = eng_time
        self.lw = lw
        self.reset(weight, state, product)


    def reset(self, weight:float = 0.0, state:str = 'waiting', product:str = 'A') -> None:
        self.weight = weight
        self.state = state
        self.product = product

        self.operable_flag = True
        self.matainance_flag = False
        self.broken_flag = False

        # record total transported product
        self.total_product = 0.0
        self.last_transport = 0.0
        # count time step, waiting time, load or unload time
        self.time_step = 100000
        self.waiting_time = 0
        self.load_time = 0
        # Record the reward
        self.cumulate_reward = 0.0
        # Total transported cumulative goods (capacity units)
        self.total_transported = 0.0

        # reset the driving distance
        self.driving_distance = 0.0
        self.total_distance = 0.0
        # Random select a position after reset the truck
        self.route = random.choice(list(self.map_distance.keys()))
        self.position, self.destination = self.route.split('_to_')
        self.travel_time = self.map_time[self.route]
        self.travel_distance = self.map_distance[self.route]
        # Use driving time to indicate whether it it broken or not
        self.driving_time = 0
        self.eng_vary_time = 0

        # Random select an engineer ID
        self.eng_id = random.randint(1, 100)
        self.eng_state = np.load(f"/home/lwh/Documents/Code/RL-Scheduling/util/cisco_engine_data/engine_{self.eng_id}.npz")['arr_0']
        self.eng_len = self.eng_state.shape[0]
        self.rul = 125
        # The engine state is a time series data, so we need to set the slide window
        self.eng_obs = []
        self.eng_add_obs(self.eng_state[self.driving_time, :])
        
        # Recover time
        self.recover_time = 0
        # Pickup decision needed flag
        self.needs_pickup = False
        '''
        Indicate last state:
        0: Normal
        1: Maintain
        2: Repair
        '''
        self.recover_flag = 0
        # Create truck in sumo. If the truck already exist, remove it first
        try:
            traci.vehicle.add(vehID=self.id, routeID=self.position + '_to_'+ self.position, typeID='truck')
        except:
            traci.vehicle.remove(vehID=self.id)
            traci.vehicle.add(vehID=self.id, routeID=self.position + '_to_'+ self.position, typeID='truck')
        traci.vehicle.setParkingAreaStop(vehID=self.id,stopID=self.position)
    
    def recover(self) -> None:
        '''Truck recover from maintain / maintainance'''
        self.state = 'waiting'
        self.operable_flag = True
        self.matainance_flag = False
        self.broken_flag = False
        # count time step, waiting time, load or unload time
        self.time_step = 100000
        self.waiting_time = 0
        self.load_time = 0
        # Random select a position after reset the truck
        self.route = random.choice(list(self.map_distance.keys()))
        self.position, self.destination = self.route.split('_to_')
        try:
            self.travel_time = self.map_time[self.route]
        except:
            self.travel_time = 0
        self.travel_distance = self.map_distance[self.route]
        # Use driving time to indicate whether it it broken or not
        self.driving_time = 0
        self.eng_vary_time = 0

        # Random select an engineer ID
        self.eng_id = random.randint(1, 100)
        self.eng_state = np.load(f"/home/lwh/Documents/Code/RL-Scheduling/util/cisco_engine_data/engine_{self.eng_id}.npz")['arr_0']
        self.eng_len = self.eng_state.shape[0]
        self.rul = 125
        # The engine state is a time series data, so we need to set the slide window
        self.eng_obs = []
        self.eng_add_obs(self.eng_state[self.driving_time, :])
        
        # Recover time
        self.recover_time = 0
        # Reset pickup flag
        self.needs_pickup = False
        # Note: keep total_transported across recover cycles

    
    def truck_step(self) -> None:
        '''
        Update the status of the truck
        '''
        # Check current location, if the vehicle remove by SUMO, add it first
        try:
            tmp_pk = traci.vehicle.getStops(vehID=self.id)
            if not tmp_pk: raise Exception("No stops found")
            parking_state = tmp_pk[-1]
        except:
            try:
                # print(f'{self.id}, position:{self.position}, destination:{self.destination}, parking: {traci.vehicle.getStops(vehID=self.id)}, state: {self.state}')
                # print(f'weight: {self.weight}, mdp state: {self.mk_state}')
                traci.vehicle.remove(vehID=self.id)
            except:
                # print(f'{self.id} has been deleted')
                # print(f'weight: {self.weight}, mdp state: {self.mk_state}')
                pass
            
            try:
                traci.vehicle.add(vehID=self.id,routeID=self.destination + '_to_'+ self.destination, typeID='truck')
                traci.vehicle.setParkingAreaStop(vehID=self.id,stopID=self.destination)
                traci.vehicle.setColor(typeID=self.id,color=self.color)
                tmp_pk = traci.vehicle.getStops(vehID=self.id)
                parking_state = tmp_pk[-1]
            except Exception as e:
                # print(f"Failed to recover truck {self.id}: {e}")
                return

        self.position = parking_state.stoppingPlaceID

        # Check the truck position in SUMO
        # if parking_state.arrival < 0:
        #     self.state = 'delivery'
        #     self.operable_flag = False
        #     if len(tmp_pk)>1:
        #         self.truck_resume()
        # Update the waiting time of the truck in the waiting state
        if self.state == 'waiting':
            if self.waiting_time > 0:
                self.waiting_time -= 1
            else:
                self.waiting_time = 0
                self.operable_flag = True
        # Check the truck in the loading state load the goods or not
        # If loaded, change the state to waiting
        elif self.state == 'loading':
            self.load_time -= 1
            if self.load_time <= 0 and self.weight > 0:
                self.state = 'waiting'
                self.operable_flag = True
            # Randomly select a destination to get raw materials
            elif self.load_time <= 0 and self.weight == 0:
                # Defer pickup destination to MQTT client
                self.state = 'waiting'
                self.operable_flag = True
                self.needs_pickup = True
        # Check the truck in the delivery state arrive the destination or not
        # If arrived, change the state to waiting (empty) or arrived (loaded)
        elif self.state == 'delivery':
            self.time_step += 1
            if parking_state.arrival >= 0:
                self.position = self.destination
                self.driving_distance = self.travel_distance
                self.total_distance += self.travel_distance
                if self.weight == 0:
                    self.state = 'waiting'
                    self.operable_flag = True
                    if self.position in self.final_factory:
                        # Defer pickup destination to MQTT client
                        self.needs_pickup = True
                else:
                    self.state = 'arrived'
                    self.operable_flag = False
            elif len(tmp_pk)>1:
                self.truck_resume()
            '''Update the engine state'''
            self.driving_time += 1
            if self.driving_time % self.eng_time == 0:
                self.eng_vary_time += 1
                if self.eng_vary_time < self.eng_len:
                    self.eng_add_obs(self.eng_state[self.eng_vary_time, :])
                elif self.eng_vary_time > self.eng_len:
                    self.broken_flag = True
                    self.operable_flag = False
                    self.state = 'repair'
                    self.recover_flag = 2
        # Repair the truck
        elif self.state == 'repair':
            self.recover_time += 1
            if self.recover_time >= self.broken_time:
                self.recover()
        elif self.state == 'maintain':
            self.recover_time += 1
            if self.recover_time >= self.maintain_time:
                self.recover()
        
        # if self.operable_flag and self.weight == 0 and self.position in self.final_factory:
        #     next_destination = f'Factory{random.randint(0,44)}'
        #     self.delivery(next_destination)
    
    def delivery(self, destination:str) -> None:
        '''
        Select a route, change the state, reset time
        '''
        self.destination = destination
        self.route = f'{self.position}_to_{self.destination}'
        self.travel_time = self.map_time[self.route]
        self.travel_distance = self.map_distance[self.route]
        self.time_step = 0
        self.state = 'delivery'
        self.operable_flag = False
        # Clear pickup flag if any
        self.needs_pickup = False

        # SUMO operation
        traci.vehicle.changeTarget(vehID=self.id, edgeID=self.factory_edge[destination])
        # Move out the car parking area
        try:
            traci.vehicle.resume(vehID=self.id)
        except:
            pass
        # Stop at next parking area
        try:
            traci.vehicle.setParkingAreaStop(vehID=self.id, stopID=self.destination)
        except:
            try:
                traci.vehicle.remove(vehID=self.id)
            except:
                pass
            traci.vehicle.add(vehID=self.id,routeID=self.destination + '_to_'+ self.destination, typeID='truck')
            traci.vehicle.setParkingAreaStop(vehID=self.id,stopID=self.destination)
            traci.vehicle.setColor(typeID=self.id,color=self.color)
    
    def load_goods(self, product:str, load_time:float) -> None:
        '''
        Load goods
        '''
        self.product = product
        self.weight = self.capacity
        self.state = 'loading'
        self.load_time = load_time
        self.operable_flag = False
    
    def unload_goods(self, load_time:float) -> None:
        '''
        Unload goods
        '''
        self.weight = 0
        self.state = 'loading'
        self.load_time = load_time
        self.product = None
        self.operable_flag = False
        self.last_transport += self.capacity
        # Add to cumulative transported
        try:
            self.total_transported += self.capacity
        except Exception:
            self.total_transported = self.capacity

    def get_truck_product(self) -> int:
        if self.product is None:
            return -1
        if self.product.startswith('P'):
            return int(self.product[1:])
        else:
            return ord(self.product) - ord('A') + 45
        
    def eng_add_obs(self, obs) -> None:
        '''
        Add the new engine state to the slide window
        '''
        tmp_obs = obs
        if len(tmp_obs.shape) == 1:
            tmp_obs = np.expand_dims(tmp_obs, axis=0)

        if len(self.eng_obs) == self.lw:
            self.eng_obs.pop(0)
            self.eng_obs.append(tmp_obs)
        else:
            self.eng_obs.append(tmp_obs)
    
    def maintain(self) -> None:
        '''
        Maintain the truck
        '''
        self.state = 'maintain'
        self.operable_flag = False
        self.matainance_flag = True
        self.recover_time = 0
        self.recover_flag = 1
    
    def waiting(self, waiting_time:int) -> None:
        '''
        Set the truck to waiting state
        '''
        self.state = 'waiting'
        self.waiting_time = waiting_time
        self.operable_flag = False

    def truck_resume(self) -> None:
        tmp_pk = traci.vehicle.getStops(vehID=self.id)
        if len(tmp_pk) > 0:
            latest_pk = tmp_pk[0]
            if latest_pk.arrival > 0:
                traci.vehicle.resume(vehID=self.id)