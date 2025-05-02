import numpy as np
import random

class Truck(object):
    def __init__(self, truck_id:str = 'truck_0', capacity:float = 5.0, weight:float = 0.0,\
                 state:str = 'waiting', product:str = 'P1', map_data:dict = None) -> None:
        self.id = truck_id
        self.capacity = capacity
        # Time delay of loading and unloading
        self.load_time = 600
        # Read data from sumo
        self.map_time = map_data['time']
        self.map_distance = map_data['distance']
        self.reset(weight, state, product)


    def reset(self, weight:float = 0.0, state:str = 'waiting', product:str = 'A') -> None:
        self.weight = weight
        self.state = state
        self.product = product

        self.operable_flag = True
        self.schedule_flag = False

        # record total transported product
        self.total_product = 0.0
        self.last_transport = 0.0
        # count time step, waiting time, load or unload time
        self.time_step = 100000
        self.waiting_time = 0
        self.load_time = 0
        # Record the reward
        self.cumulate_reward = 0.0

        # reset the driving distance
        self.driving_distance = 0.0
        self.total_distance = 0.0
        # Random select a position after reset the truck
        self.route = random.choice(list(self.map_distance.keys()))
        self.position, self.destination = self.route.split('_to_')
        self.travel_time = self.map_time[self.route]
        self.travel_distance = self.map_distance[self.route]

    
    def truck_step(self) -> None:
        '''
        Update the status of the truck
        '''
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
            if self.load_time <= 0:
                self.state = 'waiting'
                self.operable_flag = True
        # Check the truck in the delivery state arrive the destination or not
        # If arrived, change the state to waiting (empty) or arrived (loaded)
        elif self.state == 'delivery':
            self.time_step += 1
            if self.time_step >= self.travel_time:
                self.position = self.destination
                self.driving_distance = self.travel_distance
                self.total_distance += self.travel_distance
                if self.weight == 0:
                    self.state = 'waiting'
                    self.operable_flag = True
                else:
                    self.state = 'arrived'
                    self.operable_flag = False
    
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

    def get_truck_product(self) -> int:
        if self.product is None:
            return -1
        if self.product.startswith('P'):
            return int(self.product[1:])
        else:
            return ord(self.product) - ord('A') + 45
    
    def waiting(self, waiting_time:int) -> None:
        '''
        Set the truck to waiting state
        '''
        self.state = 'waiting'
        self.waiting_time = waiting_time
        self.operable_flag = False