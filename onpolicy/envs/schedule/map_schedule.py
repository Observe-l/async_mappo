import numpy as np
from gymnasium.spaces import Discrete, Box, Dict
from collections import defaultdict
from pathlib import Path
from csv import writer
from onpolicy.envs.schedule.truck import Truck
from onpolicy.envs.schedule.factory import Factory, Producer
import datetime
import random
import string

class async_scheduling(object):
    def __init__(self, args):
        self.truck_num = args.num_agents
        self.factory_num = 50

        self.init_env()
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
            self.action_space[agent_id] = Discrete(45)
        share_obs_space["global_obs"] = Box(low=-1, high=30000, shape=(share_obs_dim,))
        self.share_observation_space = [Dict(share_obs_space) for _ in range(self.truck_num)]
                                            
        self.episode_num = 0
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
        # Create path with current date
        current_date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        self.path = f"/home/lwh/Documents/Code/RL-Scheduling/result/{args.exp_type}/async_mappo/{current_date}/exp_{random_string}"
        Path(self.path).mkdir(parents=True, exist_ok=True)
        
    
    def reset(self, seed=None, options=None):
        '''Reset the environment'''
        self.make_folder()
        self.init_env()
        
        self.episode_num += 1

        obs = self._get_obs()
        self.episode_len = 0
        self.invalid = {truck_id: False for truck_id in range(self.truck_num)}
        self.waiting_tag = {truck_id: False for truck_id in range(self.truck_num)}
        self.info = {'early_termination': False}
        self.done = np.array([False for _ in range(self.truck_num)])
        return obs

    def step(self, action_dict:dict):
        # Set action
        # self.save_debug(self.episode_len, action_dict)
        self._set_action(action_dict)
        # Run the sumo until any of the agents are avaiable
        sumo_flag = True
        step_length = 0
        while sumo_flag:
            self.producer.map_produce_step()
            sumo_flag = not any([tmp_truck.operable_flag for tmp_truck in self.truck_agents])
            step_length += 1
            self.episode_len += 1
        # Get observation, reward. record the result
        obs = self._get_obs()
        rewards = self._get_reward()
        # Save the results
        self.save_results(self.episode_len, step_length, action_dict, rewards)

        # Episode boundary
        if self.episode_len >= 7 * 24 *3600:
            self.done = np.array([True for _ in range(self.truck_num)])

        return obs, rewards, self.done, self.info

    
    def _set_action(self, action_dict:dict):
        '''Set the action for the truck'''
        for agent_id, action in action_dict.items():
            tmp_truck = self.truck_agents[int(agent_id)]
            target_id = f"Factory{action}"
            if target_id == tmp_truck.position:
                self.invalid[agent_id] = True
                self.waiting_tag[agent_id] = True
                # Waiting action: wait for 2 minutes
                tmp_truck.waiting(60*2)
            else:
                tmp_truck.delivery(destination=target_id)

    def _get_obs(self):
        '''Return back a dictionary for operable agents.'''
        observation = {}

        fac_truck_num = defaultdict(int)
        for tmp_truck in self.truck_agents:
            fac_truck_num[tmp_truck.destination] += 1
        warehouse_storage = []
        com_truck_num = []
        for tmp_factory in self.factory.values():
            # Get the storage of material and product
            material_storage = list(tmp_factory.material_num.values())
            product_storage = tmp_factory.product_num
            warehouse_storage += material_storage + [product_storage]
            # Get the number of truck at current factory
            com_truck_num.append(fac_truck_num[tmp_factory.id])
        queue_obs = np.concatenate([warehouse_storage]+[com_truck_num])

        # Generate observation for those operable trucks
        operable_trucks = [tmp_truck for tmp_truck in self.truck_agents if tmp_truck.operable_flag]
        for tmp_truck in operable_trucks:
            # Get the destination of all possiable route
            distance = [value for key, value in tmp_truck.map_distance.items() if key.startswith(tmp_truck.position + '_to_')]
            # Current position
            position = int(tmp_truck.position[7:])
            # The transported product
            product = tmp_truck.get_truck_product()
            agent_id = int(tmp_truck.id.split('_')[1])
            observation[agent_id] = np.concatenate([queue_obs]+[distance]+[[position]]+[[product]])
        return observation
    
    def _get_reward(self):
        '''Get the reward of given agents'''
        rew = np.zeros((self.truck_num,1))

        for tmp_truck in self.truck_agents:
            agent_id = int(tmp_truck.id.split('_')[1])
            if tmp_truck.operable_flag:
                rew[agent_id, 0] = self._single_reward(tmp_truck)
                if self.invalid[agent_id]:
                    # rew[agent_id, 0] = -10
                    self.invalid[agent_id] = False
            tmp_truck.cumulate_reward += rew[agent_id, 0]
        return rew

    def _single_reward(self, agent:Truck):
        # First factor: unit profile per time step
        # agent.total_product is the record of number of total product at last time step
        rew_final_product = 0
        tmp_final_product = 0
        final_factory = ["Factory45", "Factory46", "Factory47", "Factory48", "Factory49"]
        for tmp_factory_id in final_factory:
            tmp_final_product +=  self.factory[tmp_factory_id].total_final_product
        rew_final_product = 10 * (tmp_final_product - agent.total_product)
        # Second factor: driving cost
        gk = 0.00001
        fk = 0.00002
        if agent.last_transport == 0:
            uk = gk
        else:
            uk = gk + fk
        rew_driving = uk * agent.driving_distance

        # Third factor: asset cost
        rew_ass = 0.1

        # Penalty factor
        gamma1 = 0.5
        gamma2 = 0.5
        rq = 1
        tq = agent.time_step
        sq = gamma1 * tq/2000 + gamma2 * (1-rq)
        if sq >= 1:
            sq = 0.99
        psq = 0.1 * np.log((1+sq)/(1-sq))

        # Short-term reward. Arrive right factory
        rew_short = 0.5 * agent.last_transport
        # Reset the short_term reward
        agent.last_transport = 0
        agent.total_product = tmp_final_product
        # Total reward
        rew = rew_final_product + rew_short - rew_driving - rew_ass - psq
        return rew

    def init_env(self):
        '''Generate trucks and factories'''
        map_data = np.load("onpolicy/envs/schedule/50f.npy",allow_pickle=True).item()
        self.truck_agents = [Truck(truck_id=f'truck_{i}', map_data=map_data) for i in range(self.truck_num)]
        self.factory = {f'Factory{i}': Factory(factory_id=f'Factory{i}', product=f'P{i}') for i in range(self.factory_num)}

        '''Generate the final product'''
        final_product = ['A', 'B', 'C', 'D', 'E']
        remaining_materials = [f'P{i}' for i in range(45)]
        transport_idx = {}
        for i, product in enumerate(final_product):
            tmp_factory_id = f'Factory{45 + i}'
            tmp_materials = [remaining_materials.pop() for _ in range(9)]
            tmp_factory = Factory(factory_id=tmp_factory_id, product=product, material=tmp_materials)
            self.factory[tmp_factory_id] = tmp_factory
            for transport_material in tmp_materials:
                transport_idx[transport_material] = tmp_factory_id
        
        self.producer = Producer(self.factory, self.truck_agents, transport_idx)

        '''Init the factory in the begining'''
        for _ in range(100):
            self.producer.produce_step()


    def make_folder(self):
        '''Create folder to save the result'''
        # Create folder
        folder_path = self.path + '/{}/'.format(self.episode_num)
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        # Create file
        self.action_file = folder_path + 'action.csv'
        self.product_file = folder_path + 'product.csv'
        self.agent_file = folder_path + 'result.csv'
        self.distance_file = folder_path + 'distance.csv'
        

        # Create result file
        with open(self.product_file,'w') as f:
            f_csv = writer(f)
            result_list = ['time', 'step_length', 'total', 'A', 'B', 'C', 'D', 'E']
            f_csv.writerow(result_list)
        
        with open(self.agent_file, 'w') as f:
            f_csv = writer(f)
            result_list = ['time', 'step_length']
            for agent in self.truck_agents:
                agent_list = ['action_'+agent.id,'reward_'+agent.id,'cumulate reward_'+agent.id]
                result_list += agent_list
            f_csv.writerow(result_list)

        # Create active truck file
        with open(self.distance_file,'w') as f:
            f_csv = writer(f)
            distance_list = ['time']
            for agent in self.truck_agents:
                agent_list = [f'step_{agent.id}', f'total_{agent.id}']
                distance_list += agent_list
            f_csv.writerow(distance_list)

        # self.debug_folder = folder_path + 'debug/'
        # Path(self.debug_folder).mkdir(parents=True, exist_ok=True)
        # for tmp_truck in self.truck_agents:
        #     debug_file = self.debug_folder + f'{tmp_truck.id}.csv'
        #     with open(debug_file, 'w') as f:
        #         f_csv = writer(f)
        #         result_list = ['time', 'action', 'position', 'destination', 'product', 'state', 'weight']
        #         f_csv.writerow(result_list)

    def early_termination(self):
        '''Make sure all the products are produced'''
        product_A = self.factory['Factory45'].total_final_product
        product_B = self.factory['Factory46'].total_final_product
        product_C = self.factory['Factory47'].total_final_product
        product_D = self.factory['Factory48'].total_final_product
        product_E = self.factory['Factory49'].total_final_product
        if product_A == 0 or product_B == 0 or product_C == 0 or product_D == 0 or product_E == 0:
            return True

    def save_results(self, time, lenth, action_dict,rewards):
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
            for tmp_agent in self.truck_agents:
                agent_id = int(tmp_agent.id.split('_')[1])
                if agent_id in action_dict.keys():
                    tmp_action = action_dict[agent_id]
                    if self.waiting_tag[agent_id]:
                        tmp_action = 'waiting'
                        self.waiting_tag[agent_id] = False
                else:
                    tmp_action = 'NA'

                if rewards[agent_id,0] != 0:
                    tmp_reward = rewards[agent_id,0]
                else:
                    tmp_reward = 'NA'
                
                agent_list += [tmp_action, tmp_reward, tmp_agent.cumulate_reward]
                
            f_csv.writerow(agent_list)
        
        with open(self.distance_file, 'a') as f:
            f_csv = writer(f)
            distance_list = [current_time]
            for tmp_agent in self.truck_agents:
                distance_list += [tmp_agent.driving_distance, tmp_agent.total_distance]
            f_csv.writerow(distance_list)
    
    def save_debug(self, time, action_dict = None):
        current_time = round(time/3600,3)
        for agent_id, action in action_dict.items():
            tmp_truck = self.truck_agents[int(agent_id)]
            debug_file = self.debug_folder + f'{tmp_truck.id}.csv'
            with open(debug_file, 'a') as f:
                f_csv = writer(f)
                debug_list = [current_time, action, tmp_truck.position, tmp_truck.destination, tmp_truck.weight, tmp_truck.state]
                f_csv.writerow(debug_list)