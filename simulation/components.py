import random
import numpy as np
from enum import Enum, auto
import math

DOWNLOAD_FACTOR = 100
COMPUTE_FACTOR = 1
COLDSTART_FACTOR = {'base': 150, 'baseplus': 70, 'opt': 10}
STABLE_WINDOW = 60

class ContainerState(Enum):
    RUNNING = auto()
    IDLE = auto()

class Container:
    def __init__(self, model_id, state, start_time, finish_time):
        self.model_id = model_id
        self.state = state
        self.start_time = start_time
        self.finish_time = finish_time

    def exec_request(self, start_time, finish_time):
        assert start_time < self.finish_time and self.state == ContainerState.IDLE
        self.state = ContainerState.RUNNING
        self.start_time = start_time
        self.finish_time = finish_time
    
    def to_idle(self, current_time):
        assert math.isclose(current_time, self.finish_time) and self.state == ContainerState.RUNNING
        self.state = ContainerState.IDLE
        self.start_time = self.finish_time
        self.finish_time = self.start_time + STABLE_WINDOW
    
    def __repr__(self) -> str:
        return f"Model {self.model_id}, state {self.state}, start {self.start_time}, finish {self.finish_time}"
    
class Node:
    def __init__(self, node_id, compute_capacity, disk_capacity):
        self.node_id = node_id
        self.compute_capacity = compute_capacity
        self.disk_capacity = disk_capacity
        self.models = {}  # Dictionary to store models, sizes and containers
        self.compute_load = 0  # Current compute load
        
    def handle_request(self, model, start_time, hook):
        # Best case, no need to consider capacity or cold start
        if self.model_warm(model.model_id):
            print(f"{start_time} Request hit warm start")
            c = self.get_warm_container(model.model_id)
            # Cancel delete event
            print(f"{start_time} Cancel delete event at {c.finish_time}")
            hook.cancel_event(c.finish_time, self.del_container)
            
            # Update container state -> RUNNING
            total_latency = model.compute_time
            c.exec_request(start_time, start_time + total_latency)
        else:
            # Start new container to handle request
            if model.model_id in self.models:
                print(f"{start_time} Request cold start")
                download_time = 0
            else:
                print(f"{start_time} Request download then cold start")
                self.add_model(model.model_id, model.model_size)
                download_time = model.download_time
            compute_time = model.compute_time
            total_latency = compute_time + download_time + model.coldstart_time
            c = self.start_container(model.model_id, start_time, start_time + total_latency)
        
        # Reschedule events
        to_idle_time = start_time + total_latency
        hook.schedule_event(to_idle_time, self.to_idle_container, c, to_idle_time)
        to_del_time = to_idle_time + STABLE_WINDOW
        hook.schedule_event(to_del_time, self.del_container, c, to_del_time)
        print(f"{start_time} Schedule events at {to_idle_time} and {to_del_time}")
        return total_latency

    def can_host_model(self, model_size):
        """ Check if the node can host a new model given its size """
        return model_size <= self.remaining_disk_capacity()
    
    def can_host_model_after_evict(self, model_size):
        total_model_size = sum(info['size'] for info in self.models.values() if not info['containers'])
        return total_model_size + model_size <= self.disk_capacity
    
    def remaining_compute_capacity(self):
        self.check_compute_load()
        return self.compute_capacity - self.compute_load
    
    def remaining_disk_capacity(self):
        return self.disk_capacity - sum(info['size'] for info in self.models.values())
    
    def model_exist(self, model_id):
        return model_id in self.models.keys()
    
    def model_warm(self, model_id):
        if not self.model_exist(model_id):
            return False
        for c in self.models[model_id]['containers']:
            if c.state == ContainerState.IDLE:
                return True
        return False
    
    def get_warm_container(self, model_id):
        for c in self.models[model_id]['containers']:
            if c.state == ContainerState.IDLE:
                return c
        assert False, 'No warm container found'
    
    def start_container(self, model_id, start_time, finish_time):
        new_container = Container(model_id, ContainerState.RUNNING, start_time, finish_time)
        model = self.models[model_id]
        model["containers"].append(new_container)
        self.compute_load += model['size']
        self.check_compute_load()
        return new_container
    
    def to_idle_container(self, container, future_time):
        print(f"{future_time} Transform container {container} to idle")
        container.to_idle(future_time)
    
    def del_container(self, container, future_time):
        print(f"{future_time} Delete container {container}")
        assert math.isclose(future_time, container.finish_time)
        model = self.models[container.model_id]
        self.compute_load -= model['size']
        model['last_access'] = future_time
        model['containers'].remove(container)
        self.check_compute_load()
        
    def add_model(self, model_id, model_size):
        """ Add a model to the node """
        assert self.can_host_model(model_size)
        self.models[model_id] = {'size': model_size, 'containers': [], 'last_access': 0}
    
    def del_model(self, model_id):
        assert len(self.models[model_id]['containers']) == 0
        del self.models[model_id]
        
    def get_lru_model(self):
        lru_model_id = -1
        min_last_access = None
        for model_id, model in self.models.items():
            if model['containers']:
                continue
            model_last_access = model['last_access']
            if not min_last_access or model_last_access < min_last_access:
                # Init min with first model, or found another min
                lru_model_id = model_id
                min_last_access = model_last_access
        return lru_model_id

    def check_compute_load(self):
        sum = 0
        for info in self.models.values():
            sum += len(info['containers']) * info['size']
        if sum != self.compute_load:
            raise Exception(f"Error checking compute load. {sum}, {self.compute_load}")
        
class Model:
    def __init__(self, model_id, model_size, runtime):
        self.model_id = model_id
        self.model_size = model_size
        self.compute_time = model_size * runtime.compute_factor
        self.download_time = model_size * runtime.download_factor
        self.coldstart_time = model_size * runtime.coldstart_factor
        self.request_count = 0  # Track how often this model is requested
        
    def __repr__(self) -> str:
        return f"Model id: {self.model_id}, size: {self.model_size}"

class Request:
    def __init__(self, request_id, model_id, start_time):
        self.request_id = request_id
        self.model_id = model_id
        self.start_time = start_time
    
    def __repr__(self) -> str:
        return f"Request id: {self.request_id}, model: {self.model_id}, start time: {self.start_time}"

class Runtime:
    def __init__(self, name, coldstart_factor):
        self.download_factor = DOWNLOAD_FACTOR
        self.compute_factor = COMPUTE_FACTOR
        self.name = name
        self.coldstart_factor = coldstart_factor

# class Strategy:
#     def place_model(self, system, model):
#         """ Decide where to place the model based on the strategy logic. """
#         raise NotImplementedError

#     def select_node_for_request(self, system, model_id, model_size, start_time):
#         """ Handle a request based on the strategy logic. """
#         raise NotImplementedError

    # def evict_model(self, system, model_size, start_time):
    #     """ Decide which model to evict when necessary. """
    #     raise NotImplementedError
    
# class RandomPlacementStrategy(Strategy):
#     def place_model(self, system, model):
#         """ Randomly choose a node that can host the model. """
#         eligible_nodes = [node for node in system.nodes if node.can_host_model(model.model_size)]
#         if not eligible_nodes:
#             return None
#         chosen_node = random.choice(eligible_nodes)
#         chosen_node.add_model(model.model_id, model.model_size)
#         print(f"Init Model {model} placed to Node {chosen_node.node_id}")
#         return chosen_node

#     def select_node_for_request(self, system, model_id, model_size, start_time):
#         """ Randomly select a node that contains the model for handling the request. """
#         # Highest priority: warm container
#         nodes_warm = [node for node in system.nodes if node.model_warm(model_id, start_time)]
#         if nodes_warm:
#             chosen_node = random.choice(nodes_warm)
#             return chosen_node
#         # Secondary: cold but without download
#         nodes_with_model = [node for node in system.nodes if node.model_exist(model_id)]
#         # Remove node with insufficient compute capacity
#         nodes_can_run_model = [node for node in nodes_with_model if node.remaining_compute_capacity(start_time) >= model_size]
#         if nodes_can_run_model:
#             chosen_node = random.choice(nodes_can_run_model)
#             return chosen_node
#         # Worse case: download model first (possibly need evict)
#         eligible_nodes = [node for node in system.nodes if node.can_host_model(model_size)]
#         if eligible_nodes:
#             chosen_node = random.choice(eligible_nodes)
#             return chosen_node
#         # Need evict first
#         for node in system.nodes:
#             node.fully_update_containers(start_time)
#         freed_node = self.evict_model(system, model_size)
#         return freed_node
        
#     def evict_model(self, system, model_size, start_time):
#         # Evict until model size can fit in returned node
#         evict_candidate_nodes = [node for node in system.nodes if node.can_host_model_after_evict(model_size)]
#         if not evict_candidate_nodes:
#             raise NotImplementedError   # Must queue
#         chosen_node = random.choice(evict_candidate_nodes)
#         while not chosen_node.can_host_model(model_size):
#             chosen_node.del_model(chosen_node.get_lru_model())
#         return chosen_node

# class BalancedPlacementStrategy():
#     def place_model(self, system, model):
#         eligible_nodes = [node for node in system.nodes if node.can_host_model(model.model_size)]
#         if not eligible_nodes:
#             return None
#         chosen_node = max(eligible_nodes, key=lambda node: node.remaining_disk_capacity())
#         chosen_node.add_model(model.model_id, model.model_size)
#         print(f"Init Model {model} placed to Node {chosen_node.node_id}")
#         return chosen_node

#     def select_node_for_request(self, system, model_id, model_size, start_time):
#         # Highest priority: warm container
#         nodes_warm = [node for node in system.nodes if node.model_warm(model_id, start_time)]
#         if nodes_warm:
#             chosen_node = max(nodes_warm, key=lambda node: node.remaining_compute_capacity(start_time))
#             return chosen_node
#         # Secondary: cold but without download
#         nodes_with_model = [node for node in system.nodes if node.model_exist(model_id)]
#         # Remove node with insufficient compute capacity
#         nodes_can_run_model = [node for node in nodes_with_model if node.remaining_compute_capacity(start_time) >= model_size]
#         if nodes_can_run_model:
#             chosen_node = max(nodes_can_run_model, key=lambda node: node.remaining_compute_capacity(start_time))
#             return chosen_node
#         # No node holds the model: download model first (possibly need evict)
#         eligible_nodes = [node for node in system.nodes if node.can_host_model(model_size)]
#         if eligible_nodes:
#             chosen_node = max(eligible_nodes, key=lambda node: node.remaining_compute_capacity(start_time))
#             return chosen_node
#         # Worse case: Evict until model size can fit in, then download and cold start
#         for node in system.nodes:
#             node.fully_update_containers(start_time)
#         evict_candidate_nodes = [node for node in system.nodes if node.can_host_model_after_evict(model_size)]
#         if not evict_candidate_nodes:
#             raise NotImplementedError   # Must queue
#         chosen_node = max(evict_candidate_nodes, key=lambda node: node.remaining_compute_capacity(start_time))
#         while not chosen_node.can_host_model(model_size):
#             chosen_node.del_model(chosen_node.get_lru_model())
#         return chosen_node       

class WorkloadGenerator:
    def __init__(self, num_models):
        self.num_models = num_models

    def generate_requests(self, num_requests, stress_level=1):
        # Adjust the rate and size of requests based on intensity
        interval = 1.0 / stress_level

        requests = []
        current_time = 0
        for i in range(num_requests):
            time_to_next_request = round(np.random.exponential(scale=interval * 1000))
            current_time += time_to_next_request
            model_id = random.randrange(self.num_models)
            requests.append(Request(i, model_id, current_time))
        return requests
