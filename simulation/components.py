import random
import numpy as np

DOWNLOAD_FACTOR = 1
COMPUTE_FACTOR = 0.01
COLDSTART_FACTOR = {'base': 1.5, 'baseplus': 0.7, 'opt': 0.1}
STABLE_WINDOW = 60

class Node:
    def __init__(self, node_id, compute_capacity, disk_capacity):
        self.node_id = node_id
        self.compute_capacity = compute_capacity
        self.disk_capacity = disk_capacity
        self.models = {}  # Dictionary to store models and their sizes
        self.compute_load = 0  # Current compute load
        self.queue = [] # Queue of requests
        
    def handle_request(self, model, start_time, runtime):
        if model.model_id in self.models:
            last_accessed = self.models[model.model_id]['last_accessed']
            time_since_last_access = start_time - last_accessed
            cold_start_penalty = model.model_size * runtime.coldstart_factor if time_since_last_access > STABLE_WINDOW or last_accessed == 0 else 0
            download_time = 0
        else:
            self.add_model(model)
            cold_start_penalty = model.model_size * runtime.coldstart_factor  # Assuming cold start on first download
            download_time = model.download_time

        self.models[model.model_id]['last_accessed'] = start_time
        compute_time = model.compute_time
        total_latency = compute_time + download_time + cold_start_penalty
        return total_latency

    def can_host_model(self, model_size):
        """ Check if the node can host a new model given its size """
        total_model_size = sum(info['size'] for info in self.models.values())
        return total_model_size + model_size <= self.disk_capacity

    def add_model(self, model_id, model_size):
        """ Add a model to the node """
        self.models[model_id] = {'size': model_size, 'last_accessed': 0}  # The zero is a placeholder for future use, such as frequency of access

class Model:
    def __init__(self, model_id, model_size):
        self.model_id = model_id
        self.model_size = model_size
        self.compute_time = model_size * COMPUTE_FACTOR
        self.download_time = model_size * DOWNLOAD_FACTOR
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

class Strategy:
    def place_model(self, system, model):
        """ Decide where to place the model based on the strategy logic. """
        raise NotImplementedError

    def select_node_for_request(self, system, request):
        """ Handle a request based on the strategy logic. """
        raise NotImplementedError

    def evict_model(self, system):
        """ Decide which model to evict when necessary. """
        raise NotImplementedError
    
class RandomPlacementStrategy(Strategy):
    def place_model(self, system, model):
        """ Randomly choose a node that can host the model. """
        eligible_nodes = [node for node in system.nodes if node.can_host_model(model.model_size)]
        if not eligible_nodes:
            return None
        chosen_node = random.choice(eligible_nodes)
        chosen_node.add_model(model.model_id, model.model_size)
        return chosen_node

    def select_node_for_request(self, system, request):
        """ Randomly select a node that contains the model for handling the request. """
        nodes_with_model = [node for node in system.nodes if request.model_id in node.models]
        if nodes_with_model:
            chosen_node = random.choice(nodes_with_model)
            # Further logic to handle compute load
            return chosen_node
        return None
    
    # def evict_model(self, system):
    #     return None
    
class WorkloadGenerator:
    def __init__(self, models):
        self.models = models

    def generate_requests(self, num_requests, stress_level=1):
        # Adjust the rate and size of requests based on intensity
        interval = 1.0 / stress_level

        requests = []
        current_time = 0
        for i in range(num_requests):
            time_to_next_request = np.random.exponential(scale=interval)
            current_time += time_to_next_request
            model = random.choice(self.models)
            requests.append(Request(i, model.model_id, current_time))
        return requests
