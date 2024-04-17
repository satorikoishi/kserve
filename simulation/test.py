from components import *
import numpy as np
import argparse
import heapq

class System:
    def __init__(self, nodes, models):
        self.nodes = nodes
        self.models = models
        self.current_time = 0
        self.events = []    # Heap for events
        self.event_map = {} # Map for deleting events
        self.latencies = [] # Collect result
        
        for model in self.models:
            self.init_place_model(model)
    
    def init_place_model(self, model):
        eligible_nodes = [node for node in self.nodes if node.can_host_model(model.model_size)]
        if not eligible_nodes:
            return None
        chosen_node = max(eligible_nodes, key=lambda node: node.remaining_disk_capacity())
        chosen_node.add_model(model.model_id, model.model_size)
        print(f"Init Model {model} placed to Node {chosen_node.node_id}")
        return chosen_node
    
    def select_node_for_request(self, req):
        model = self.models[req.model_id]
        # Highest priority: warm container
        nodes_warm = [node for node in self.nodes if node.model_warm(model.model_id)]
        if nodes_warm:
            chosen_node = max(nodes_warm, key=lambda node: node.remaining_compute_capacity())
            print(f"WARM node {chosen_node.node_id} for req {req.request_id}")
            return chosen_node
        nodes_available = [node for node in self.nodes if node.remaining_compute_capacity() >= model.model_size]
        if not nodes_available:
            print(f"No available node for req {req.request_id}")
            return None     # No available nodes, wait for events to make progress
        # Secondary: cold but without download
        nodes_with_model = [node for node in nodes_available if node.model_exist(model.model_id)]
        if nodes_with_model:
            chosen_node = max(nodes_with_model, key=lambda node: node.remaining_compute_capacity())
            print(f"COLD node {chosen_node.node_id} for req {req.request_id}")
            return chosen_node
        # Third: no node holds the model, download model first (possibly need evict)
        eligible_nodes = [node for node in nodes_available if node.can_host_model(model.model_size)]
        if eligible_nodes:
            chosen_node = max(eligible_nodes, key=lambda node: node.remaining_compute_capacity())
            print(f"DOWNLOAD node {chosen_node.node_id} for req {req.request_id}")
            return chosen_node
        # Worse case: Evict until model size can fit in, then download and cold start
        evict_candidate_nodes = [node for node in nodes_available if node.can_host_model_after_evict(model.model_size)]
        if not evict_candidate_nodes:
            print(f"No disk space for req {req.request_id}")
            return None
            # raise Exception("No sufficient disk space for available nodes")   # Must queue
        chosen_node = max(evict_candidate_nodes, key=lambda node: node.remaining_compute_capacity())
        while not chosen_node.can_host_model(model.model_size):
            chosen_node.del_model(chosen_node.get_lru_model())
        print(f"EVICT MODEL node {chosen_node.node_id} for req {req.request_id}")
        return chosen_node
    
    def schedule_event(self, time, callback, *args):
        """ Schedule a new event at the specified time with a callback function. """
        heapq.heappush(self.events, (time, callback, args))
        if time not in self.event_map:
            self.event_map[time] = []
        self.event_map[time].append((callback, args))
    
    def cancel_event(self, time, callback):
        """ Cancel an event. This assumes you can identify the event by time and callback. """
        if time in self.event_map:
            self.event_map[time] = [(cb, args) for (cb, args) in self.event_map[time] if cb != callback]
            if not self.event_map[time]:
                del self.event_map[time]
                # Rebuild the heap without the cancelled event
                self.events = [(t, cb, args) for t, cb, args in heapq.heapify(self.events) if (cb, args) in self.event_map[t]]
                heapq.heapify(self.events)
        else:
            raise Exception("Cancel event not found")
        
    def run_simulation(self, requests):
        request_index = 0
        
        while request_index < len(requests) or self.events:
            if request_index < len(requests) and (not self.events or requests[request_index].start_time < self.events[0][0]):
                # Handle next request if it's time
                req = requests[request_index]
                if self.current_time < req.start_time:
                    self.current_time = req.start_time
                node = self.select_node_for_request(req)
                if not node:
                    # print(f"Request {req.request_id} has no available node, waiting for event")
                    if not self.events:
                        raise Exception("Simulation deadlock: No more events to process but requests cannot be handled")
                    self.handle_event()     # Handle event first
                    continue
                print(f"{self.current_time} Request {req.request_id} with Model {req.model_id} handled by Node {node.node_id}")
                duration = node.handle_request(self.models[req.model_id], self.current_time, self)
                latency = self.current_time + duration - req.start_time
                self.latencies.append(latency)
                print(f"Duration: {duration}, Latency: {latency}")
                request_index += 1
            else:
                self.handle_event()
        
        # for request in requests:
        #     model = self.models[request.model_id]
        #     node = self.strategy.select_node_for_request(self, model.model_id, model.model_size, request.start_time)
        #     print(f"Request {request.request_id} with Model {request.model_id} handled by Node {node.node_id}")
        #     latency = node.handle_request(model, request.start_time)
        #     print(f"Latency: {latency}")
    
    def handle_event(self):
        # Handle next event
        event_time, event_callback, event_args = heapq.heappop(self.events)
        assert event_time >= self.current_time
        self.current_time = event_time
        print(f"Handling event, time {self.current_time}")
        event_callback(*event_args)
    
    def summarize_results(self):
        # Basic statistics
        mean_latency = np.mean(self.latencies)
        median_latency = np.median(self.latencies)
        min_latency = np.min(self.latencies)
        max_latency = np.max(self.latencies)
        std_deviation = np.std(self.latencies)

        print(f"Mean latency: {mean_latency:.2f}")
        print(f"Median latency: {median_latency:.2f}")
        print(f"Minimum latency: {min_latency:.2f}")
        print(f"Maximum latency: {max_latency:.2f}")
        print(f"Standard deviation: {std_deviation:.2f}")
        
        percentiles_to_calculate = [50, 90, 95, 99]
        results = np.percentile(self.latencies, percentiles_to_calculate)
        
        for percentile, value in zip(percentiles_to_calculate, results):
            print(f"{percentile}th percentile latency: {value:.2f} ms")
            
def main():
    print("---------------------------- Start simulation --------------------------")
    num_nodes = 2
    runtimes = [Runtime(name, factor) for name, factor in COLDSTART_FACTOR.items()]

    for runtime in runtimes:
        nodes = [Node(i, compute_capacity=200, disk_capacity=1000) for i in range(num_nodes)]
        models = [Model(i, 200, runtime) for i in range(10)]
        # models = [Model(i, np.random.randint(50, 200), runtime) for i in range(20)]
        print(models)
        generator = WorkloadGenerator(models)
        requests = generator.generate_requests(3, 1)
        print(requests)
        
        system = System(nodes, models)
        system.run_simulation(requests)
        system.summarize_results()

if __name__ == "__main__":
    main()