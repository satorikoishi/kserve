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
        self.profiler = Profiler()
        
        for model in self.models:
            self.init_place_model(model)
    
    def init_place_model(self, model):
        eligible_nodes = [node for node in self.nodes if node.can_host_model(model.model_size)]
        if not eligible_nodes:
            return None
        chosen_node = max(eligible_nodes, key=lambda node: node.remaining_disk_capacity())
        chosen_node.add_model(model.model_id, model.model_size)
        logger.debug(f"Init Model {model} placed to Node {chosen_node.node_id}")
        return chosen_node
    
    def select_node_for_request(self, req):
        model = self.models[req.model_id]
        # Highest priority: warm container
        nodes_warm = [node for node in self.nodes if node.model_warm(model.model_id)]
        if nodes_warm:
            chosen_node = max(nodes_warm, key=lambda node: node.remaining_compute_capacity())
            logger.debug(f"{self.current_time} WARM node {chosen_node.node_id} for req {req.request_id}")
            self.profiler.warm += 1
            return chosen_node
        nodes_available = [node for node in self.nodes if node.remaining_compute_capacity() >= model.model_size]
        logger.debug(f"{self.current_time} Available nodes: {[node.node_id for node in nodes_available]}")
        if not nodes_available:
            logger.debug(f"{self.current_time} No available node for req {req.request_id}")
            return None     # No available nodes, wait for events to make progress
        # Secondary: cold but without download
        nodes_with_model = [node for node in nodes_available if node.model_exist(model.model_id)]
        if nodes_with_model:
            chosen_node = max(nodes_with_model, key=lambda node: node.remaining_compute_capacity())
            logger.debug(f"{self.current_time} COLD node {chosen_node.node_id} for req {req.request_id}")
            self.profiler.cold += 1
            return chosen_node
        # Third: no node holds the model, download model first (possibly need evict)
        eligible_nodes = [node for node in nodes_available if node.can_host_model(model.model_size)]
        if eligible_nodes:
            chosen_node = max(eligible_nodes, key=lambda node: node.remaining_compute_capacity())
            logger.debug(f"{self.current_time} DOWNLOAD node {chosen_node.node_id} for req {req.request_id}")
            self.profiler.download += 1
            return chosen_node
        # Worse case: Evict until model size can fit in, then download and cold start
        evict_candidate_nodes = [node for node in nodes_available if node.can_host_model_after_evict(model.model_size)]
        if not evict_candidate_nodes:
            logger.debug(f"{self.current_time} No disk space for req {req.request_id}")
            return None
            # raise Exception("No sufficient disk space for available nodes")   # Must queue
        chosen_node = max(evict_candidate_nodes, key=lambda node: node.remaining_compute_capacity())
        while not chosen_node.can_host_model(model.model_size):
            chosen_node.del_model(chosen_node.get_lru_model())
        logger.debug(f"{self.current_time} EVICT MODEL node {chosen_node.node_id} for req {req.request_id}")
        self.profiler.evict += 1
        return chosen_node
    
    def schedule_event(self, time, callback, *args):
        """ Schedule a new event at the specified time with a callback function. """
        logger.debug(f"{self.current_time} Schedule event {callback.__name__}, at {time}")
        heapq.heappush(self.events, Event(time, callback, args))
        if time not in self.event_map:
            self.event_map[time] = []
        self.event_map[time].append((callback, args))
    
    def cancel_event(self, time, callback, *input_args):
        """ Cancel an event. This assumes you can identify the event by time, callback and args. """
        logger.debug(f"{self.current_time} Cancel event {callback.__name__}, at {time}")
        if time in self.event_map:
            self.event_map[time].remove((callback, input_args))
            if not self.event_map[time]:
                del self.event_map[time]
            # Rebuild the heap without the cancelled event
            self.events = []
            for time, events in self.event_map.items():
                for cb, args in events:
                    heapq.heappush(self.events, Event(time, cb, args))
        else:
            raise Exception("Cancel event not found")
        
    def run_simulation(self, requests):
        request_index = 0
        
        while request_index < len(requests) or self.events:
            if request_index < len(requests) and (not self.events or requests[request_index].start_time < self.events[0].event_time):
                # Handle next request if it's time
                req = requests[request_index]
                if self.current_time < req.start_time:
                    self.current_time = req.start_time
                node = self.select_node_for_request(req)
                if not node:
                    # logger.debug(f"Request {req.request_id} has no available node, waiting for event")
                    if not self.events:
                        raise Exception("Simulation deadlock: No more events to process but requests cannot be handled")
                    self.handle_event()     # Handle event first
                    continue
                logger.debug(f"{self.current_time} Request {req.request_id} with Model {req.model_id} handled by Node {node.node_id}")
                duration = node.handle_request(self.models[req.model_id], self.current_time, self)
                latency = self.current_time + duration - req.start_time
                self.latencies.append(latency)
                logger.debug(f"{self.current_time} Duration: {duration}, Latency: {latency}")
                request_index += 1
            else:
                self.handle_event()
        
        # for request in requests:
        #     model = self.models[request.model_id]
        #     node = self.strategy.select_node_for_request(self, model.model_id, model.model_size, request.start_time)
        #     logger.debug(f"Request {request.request_id} with Model {request.model_id} handled by Node {node.node_id}")
        #     latency = node.handle_request(model, request.start_time)
        #     logger.debug(f"Latency: {latency}")
    
    def handle_event(self):
        # Handle next event
        event_time, event_callback, event_args = heapq.heappop(self.events)
        assert event_time >= self.current_time, f"{event_time}, {self.current_time}, {event_callback}, {event_args}"
        self.current_time = event_time
        
        self.event_map[event_time].remove((event_callback, event_args))
        if not self.event_map[event_time]:
            del self.event_map[event_time]
            
        logger.debug(f"{self.current_time} Handling event")
        event_callback(*event_args)
    
    def summarize_results(self):
        self.latencies = [float(x) / 1000 for x in self.latencies]
        # Basic statistics
        mean_latency = np.mean(self.latencies)
        median_latency = np.median(self.latencies)
        min_latency = np.min(self.latencies)
        max_latency = np.max(self.latencies)
        std_deviation = np.std(self.latencies)

        logger.info(f"Mean latency: {mean_latency:.2f}")
        logger.info(f"Median latency: {median_latency:.2f}")
        logger.info(f"Minimum latency: {min_latency:.2f}")
        logger.info(f"Maximum latency: {max_latency:.2f}")
        logger.info(f"Standard deviation: {std_deviation:.2f}")
        
        percentiles_to_calculate = [50, 90, 95, 99]
        results = np.percentile(self.latencies, percentiles_to_calculate)
        
        for percentile, value in zip(percentiles_to_calculate, results):
            logger.info(f"{percentile}th percentile latency: {value:.2f} s")
            
        logger.info(self.profiler)
            
def main():
    parser = argparse.ArgumentParser(description="Run a container-based simulation for model request handling.")
    parser.add_argument('-n', '--num_nodes', type=int, default=2, help='Number of nodes in the simulation.')
    parser.add_argument('-m', '--num_models', type=int, default=10, help='Number of models.')
    parser.add_argument('-r', '--num_requests', type=int, default=3, help='Number of requests to generate.')
    parser.add_argument('-i', '--request_interval', type=int, default=1, help='Interval between generated requests.')
    
    args = parser.parse_args()
    logger.info("---------------------------- Start simulation --------------------------")
    num_nodes = args.num_nodes
    num_models = args.num_models
    num_requests = args.num_requests
    request_interval = args.request_interval
    
    runtimes = [Runtime(name, factor) for name, factor in COLDSTART_FACTOR.items()]
    generator = WorkloadGenerator(num_models)
    requests = generator.generate_requests(num_requests, request_interval)
    for r in requests:
        logger.debug(r)

    for runtime in runtimes:
        # Assume 1T disk, 30G compute
        nodes = [Node(i, compute_capacity=300, disk_capacity=10000) for i in range(num_nodes)]
        # Model size ranging from 0.5G to 3G
        models = [Model(i, np.random.randint(5, 30), runtime) for i in range(num_models)]
        logger.debug(models)
        
        system = System(nodes, models)
        system.run_simulation(requests)
        system.summarize_results()

if __name__ == "__main__":
    main()