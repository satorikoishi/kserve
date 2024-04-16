from components import *
import numpy as np
import argparse

class System:
    def __init__(self, nodes, models, strategy):
        self.nodes = nodes
        self.models = models
        self.strategy = strategy
        
        for model in self.models:
            self.strategy.place_model(self, model)
        
    def run_simulation(self, requests):
        for request in requests:
            model = self.models[request.model_id]
            node = self.strategy.select_node_for_request(self, model.model_id, model.model_size, request.start_time)
            print(f"Request {request.request_id} with Model {request.model_id} handled by Node {node.node_id}")
            latency = node.handle_request(model, request.start_time)
            print(f"Latency: {latency}")
            
def main():
    print("---------------------------- Start simulation --------------------------")
    num_nodes = 10
    strategy = BalancedPlacementStrategy()
    runtimes = [Runtime(name, factor) for name, factor in COLDSTART_FACTOR.items()]

    for runtime in runtimes:
        nodes = [Node(i, compute_capacity=200, disk_capacity=1000) for i in range(num_nodes)]
        models = [Model(i, 100, runtime) for i in range(20)]
        # models = [Model(i, np.random.randint(50, 200), runtime) for i in range(20)]
        print(models)
        generator = WorkloadGenerator(models)
        requests = generator.generate_requests(100, 1)
        print(requests)
        
        system = System(nodes, models, strategy)
        system.run_simulation(requests)

if __name__ == "__main__":
    main()