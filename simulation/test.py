from components import *
import numpy as np

class System:
    def __init__(self, nodes, models, strategy, runtime):
        self.nodes = nodes
        self.models = models
        self.strategy = strategy
        self.runtime = runtime
        
    def run_simulation(self, requests):
        for request in requests:
            model = self.models[request.model_id]
            if not self.strategy.place_model(self, model):
                self.strategy.evict_model(self)
                node_id = self.strategy.place_model(self, model)
                print(f"Model {model} placed to Node {node_id}")
            node = self.strategy.select_node_for_request(self, request)
            print(f"Request {request.request_id} with Model {request.model_id} handled by Node {node.node_id}")
            latency = node.handle_request(model, request.start_time, self.runtime)
            print(f"Latency: {latency}")
            
# Example usage
def main():
    num_nodes = 10
    nodes = [Node(i, compute_capacity=100, disk_capacity=1000) for i in range(num_nodes)]
    models = [Model(i, np.random.randint(50, 200)) for i in range(20)]
    strategy = RandomPlacementStrategy()
    runtimes = [Runtime(name, factor) for name, factor in COLDSTART_FACTOR.items()]
    
    print(models)
        
    generator = WorkloadGenerator(models)
    requests = generator.generate_requests(10, 1)
    print(requests)

    for runtime in runtimes:
        system = System(nodes, models, strategy, runtime)
        system.run_simulation(requests)

if __name__ == "__main__":
    main()