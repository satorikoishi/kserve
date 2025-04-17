import numpy as np
import random

INSTANCE_COST = {
    "cpu": {
        "price_per_hr": 0.0416,
        "warm_duration": 300,
    },
    "gpu": {
        "price_per_hr": 0.526,
        "warm_duration": 300,
    }
}

def get_prewarm_cost(instance_type):
    cfg = INSTANCE_COST[instance_type]
    cost_per_second = cfg["price_per_hr"] / 3600
    return cost_per_second * cfg["warm_duration"]

# Data structures for request and server instances:
class Request:
    def __init__(self, id, arrival_time, context_length, priority, deadline):
        self.id = id
        self.arrival_time = arrival_time   # e.g., timestamp
        self.context_length = context_length  # number of tokens
        self.priority = priority           # e.g., "high", "normal", "low"
        self.deadline = deadline           # allowed response time in ms

class Instance:
    def __init__(self, id, state, available_resources, created_time):
        self.id = id
        self.state = state                # "warm" or "cold"
        self.available_resources = available_resources  # e.g., memory, GPU
        self.created_time = created_time  # e.g., timestamp

# Global system state
REQUEST_QUEUE = []          # Queue for incoming requests
ACTIVE_INSTANCES = []       # List of available instances

# Parameters
PRE_WARM_THRESHOLD = 2      # if less than this many warm instances available, pre-warm new ones
PREDICTION_WINDOW = 60      # seconds to look ahead for forecasting load
MAX_WAIT_TIME = 300         # maximum wait time in ms for a high-priority request

def predict_demand():
    """
    Dummy predictor that estimates the request load based on historical data 
    or current request rate. Returns predicted number of requests in PREDICTION_WINDOW.
    """
    # In practice, this would use historical or real-time metrics.
    return len(REQUEST_QUEUE)  # simple proxy: current queue size

def prewarm_instance():
    """
    Launch a new instance and initialize it until it becomes "warm".
    """
    new_instance = Instance(id=f"inst_{len(ACTIVE_INSTANCES)+1}",
                            state="cold",
                            available_resources={"memory": 16, "gpu": True},
                            created_time=current_time())
    ACTIVE_INSTANCES.append(new_instance)
    # Asynchronously trigger pre-warming: load the model, warm up caches, etc.
    warm_up_instance(new_instance)
    return new_instance

def select_warm_instance(required_resources):
    """
    Select a warm instance that matches required resources (and is not overloaded)
    """
    for instance in ACTIVE_INSTANCES:
        if instance.state == "warm" and meets_resources(instance, required_resources):
            return instance
    return None

def schedule_request(request):
    """
    Core scheduling logic for an incoming request.
    """
    # Check if there’s any warm instance available to handle the request immediately.
    required_resources = resource_estimation(request.context_length)
    instance = select_warm_instance(required_resources)

    if instance:
        # Route request to warm instance immediately.
        route_request_to_instance(request, instance)
        print(f"Request {request.id} assigned to warm instance {instance.id}")
    else:
        # If no warm instance is available, assess predicted load.
        predicted_requests = predict_demand()
        if predicted_requests > PRE_WARM_THRESHOLD:
            # Pre-warm new instance(s) to reduce cold start latency.
            instance = prewarm_instance()
            print(f"Pre-warming instance {instance.id} for request {request.id}")
            # In a real scenario, request might be queued until the instance is warm.
            queue_request_until_warm(request, instance, MAX_WAIT_TIME)
        else:
            # If predicted load is low and latency requirements are not strict,
            # allow a cold start.
            instance = start_cold_instance()
            route_request_to_instance(request, instance)
            print(f"Request {request.id} routed to new cold instance {instance.id}")

# Helper functions (stubs for necessary functionality)
def current_time():
    # Returns current time (timestamp)
    import time
    return time.time()

def resource_estimation(context_length):
    """
    Determines resource needs based on context length.
    For instance, longer contexts may need more memory.
    """
    # Simplified example: return a resource profile dict.
    return {"memory": 8 if context_length < 1500 else 16, "gpu": True}

def meets_resources(instance, required_resources):
    """
    Check if an instance can support the required resources.
    """
    # Simplified check
    return instance.available_resources["memory"] >= required_resources["memory"]

def warm_up_instance(instance):
    """
    Placeholder for warming up the instance (e.g., load model into memory).
    """
    # In a real system, this could be asynchronous.
    instance.state = "warm"

def route_request_to_instance(request, instance):
    """
    Placeholder for dispatching the request to the selected instance.
    """
    # This function would trigger the inference task on the instance.
    pass

def queue_request_until_warm(request, instance, max_wait):
    """
    Hold the request until the pre-warmed instance becomes ready, or until
    max_wait time is exceeded.
    """
    # This implementation can block the request or use a callback/notification mechanism.
    pass

def start_cold_instance():
    """
    Start an instance that begins in the cold state and gradually warms up while processing.
    """
    new_instance = Instance(id=f"inst_{len(ACTIVE_INSTANCES)+1}",
                            state="cold",
                            available_resources={"memory": 16, "gpu": True},
                            created_time=current_time())
    ACTIVE_INSTANCES.append(new_instance)
    return new_instance

def predict_invocations(historical_data, window_size=10):
    """
    Predict the number of function invocations in the next time window using a simple moving average.
    
    Parameters:
    - historical_data (list of int): Invocation counts observed in previous time units.
    - window_size (int): The number of the most recent data points to consider for the moving average.
    
    Returns:
    - float: Predicted number of invocations in the next time unit.
    """
    if len(historical_data) < window_size:
        relevant_data = historical_data
    else:
        relevant_data = historical_data[-window_size:]
    
    prediction = np.mean(relevant_data)
    return prediction

def allocate_resources(predicted_invocations, cost_per_instance, budget, capacity_per_instance=10):
    """
    Determine the number of instances to pre-warm based on predicted invocations while observing a cost constraint.
    
    Parameters:
    - predicted_invocations (float): Predicted number of incoming invocations for the next time window.
    - cost_per_instance (float): Cost to pre-warm one instance.
    - budget (float): Total budget available for pre-warming.
    - capacity_per_instance (int): Maximum number of requests one instance can handle in a time window.
    
    Returns:
    - int: Number of instances to pre-warm.
    - float: Total cost incurred for these instances.
    """
    # Calculate how many instances are minimally needed to handle the predicted invocations.
    required_instances = int(np.ceil(predicted_invocations / capacity_per_instance))
    
    # Calculate the total cost if all required instances are pre-warmed.
    total_cost = required_instances * cost_per_instance
    
    # Adjust the number of instances if the cost exceeds the available budget.
    if total_cost > budget:
        max_instances = int(budget // cost_per_instance)
        print(f"Cost constraint: Reducing instances from {required_instances} to {max_instances}")
        required_instances = max_instances
        total_cost = required_instances * cost_per_instance
        
    return required_instances, total_cost

# Example simulation demonstrating prediction and resource allocation.
if __name__ == "__main__":
    # Simulate historical invocation counts for the past 20 time windows (e.g., minutes)
    historical_invocations = [random.randint(5, 20) for _ in range(20)]
    print("Historical invocation data:", historical_invocations)
    
    # Predict the number of invocations for the next time window (using the latest 10 data points)
    predicted = predict_invocations(historical_invocations, window_size=10)
    print(f"Predicted invocations for the next time window: {predicted:.2f}")
    
    # Define cost and capacity constraints
    cost_per_instance = 2.5     # Arbitrary cost per instance (e.g., dollars per minute)
    budget = 15.0               # Total cost budget available
    capacity_per_instance = 10  # Each instance can process up to 10 requests per time window
    
    # Allocate resources based on the prediction and cost constraint
    num_instances, total_cost = allocate_resources(predicted, cost_per_instance, budget, capacity_per_instance)
    print(f"Allocating {num_instances} instance(s) with a total cost of ${total_cost:.2f}")
    
    # Simulate adding a new high-priority request:
    request1 = Request(id="req_1", arrival_time=current_time(), context_length=2048,
                       priority="high", deadline=250)
    REQUEST_QUEUE.append(request1)
    schedule_request(request1)