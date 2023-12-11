from kubernetes import client, config
from datetime import datetime, timezone
import re
import argparse

def find_pod_by_partial_name(namespace, partial_pod_name):
    """
    Find a pod in the specified namespace that matches the partial name pattern.

    Args:
    namespace (str): The namespace in which to search for the pod.
    partial_pod_name (str): A regex pattern that matches the partial name of the pod.

    Returns:
    str: The name of the first pod that matches the pattern, or None if no match is found.
    """
    try:
        config.load_kube_config()
        v1 = client.CoreV1Api()
        pods = v1.list_namespaced_pod(namespace)

        for pod in pods.items:
            if partial_pod_name in pod.metadata.name:
                return pod.metadata.name

        return None
    except client.rest.ApiException as e:
        print(f"Exception when calling CoreV1Api->list_namespaced_pod: {e}")
        return None

def get_container_duration(namespace, pod_name, container_name):
    """
    Get the duration of a container (init or regular) within a pod.

    Args:
    namespace (str): The namespace in which the pod exists.
    pod_name (str): The name of the pod.
    container_name (str): The name of the container.

    Returns:
    str: Duration of the container or a status message.
    """
    try:
        config.load_kube_config()
        v1 = client.CoreV1Api()
        pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)

        all_containers = (pod.status.init_container_statuses or []) + (pod.status.container_statuses or [])

        for container_status in all_containers:
            if container_name in container_status.name:
                print(f"Found container: {container_status.name}")
                if container_status.state.terminated:
                    start_time = container_status.state.terminated.started_at
                    end_time = container_status.state.terminated.finished_at
                    duration = end_time - start_time
                    return f"Duration: {duration.total_seconds():.6f} seconds"
                elif container_status.state.running:
                    start_time = container_status.state.running.started_at
                    now = datetime.now(timezone.utc)
                    duration = now - start_time
                    return f"Duration: {duration.total_seconds():.6f} seconds (still running)"
                else:
                    return "Container is not in a terminated or running state."
        return "Container not found."
    except client.rest.ApiException as e:
        print(f"Exception when calling CoreV1Api->read_namespaced_pod: {e}")
        return None

def get_pod_logs(namespace, pod_name, container_name="kserve-container", line_limit=100):
    """
    Get logs from a specified pod and container in the given namespace.

    Args:
    namespace (str): The namespace in which the pod exists.
    pod_name (str): The name of the pod.
    container_name (str, optional): The name of the container. Defaults to None.

    Returns:
    str: The logs from the specified pod and container.
    """
    try:
        config.load_kube_config()

        v1 = client.CoreV1Api()
        log = v1.read_namespaced_pod_log(name=pod_name, namespace=namespace, container=container_name)
        log_lines = log.splitlines()

        return log_lines[:line_limit]
    except client.rest.ApiException as e:
        print(f"Exception when calling CoreV1Api->read_namespaced_pod_log: {e}")
        return None

def parse_log_timestamp(log_timestamp):
    """
    Parse a log timestamp into a datetime object.

    Args:
    log_timestamp (str): Timestamp string from a log line.

    Returns:
    datetime: A datetime object representing the timestamp.
    """
    for fmt in ("%Y-%m-%dT%H:%M:%S,%f", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(log_timestamp, fmt)
        except ValueError:
            continue
    return None

def parse_log_line(log_line, required_substrings_sequence, index):
    """
    Parse a log line to extract the timestamp and the event/message.

    Args:
    log_line (str): A single line of log.

    Returns:
    tuple: A tuple containing the timestamp and the event/message.
    """
    # Regex pattern to match the timestamp and the event/message
    pattern = r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:,\d{3}|\.\d{3})) (.*)'
    match = re.match(pattern, log_line)

    if match:
        timestamp_str = match.group(1)
        event = match.group(2)
        timestamp = parse_log_timestamp(timestamp_str)
        
        if all(substring in event for substring in required_substrings_sequence[index]):
            return timestamp, event
    return None, None

# Set up argument parser
parser = argparse.ArgumentParser(description="Get the duration of a container in a Kubernetes pod")
parser.add_argument("-n", "--namespace", default="default", help="The namespace of the pod")
parser.add_argument("-p", "--pod_name", help="The partial pod name pattern (regex)")
parser.add_argument("-c", "--container_name", help="The container name")

# Parse arguments
args = parser.parse_args()

pod_name = find_pod_by_partial_name(args.namespace, args.pod_name)
if not pod_name:
    print("No pod found matching the pattern.")
    exit(1)

print(f"Found pod: {pod_name}")
duration = get_container_duration(args.namespace, pod_name, args.container_name)
if duration:
    print(duration)
logs = get_pod_logs(args.namespace, pod_name)
if not logs:
    print("No logs available")
    exit(1)
    
required_substrings_sequence=[["When deploying to production, make sure to limit the set of"],
                              ["main org.pytorch.serve.snapshot.SnapshotManager", "Started restoring models from snapshot"],
                              ["Started server process"],
                              ["[INFO ] main org.pytorch.serve.wlm.ModelManager - Model ", "loaded."],
                              ["stdout MODEL_LOG - Torch worker started"],
                              ["MODEL_LOG - Transformer model from path /home/", " loaded successfully"],
                              ["org.pytorch.serve.wlm.WorkerThread - Backend response t"]]

substring_index = 0
previous_timestamp = None
for line in logs:
    # print(f'Line: {line}')
    timestamp, event = parse_log_line(line, required_substrings_sequence, substring_index)
            
    if timestamp and event:
        print(f"Timestamp: {timestamp}, Event: {event}")
        if substring_index > 0:
            duration = timestamp - previous_timestamp
            print(f"Duration from previous key event: {duration}")
        previous_timestamp = timestamp
        substring_index += 1
        if substring_index == len(required_substrings_sequence):
            break
    # else:
    #     print("Log line could not be parsed.")