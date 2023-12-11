from kubernetes import client, config
from datetime import datetime, timezone
import re
import argparse

def find_pod_by_partial_name(namespace, partial_name_pattern):
    """
    Find a pod in the specified namespace that matches the partial name pattern.

    Args:
    namespace (str): The namespace in which to search for the pod.
    partial_name_pattern (str): A regex pattern that matches the partial name of the pod.

    Returns:
    str: The name of the first pod that matches the pattern, or None if no match is found.
    """
    try:
        config.load_kube_config()
        v1 = client.CoreV1Api()
        pods = v1.list_namespaced_pod(namespace)

        for pod in pods.items:
            if partial_name_pattern in pod.metadata.name:
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

# Set up argument parser
parser = argparse.ArgumentParser(description="Get the duration of a container in a Kubernetes pod")
parser.add_argument("-n", "--namespace", default="default", help="The namespace of the pod")
parser.add_argument("-p", "--pod_name", help="The partial pod name pattern (regex)")
parser.add_argument("-c", "--container_name", help="The container name")

# Parse arguments
args = parser.parse_args()

pod_name = find_pod_by_partial_name(args.namespace, args.pod_name)
if pod_name:
    print(f"Found pod: {pod_name}")
    duration = get_container_duration(args.namespace, pod_name, args.container_name)
    if duration:
        print(duration)
else:
    print("No pod found matching the pattern.")