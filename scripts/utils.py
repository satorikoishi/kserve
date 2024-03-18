import subprocess
import time
from datetime import datetime, timezone
from kubernetes import client, config
import re

def get_model_basename(model_name):
    """
    Get the basename of the model name, removing any prefix.

    Parameters:
    model_name (str): The original model name.

    Returns:
    str: The basename of the model name.
    """
    return model_name.split("/")[-1] if "/" in model_name else model_name

def get_model_seriesname(model_basename):
    # Adapt to raw model name
    if "/" in model_basename:
        model_basename = model_basename.split("/")[-1]
    # Process based on model basename
    series_parts = model_basename.split('-')
    if len(series_parts) > 2:
        if 't5' in model_basename:
            series_name = '-'.join(series_parts[:-1])
        elif 'bert' in model_basename:
            series_name = series_parts[0]
        else:
            assert False, f"Unknown model {model_basename}, need further check"
    else:
        series_name = series_parts[0]
    return series_name

def get_endpoint_name(model_name):
    return f"{model_name}-endpoint-serverless"

def us_to_sec(us):
    seconds = us / 1000000
    return round(seconds, 3)

def are_pods_terminating(prefix):
    cmd = f"kubectl get pods --field-selector=status.phase=Running -o custom-columns=:metadata.name"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("Error in fetching pod details")
        return False

    pod_names = result.stdout.decode().split()
    # Filter pods with the given prefix and check if any are still running
    return any(pod.startswith(prefix) for pod in pod_names)

def wait_for_pods_termination(prefix, timeout=600):
    print(f"Waiting for pods with prefix {prefix} to terminate...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        if not are_pods_terminating(prefix):
            print(f"All pods with prefix {prefix} are terminated.")
            return True
        time.sleep(10)
        print("Checking again...")

    print(f"Timeout reached. Some pods with prefix {prefix} may still be running.")
    return False

def prepare_deployment(model_name, runtime_config):
    cmd = f"python3 ./scripts/prepare_deployment.py -m {model_name}"
    if runtime_config == "base":
        pass
    elif runtime_config == "opt":
        cmd += " --tl --noarch"
    else:
        assert False, f"Unknown runtime config: {runtime_config}"
    subprocess.run(cmd, shell=True)
    
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

def get_pod_logs(namespace, pod_name, container_name="kserve-container", line_limit=0):
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

        return log_lines if len(log_lines) <= line_limit or line_limit == 0 else log_lines[:line_limit]
    except client.rest.ApiException as e:
        print(f"Exception when calling CoreV1Api->read_namespaced_pod_log: {e}")
        return None