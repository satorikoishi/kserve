import subprocess
import time
from datetime import datetime, timezone
from kubernetes import client, config
import re
import os
import csv
import yaml
from kserve import KServeClient
from kserve.models.v1beta1_inference_service import V1beta1InferenceService
from kserve.models.v1beta1_inference_service_spec import V1beta1InferenceServiceSpec
from kserve.models.v1beta1_predictor_spec import V1beta1PredictorSpec
from kserve.models.v1beta1_model_spec import V1beta1ModelSpec
from kserve.models.v1beta1_model_format import V1beta1ModelFormat
from kubernetes.client.models import V1ResourceRequirements
from kubernetes.client.exceptions import ApiException

def runtime_suffix(runtime):
    mapping = {"opt": "", "base": "-mar", "baseplus": "-mar-tl"}
    return mapping[runtime]

def create_service(model_name, runtime):
    # Load the Kubernetes configuration
    config.load_kube_config()
    model_seriesname = get_model_seriesname(model_name)
    # Define the InferenceService
    inference_service = V1beta1InferenceService(
        api_version="serving.kserve.io/v1beta1",
        kind="InferenceService",
        metadata=client.V1ObjectMeta(name=model_seriesname, namespace="default"),
        spec=V1beta1InferenceServiceSpec(
            predictor=V1beta1PredictorSpec(
                container_concurrency=1,
                min_replicas=0,
                max_replicas=10,
                model=V1beta1ModelSpec(
                    model_format=V1beta1ModelFormat(
                        name="pytorch"
                    ),
                    storage_uri=f"pvc://task-pv-claim/{model_name}{runtime_suffix(runtime)}",
                    resources=V1ResourceRequirements(
                        limits={"cpu": "2", "memory": "16Gi"},
                        requests={"cpu": "2", "memory": "16Gi"}
                    )
                )
            )
        )
    )
    # Create the InferenceService using the KServeClient
    kserve_client = KServeClient()
    # Avoid conflict
    existing_services = kserve_client.get(namespace="default")
    for svc in existing_services['items']:
        if svc['metadata']['name'] == model_seriesname:
            delete_service(model_name)
            time.sleep(1)
    # Create service
    kserve_client.create(inference_service, namespace="default", watch=True)
    print(f"InferenceService {model_seriesname} created.")
    time.sleep(1)
    
def delete_service(model_name):
    config.load_kube_config()
    model_seriesname = get_model_seriesname(model_name)
    kserve_client = KServeClient()
    kserve_client.delete(model_seriesname, namespace="default")
    wait_for_pods_termination(model_seriesname)

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

def ms_to_sec(ms):
    seconds = ms / 1000
    return round(seconds, 3)

def sec_to_sec(seconds):
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
    elif runtime_config == "baseplus":
        cmd += " --tl"
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
    
def switch_torchserve_config(runtime):
    runtime_config_dir = os.path.join(os.path.dirname(__file__), "../config/runtimes")
    runtime_config_path = os.path.join(runtime_config_dir, "kustomization.yaml")
    storage_config_dir = os.path.join(os.path.dirname(__file__), "../config/configmap")
    storage_config_path = os.path.join(storage_config_dir, "inferenceservice.yaml")

    ######### Runtime Config ############
    with open(runtime_config_path, 'r') as file:
        runtime_config_data = yaml.safe_load(file)

    # Check and update the torchserve image config
    for image in runtime_config_data.get('images', []):
        if image['name'] == 'kserve-torchserve':
            if runtime == 'opt':
                image['newName'] = 'jwkaguya/torchserve-kfs'
                image['newTag'] = '0.4.2'
            elif runtime == 'base':
                image['newName'] = 'pytorch/torchserve-kfs'
                image['newTag'] = '0.8.2'
            else:
                assert False, f"Unexpected runtime {runtime}"

    with open(runtime_config_path, 'w') as file:
        yaml.safe_dump(runtime_config_data, file)
    
    ######### Storage Initializer Config ############
    with open(storage_config_path, 'r') as file:
        storage_config_data = file.read()
    
    # Locate the line containing 'enableDirectPvcVolumeMount' and modify it
    # Reverse the list to find the last occurrence first
    lines_reversed = storage_config_data.split('\n')[::-1]
    for i, line in enumerate(lines_reversed):
        if '"enableDirectPvcVolumeMount"' in line:
            indent = line[:line.index('"enableDirectPvcVolumeMount"')]
            if runtime == 'opt':
                new_line = f'{indent}"enableDirectPvcVolumeMount": true'
            elif runtime == 'base':
                new_line = f'{indent}"enableDirectPvcVolumeMount": false'
            else:
                assert False, f"Unexpected runtime {runtime}"
            lines_reversed[i] = new_line
            break
    modified_content = '\n'.join(lines_reversed[::-1])
    
    with open(storage_config_path, 'w') as file:
        file.write(modified_content)
    
    # Apply config
    print(f"kserve-torchserve configuration switched to: {runtime}.")
    try:
        subprocess.run(["kubectl", "apply", "-k", runtime_config_dir], check=True)
        subprocess.run(["kubectl", "apply", "-k", storage_config_dir], check=True)
        print(f"Successfully applied kustomization from {runtime_config_dir} and {storage_config_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to apply kustomization: {e}")
        
def set_stable_window(new_stable_window='1m'):
    # Load kubeconfig
    config.load_kube_config()

    # Define the namespace and ConfigMap name
    namespace = 'knative-serving'
    configmap_name = 'config-autoscaler'

    # Create an API client for the ConfigMaps
    v1 = client.CoreV1Api()

    try:
        # Fetch the existing ConfigMap
        cm = v1.read_namespaced_config_map(configmap_name, namespace)
        
        # Update the stable-window value
        if 'data' not in cm.data:
            cm.data = {}
        cm.data['stable-window'] = new_stable_window
        
        # Apply the update
        v1.replace_namespaced_config_map(configmap_name, namespace, cm)
        print(f"Updated 'stable-window' in '{configmap_name}' ConfigMap to '{new_stable_window}'.")
    except client.exceptions.ApiException as e:
        print("Failed to update ConfigMap:", e)

def round_floats_in_data(data):
    rounded_data = []
    for item in data:
        if isinstance(item, tuple):
            rounded_item = tuple(round(x, 3) if isinstance(x, float) else x for x in item)
            rounded_data.append(rounded_item)
        else:
            rounded_data.append(item)
    return rounded_data

def analyze_cprofile():
    cprofile_data_path = os.path.join(os.path.dirname(__file__), "../results/load_profile", "container-cprofile-flan-t5-large.log")
    
    methods = ["LoadStateDict", "TorchLoad", "LoadPretrained"]
    with open(cprofile_data_path, 'r') as f:
        data = f.read()
        parts_split_by_splitter_only = data.split("Splitter")
        cleaned_parts_after_simple_split = [part.replace("-", "").strip() for part in parts_split_by_splitter_only]
        for idx, part in enumerate(cleaned_parts_after_simple_split):
            print(f"Processing method: {methods[idx]}")
            data_lines = part.strip().split("\n")
            assert "ncalls  tottime  percall  cumtime  percall" in data_lines[5], f"Missing line {data_lines[5]}"
            data_lines = data_lines[6:]
        
            functions = []

            for line in data_lines:
                # print(line)
                parts = line.split(maxsplit=5)
                ncalls, tottime, totpercall, cumtime, cumpercall, function = parts
                tottime = float(tottime)
                functions.append((function, ncalls, tottime))

            # Sort the functions by tottime in descending order
            sorted_functions = sorted(functions, key=lambda x: x[2], reverse=True)

            # Calculate total time for percentage calculations
            total_time = sum(tottime for _, _, tottime in sorted_functions)

            # Calculate the percentage of time spent
            functions_with_percent = [(func, ncalls, tottime, (tottime / total_time) * 100) for func, ncalls, tottime in sorted_functions]
            
            result = functions_with_percent[:5]
            other = functions_with_percent[5:]
            result.append(('Other', '0', sum(tottime for _, _, tottime, _ in other), sum(percent for _, _, _, percent in other)))
            result.append(('Total', '0', total_time, 100))
            
            result = round_floats_in_data(result)
            
            csv_path = os.path.join(os.path.dirname(__file__), "../results/load_profile", f"{methods[idx].lower()}.csv")
            with open(csv_path, 'w') as f:
                csvwriter = csv.writer(f)
                row = ["Func", "Calls", "Time", "Percent"]
                csvwriter.writerow(row)
                csvwriter.writerows(result)