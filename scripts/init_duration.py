from kubernetes import client, config
from datetime import datetime, timezone
import re
import argparse
import csv
import os
from utils import find_pod_by_partial_name, parse_log_timestamp, get_pod_logs

CONTAINER_EVENT_COUNT = 3
# base runtime + MAR file, all events included
full_required_substrings_sequence = [["When deploying to production, make sure to limit the set of"],
                                    ["main org.pytorch.serve.snapshot.SnapshotManager", "Started restoring models from snapshot"],
                                    ["[DEBUG] main org.pytorch.serve.wlm.ModelVersionedRefs", "Adding new version"],
                                    ["[INFO ] main org.pytorch.serve.wlm.ModelManager - Model ", "loaded."],
                                    ["org.pytorch.serve.wlm.WorkerThread", "Flushing req.cmd LOAD to backend"],
                                    ["MODEL_LOG - Transformer model from path /home/", " loaded successfully"],
                                    ["org.pytorch.serve.wlm.WorkerThread - Backend response t"]]
full_key_events = ["Storage Init", "Kserve Container Init", 
                  "First Log", 
                  "Load Config", 
                  "Unzip Model Archive",
                  "Setup Model Dependency", 
                  "Start Torch Worker", 
                  "Worker Load Model", 
                  "Service Ready"]
# opt runtime, parallel events included
server_required_substrings_sequence=[["When deploying to production, make sure to limit the set of"],
                                    ["main org.pytorch.serve.snapshot.SnapshotManager", "Started restoring models from snapshot"],
                                    ["[DEBUG] main org.pytorch.serve.wlm.ModelVersionedRefs", "Adding new version"],
                                    ["[INFO ] main org.pytorch.serve.wlm.ModelManager - Model ", "loaded."],
                                    ["org.pytorch.serve.wlm.WorkerThread", "Flushing req.cmd LOAD to backend"],
                                    ["org.pytorch.serve.wlm.WorkerThread - Backend response t"]]
worker_required_substrings_sequence=[["When deploying to production, make sure to limit the set of"],
                                    ["INFO - Transformers version"],
                                    ["Transformer model from path"]]
server_key_events = ["Storage Init", "Kserve Container Init", 
                  "First Log", 
                  "Load Config", 
                  "Unzip Model Archive",
                  "Setup Model Dependency", 
                  "Start Torch Worker", 
                  "Service Ready"]
worker_key_events = ["Worker Init", "Worker Load Model"]

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

def get_container_event_ts(namespace, pod_name):
    try:
        config.load_kube_config()
        v1 = client.CoreV1Api()
        pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)

        all_containers = (pod.status.init_container_statuses or []) + (pod.status.container_statuses or [])
        
        event_ts = []
        for container_status in all_containers:
            if container_status.name == "storage-initializer":
                print(f"Found container: {container_status.name}")
                assert container_status.state.terminated
                start_time = container_status.state.terminated.started_at
                end_time = container_status.state.terminated.finished_at
                event_ts.append([start_time, "Init start"])
                event_ts.append([end_time, "Init finish"])
            if container_status.name == "kserve-container":
                print(f"Found container: {container_status.name}")
                assert container_status.state.running
                start_time = container_status.state.running.started_at
                event_ts.append([start_time, "Kserve start"])
        return event_ts
    except client.rest.ApiException as e:
        print(f"Exception when calling CoreV1Api->read_namespaced_pod: {e}")
        return None

def parse_log_line(log_line, required_substrings):
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
        
        if all(substring in event for substring in required_substrings):
            return timestamp, event
    return None, None

def serial_parse_log_event(logs, required_substrings_sequence=full_required_substrings_sequence):
    log_event_ts = []
    substring_index = 0
    for line in logs:
        # print(f'Line: {line}')
        timestamp, event = parse_log_line(line, required_substrings_sequence[substring_index])
                
        if timestamp and event:
            print(f"Timestamp: {timestamp}, Event: {event}")
            log_event_ts.append([timestamp, event])
            substring_index += 1
            if substring_index == len(required_substrings_sequence):
                break
        # else:
        #     print("Log line could not be parsed.")
    print(f"Log Event TS: {log_event_ts}")
    assert len(log_event_ts) == len(required_substrings_sequence), f"Log event ts len {len(log_event_ts)}, Subsequence len {len(required_substrings_sequence)}"
    return log_event_ts

def parallel_parse_log_event(logs):
    server_log_event_ts = serial_parse_log_event(logs, server_required_substrings_sequence)
    worker_log_event_ts = serial_parse_log_event(logs, worker_required_substrings_sequence)
    return server_log_event_ts, worker_log_event_ts

def summarize_event_ts(gather_event_ts, key_events):
    res_event_ts = []
    for i, ke in enumerate(key_events):
        start = gather_event_ts[i][0].replace(tzinfo=None)
        end = gather_event_ts[i + 1][0].replace(tzinfo=None)
        res_event_ts.append([ke, (end - start).total_seconds()])
    res_event_ts.append(["Total", (gather_event_ts[-1][0].replace(tzinfo=None) - gather_event_ts[0][0].replace(tzinfo=None)).total_seconds()])
    print(res_event_ts)
    return res_event_ts

def search_model_basename(logs):
    for log_line in logs:
        match = re.search(r'Model Store: /mnt/pvc/(.*?)/model-store', log_line)
        if match:
            return match.group(1)
    return None

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Get the duration of a container in a Kubernetes pod")
    parser.add_argument("-n", "--namespace", default="default", help="The namespace of the pod")
    parser.add_argument("-p", "--pod_name", help="The partial pod name pattern (regex)")
    # parser.add_argument("-c", "--container_name", help="The container name")
    parser.add_argument("--resdir", default=".", help="result directory")
    parser.add_argument("--suffix", default="", help="result file suffix")
    parser.add_argument("--pp", action='store_true', help="Parse log in parallel way")

    # Parse arguments
    args = parser.parse_args()

    pod_name = find_pod_by_partial_name(args.namespace, args.pod_name)
    if not pod_name:
        print("No pod found matching the pattern.")
        exit(1)
    print(f"Found pod: {pod_name}")
    
    # Get container events
    container_event_ts = get_container_event_ts(args.namespace, pod_name)
    assert len(container_event_ts) == CONTAINER_EVENT_COUNT
    print(f"Container Event TS: {container_event_ts}")
    
    # Read logs
    logs = get_pod_logs(args.namespace, pod_name)
    if not logs:
        print("No logs available")
        exit(1)
        
    # Parse log and get events
    if args.pp:
        log_event_ts, worker_log_event_ts = parallel_parse_log_event(logs)
        gather_event_ts = container_event_ts + log_event_ts
        res_event_ts = summarize_event_ts(gather_event_ts, server_key_events)
        res_event_ts += summarize_event_ts(worker_log_event_ts, worker_key_events)
    else:
        log_event_ts = serial_parse_log_event(logs)
        gather_event_ts = container_event_ts + log_event_ts
        res_event_ts = summarize_event_ts(gather_event_ts, full_key_events)
    res_event_ts.append(["Total App", (log_event_ts[-1][0].replace(tzinfo=None) - log_event_ts[0][0].replace(tzinfo=None)).total_seconds()])
    
    # Output to file
    model_basename = search_model_basename(logs)
    target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../results/{args.resdir}")
    os.makedirs(target_dir, exist_ok=True)
    csv_filename = os.path.join(target_dir, f"init-{model_basename}{args.suffix}.csv")
    with open(csv_filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Event', 'Duration'])
        csvwriter.writerows(res_event_ts)
        print(f'Successfully written to {csv_filename}')
