import pandas as pd
import os
import csv
import subprocess
import json
import asyncio
import time
from utils import get_model_seriesname, get_endpoint_name, sec_to_sec
from concurrent.futures import ThreadPoolExecutor
import boto3
from botocore.config import Config

model_list = ['flan-t5-base']   # flan-t5-large fails due to insufficient memory
trace_labels = ['Sporadic', 'Bursty', 'Periodic']
TIME_INTERVAL = 60.0

MAX_CONCURRENCY = 10
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENCY)
custom_config = Config(
    retries={'max_attempts': 0, # disable retry
                'mode': 'standard'
                },
    read_timeout=300,
    connect_timeout=60
)
sm_client = boto3.client('sagemaker-runtime', config=custom_config)

async def send_request(endpoint_name, timestamp, payload_text):
    print("Sending request")
    start_time = time.time()
    
    def invoke_once():
        print(f"Invoking endpoint {endpoint_name}")
        response = sm_client.invoke_endpoint(
            EndpointName=endpoint_name, Body=payload_text.encode(encoding="UTF-8"), ContentType="text/csv"
        )
        print(f"Received resp from endpoint {endpoint_name}")
        return response
    
    response = await asyncio.get_event_loop().run_in_executor(
        executor, invoke_once)
    end_time = time.time()
    e2e_latency = end_time - start_time
    print(f"E2E: {e2e_latency}")

    resp_content = response["Body"].read().decode('utf-8')
    status_code = response['ResponseMetadata']['HTTPStatusCode']
    print(resp_content)
    print(status_code)
    return timestamp, status_code, sec_to_sec(e2e_latency)

async def handle_group(tasks, save_file_name):
    # Save results every minute (group)
    results = await asyncio.gather(*tasks)
    print(results)
    
    file_exists = os.path.isfile(save_file_name)
    with open(save_file_name, mode='a') as f:
        writer = csv.writer(f)
        if not file_exists: 
            writer.writerow(['Timestamp', 'StatusCode', 'E2ELatency']) # Column header
        # Write data
        writer.writerows(results)
    print(f'Data has been appended to {save_file_name}')
    
async def run_trace_once(model_name, trace, save_file_name, endpoint_name):
    start_time = time.time()
    groups = []
    
    # Get payload
    with open(os.path.join(os.path.dirname(__file__), f"../yaml/inputs/{model_name}-input.json"), 'r') as f:
        data = json.load(f)
        payload_text = data["instances"][0]["data"]
        print(payload_text)
    for minute, req_count in enumerate(trace):
        # Fix time
        target_time = start_time + (minute + 1) * TIME_INTERVAL
        current_time = time.time()
        current_loop_time_interval = target_time - current_time
        print(f"Current loop takes: {current_loop_time_interval}")
        assert current_loop_time_interval > 0.5 * TIME_INTERVAL and current_loop_time_interval < 1.5 * TIME_INTERVAL
        
        # Skip minutes with no requests
        if req_count == 0:
            await asyncio.sleep(max(current_loop_time_interval, 0))
            continue  
        
        interval = current_loop_time_interval / req_count  # Calculate interval, allowing for fractions of a second
        print(f"Minute {minute+1}: Sending {req_count} requests, interval = {interval} seconds")
            
        tasks = []
        for _ in range(req_count):
            # Send requests
            tasks.append(asyncio.create_task(send_request(endpoint_name, minute+1, payload_text)))
            await asyncio.sleep(interval)
        groups.append(asyncio.create_task(handle_group(tasks, save_file_name)))
        
    await asyncio.gather(*groups)

async def run_trace():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), f"../results/trace/chosen_data.csv")).to_numpy()
    df = df[:, 4:]
    
    for model_name in model_list:
        endpoint_name = get_endpoint_name(model_name)
        model_name = get_model_seriesname(model_name)
        for i, label in enumerate(trace_labels):
            save_file_name = os.path.join(os.path.dirname(__file__), f"../results/trace/sagemaker-{model_name}-{label}.csv")
            await run_trace_once(model_name, df[i], save_file_name, endpoint_name)
                
if __name__ == "__main__":
    asyncio.run(run_trace())