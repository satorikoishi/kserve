import pandas as pd
import os
import csv
import subprocess
import json
import aiohttp
import asyncio
import time
from utils import switch_torchserve_config, set_stable_window, get_model_seriesname, sec_to_sec, ms_to_sec, create_service, delete_service

model_list = ['flan-t5-large']
trace_labels = ['Sporadic', 'Bursty', 'Periodic']
runtime_list = ['opt']
# runtime_list = ['base', 'baseplus', 'opt']
stable_window_list = ['1m']
TIME_INTERVAL = 60.0

def generate_request(model_name):
    ingress_host = os.getenv('INGRESS_HOST')
    ingress_port = os.getenv('INGRESS_PORT')
    url = f'http://{ingress_host}:{ingress_port}/v1/models/{model_name}:predict'
    headers = {
        'Host':f'{model_name}.default.example.com',
        'Content-Type':'application/json'
    }
    with open(os.path.join(os.path.dirname(__file__), f"../yaml/inputs/{model_name}-input.json"), 'r') as f:
        payload = json.load(f)
    return url, headers, payload

async def send_request(session, url, headers, payload, timestamp):
    print("Sending request")
    start_time = time.time()
    async with session.post(url, headers=headers, json=payload) as response:
        response_text = await response.text()
        end_time = time.time()
        e2e_lat = end_time - start_time
        status_code = response.status
        print(f"E2E: {e2e_lat}")
        print(f"Status: {status_code}")
        internal_latency = response.headers.get('X-Envoy-Upstream-Service-Time')
        print(f"Internal: {internal_latency}")
        print(response_text)
        # assert status_code == 200, f"Resp Error: {status_code}"
        return timestamp, status_code, sec_to_sec(e2e_lat), ms_to_sec(int(internal_latency))

async def handle_group(tasks, save_file_name):
    # Save results every minute (group)
    results = await asyncio.gather(*tasks)
    print(results)
    
    file_exists = os.path.isfile(save_file_name)
    with open(save_file_name, mode='a') as f:
        writer = csv.writer(f)
        if not file_exists: 
            writer.writerow(['Timestamp', 'StatusCode', 'E2ELatency', 'InternalLatency']) # Column header
        # Write data
        writer.writerows(results)
    print(f'Data has been appended to {save_file_name}')
    
async def run_trace_once(model_name, trace, save_file_name):
    url, headers, payload = generate_request(model_name)
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        groups = []
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
                tasks.append(asyncio.create_task(send_request(session, url, headers, payload, minute+1)))
                await asyncio.sleep(interval)
            groups.append(asyncio.create_task(handle_group(tasks, save_file_name)))
            
        await asyncio.gather(*groups)

async def run_trace():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), f"../results/trace/chosen_data.csv")).to_numpy()
    df = df[:, 4:]
    for runtime in runtime_list:
        switch_torchserve_config(runtime)
        for stable_window in stable_window_list:
            set_stable_window(stable_window)
            for model_name in model_list:
                create_service(model_name, runtime) # TODO: should i wait until scale to zero?
                model_name = get_model_seriesname(model_name)
                for i, label in enumerate(trace_labels):
                    save_file_name = os.path.join(os.path.dirname(__file__), f"../results/trace/{runtime}-{model_name}-{label}-{stable_window}.csv")
                    await run_trace_once(model_name, df[i], save_file_name)
                delete_service(model_name)
                
if __name__ == "__main__":
    asyncio.run(run_trace())