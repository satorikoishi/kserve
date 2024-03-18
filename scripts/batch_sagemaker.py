import subprocess
import boto3
import time
import os
import csv
from datetime import datetime, timedelta
from utils import get_model_basename, get_endpoint_name, us_to_sec
from sagemaker_invoke import serverless_invoke

model_name_list = ["bigscience/bloom-560m", 
                   "bert-base-uncased", "bert-large-uncased", 
                   "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large"]
metric_list = ["ModelLatency", "OverheadLatency"]

cloudwatch = boto3.client('cloudwatch')

def fetch_sagemaker_metric(endpoint_name, metric_name, variant_name='AllTraffic', 
                           start_time=datetime.utcnow() - timedelta(minutes=5), 
                           end_time=datetime.utcnow() + timedelta(minutes=5)):
    count = 0
    while True:
        response = cloudwatch.get_metric_statistics(
            Namespace='AWS/SageMaker',
            MetricName=metric_name,
            Dimensions=[
                {
                    'Name': 'EndpointName',
                    'Value': endpoint_name
                },
                {
                    'Name': 'VariantName',
                    'Value': variant_name  # Assuming you want metrics for all traffic
                }
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=60,  # Aggregation period in seconds
            Statistics=[
                'Minimum', 'Maximum',
            ]
            # Unit='Seconds'
        )
        data_points = response['Datapoints']
        count += 1
        if data_points or count > 10:
            break
        print("Waiting for aws cloudwatch to collect metric...")
        time.sleep(60)
    
    assert len(data_points) == 1, f"Expected exactly one data point, got {len(data_points)}"
    latency_data = data_points[0]
    minimum_latency = latency_data['Minimum']
    maximum_latency = latency_data['Maximum']
    assert minimum_latency == maximum_latency, "Expected one request"
    
    return us_to_sec(maximum_latency)

def result_output(results):
    target_dir = os.path.join(os.path.dirname(__file__), f"../results/sagemaker")
    os.makedirs(target_dir, exist_ok=True)
    csv_filename = os.path.join(target_dir, f"init-summary.csv")
    with open(csv_filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        row = ["Model Name", "E2ELatency"] + metric_list
        csvwriter.writerow(row)
        for model, model_res in results.items():
            csvwriter.writerow([model] + model_res)
        print(f'Successfully written to {csv_filename}')

def main():
    results = {}
    for model_name in model_name_list:        
        # Deploy model
        subprocess.run(f"python3 ./scripts/sagemaker_deployment.py -m {model_name}", shell=True, check=True)
        
        # Invoke Func
        model_name = get_model_basename(model_name)
        results[model_name] = []
        print(f"Invoking {model_name}...")
        e2e_latency = serverless_invoke(model_name)
        results[model_name].append(e2e_latency)
        print(f"E2E latency: {e2e_latency}")

        # Collect ModelLatency metrics
        endpoint_name = get_endpoint_name(model_name)
        for metric in metric_list:
            res = fetch_sagemaker_metric(endpoint_name, metric)
            results[model_name].append(res)
            print(f"{metric}: {res}")

    # Write results
    result_output(results)
        
if __name__ == "__main__":
    main()