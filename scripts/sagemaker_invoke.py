import sagemaker
import argparse
import time
import boto3
from utils import get_endpoint_name
from botocore.config import Config

def serverless_invoke(model_name):
    custom_config = Config(
        retries={'max_attempts': 0, # disable retry
                 'mode': 'standard'
                 },
        read_timeout=300,
        connect_timeout=60
    )
    sm = boto3.client('sagemaker-runtime', config=custom_config)
    # sm = sagemaker.Session().sagemaker_runtime_client

    prompt = "The best part of Amazon SageMaker is that it makes machine learning easy."

    endpoint_name = get_endpoint_name(model_name)
    
    start = time.time()
    response = sm.invoke_endpoint(
        EndpointName=endpoint_name, Body=prompt.encode(encoding="UTF-8"), ContentType="text/csv"
    )
    end = time.time()

    resp_content = response["Body"].read()
    print(resp_content)
    return end - start

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invoke Sagemaker functions.")
    parser.add_argument("--model_name", "-m", type=str, required=True, help="The name of the model to download.")
    
    args = parser.parse_args()
    print(serverless_invoke(args.model_name))
    