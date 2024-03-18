import sagemaker
import argparse
import time
from utils import get_endpoint_name

def serverless_invoke(model_name):
    sm = sagemaker.Session().sagemaker_runtime_client

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
    