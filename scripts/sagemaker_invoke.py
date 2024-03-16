import sagemaker
import argparse

def serverless_invoke(model_name):
    sm = sagemaker.Session().sagemaker_runtime_client

    prompt = "The best part of Amazon SageMaker is that it makes machine learning easy."

    endpoint_name = model_name + '-endpoint-serverless'
    response = sm.invoke_endpoint(
        EndpointName=endpoint_name, Body=prompt.encode(encoding="UTF-8"), ContentType="text/csv"
    )

    resp_content = response["Body"].read()
    print(resp_content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invoke Sagemaker functions.")
    parser.add_argument("--model_name", "-m", type=str, required=True, help="The name of the model to download.")
    
    args = parser.parse_args()
    serverless_invoke(args.model_name)
    