import sagemaker
import boto3
import os
import argparse
import tarfile
import shutil
from utils import get_model_basename, get_model_seriesname, get_endpoint_name
from transformers import AutoTokenizer, AutoModel
from sagemaker.pytorch import PyTorchModel
from sagemaker.serverless.serverless_inference_config import ServerlessInferenceConfig

# Variables
model_file_name = 'model.tar.gz'
local_model_path = './temp'
image_uri = '763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.2.0-cpu-py310-ubuntu20.04-sagemaker'

def get_role():
    iam = boto3.client('iam')
    substring = 'AmazonSageMaker-ExecutionRole'
    paginator = iam.get_paginator('list_roles')
    
    # Iterate through the roles in pages
    for page in paginator.paginate():
        for role in page['Roles']:
            # Check if the substring is in the role name
            if substring in role['RoleName']:
                arn = role['Arn']
                print(f"Found role: {arn}")
                return arn
    return None

def package_model(model_name):
    if os.path.exists(local_model_path):
        return
    # Step 1: Package, then Upload the model to S3
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    os.makedirs(os.path.join(local_model_path, "code"))
    model.save_pretrained(save_directory=local_model_path)
    tokenizer.save_pretrained(save_directory=local_model_path)
    model_seriesname = get_model_seriesname(model_name)
    inference_script_path = os.path.join(os.path.dirname(__file__), "../model_archive/sagemaker", f"{model_seriesname}_inference.py")
    requirements_path = os.path.join(os.path.dirname(__file__), "../model_archive/sagemaker", f"{model_seriesname}_requirements.txt")
    shutil.copy(inference_script_path, os.path.join(local_model_path, "code/inference_code.py"))
    shutil.copy(requirements_path, os.path.join(local_model_path, "code/requirements.txt"))

    original_dir = os.getcwd()
    try:
        os.chdir(local_model_path)
        with tarfile.open(model_file_name, "w:gz") as tar:
            tar.add(".")
    finally:
        os.chdir(original_dir)

def upload_model(model_name):
    # Initialize clients
    s3 = boto3.client('s3')
    sagemaker_session = sagemaker.Session()
    bucket_name = sagemaker_session.default_bucket()
    with open(f'{local_model_path}/{model_file_name}', 'rb') as f:
        s3.upload_fileobj(f, bucket_name, f'{model_name}/{model_file_name}')
    
def setup_deployment(model_name):
    endpoint_name = get_endpoint_name(model_name)
    
    # Delete existing ones to update
    sagemaker_client = boto3.client('sagemaker')
    model_matches = sagemaker_client.list_models(NameContains=model_name)["Models"]
    if model_matches:
        print(f"Delete found matching model: {model_matches}")
        sagemaker_client.delete_model(ModelName=model_name)
    endpoint_config_matches = sagemaker_client.list_endpoint_configs(NameContains=endpoint_name)["EndpointConfigs"]
    if endpoint_config_matches:
        print(f"Delete found matching ep config: {endpoint_config_matches}")
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
    endpoint_matches = sagemaker_client.list_endpoints(NameContains=endpoint_name)["Endpoints"]
    if endpoint_matches:
        print(f"Delete found matching ep: {endpoint_matches}")
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
    
    # Deploy
    print("Deploying...")
    serverless_config = ServerlessInferenceConfig(memory_size_in_mb=3072, max_concurrency=1)
    model = PyTorchModel(
        entry_point="inference_code.py",
        model_data=f'{local_model_path}/{model_file_name}',
        role=get_role(),
        image_uri=image_uri,
        name=model_name
    )
    predictor = model.deploy(endpoint_name=endpoint_name, serverless_inference_config=serverless_config)
    print(predictor)
    
def setup_deployment_bystep(model_name):
    sagemaker_client = boto3.client('sagemaker')
    sagemaker_session = sagemaker.Session()
    # Step 2: Create a PyTorch model in SageMaker
    print("Creating model: ", model_name)
    role = get_role()  # SageMaker execution role ARN
    bucket_name = sagemaker_session.default_bucket()
    model_data = f's3://{bucket_name}/{model_name}/{model_file_name}'
    
    create_model_response = sagemaker_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            'Image': image_uri,
            'ModelDataUrl': model_data,
        },
        ExecutionRoleArn=role,
    )
    print("Model ARN: ", create_model_response['ModelArn'])

    # Step 3: Create an endpoint configuration
    endpoint_config_name = model_name + '-config-serverless'
    print("Creating endpoint config: ", endpoint_config_name)

    create_endpoint_config_response = sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            'ModelName': model_name,
            'VariantName': 'AllTraffic',
            "ServerlessConfig": {
                "MemorySizeInMB": 2048,
                "MaxConcurrency": 10,
            }
        }]
    )
    print("Endpoint Config ARN: ", create_endpoint_config_response['EndpointConfigArn'])

    # Step 4: Create an endpoint
    endpoint_name = model_name + '-endpoint-serverless'
    print("Creating endpoint: ", endpoint_name)

    create_endpoint_response = sagemaker_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name,
    )
    print("Endpoint ARN: ", create_endpoint_response['EndpointArn'])

    print(f"Endpoint {endpoint_name} is being created. Please wait...")
    waiter = sagemaker_client.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=endpoint_name)

    print(f"Endpoint {endpoint_name} is ready!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload model to S3 and deploy to Sagemaker.")
    parser.add_argument("--model_name", "-m", type=str, required=True, help="The name of the model to download.")
    
    args = parser.parse_args()
    model_name = args.model_name
    package_model(model_name)
    model_name = get_model_basename(model_name)
    setup_deployment(model_name)
    if os.path.exists(local_model_path):
        shutil.rmtree(local_model_path)