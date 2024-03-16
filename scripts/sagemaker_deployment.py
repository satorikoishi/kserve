import sagemaker
import boto3
import os
import argparse
import tarfile
from utils import get_model_basename

# Initialize clients
s3 = boto3.client('s3')
region = boto3.Session().region_name
sagemaker_client = boto3.client('sagemaker', region_name=region)
sagemaker_session = sagemaker.Session()

# Variables
model_file_name = 'model.tar.gz'
image_uri = '763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.2.0-cpu-py310-ubuntu20.04-sagemaker'

def get_bucket_name():
    # Retrieve the list of existing buckets
    response = s3.list_buckets()

    # Check if at least one bucket exists and retrieve its name
    if response['Buckets']:
        # Assuming you only have one bucket, get its name
        bucket_name = response['Buckets'][0]['Name']
        print(f"The bucket name is: {bucket_name}")
    else:
        print("No buckets found.")
        
    return bucket_name

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
    save_directory = os.path.join(os.path.dirname(__file__), f"../model_archive/{model_name}")

    with tarfile.open(model_file_name, "w:gz") as tar:
        for filename in os.listdir(save_directory):
            file_path = os.path.join(save_directory, filename)
            if os.path.isfile(file_path):  # Check if it's a file, ignore subdirectories
                tar.add(file_path, arcname=os.path.basename(file_path))
    
def upload_model():
    # Step 1: Upload the model to S3
    bucket_name = get_bucket_name()
    model_path = f'{model_name}/{model_file_name}' # S3 key prefix
    with open(model_file_name, 'rb') as f:
        s3.upload_fileobj(f, bucket_name, model_path)
    os.remove(model_file_name)
    return f's3://{bucket_name}/{model_path}'

def setup_deployment(model_name, model_data):
    # Step 2: Create a PyTorch model in SageMaker
    print("Creating model: ", model_name)
    role = get_role()  # SageMaker execution role ARN

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
    model_name = get_model_basename(args.model_name)
    package_model(model_name)
    
    model_data = upload_model()
    setup_deployment(model_name, model_data)