#!/bin/bash

# Usage: ./send_request.sh <MODEL_NAME>

MODEL_NAME=$1

# Get the service hostname
SERVICE_HOSTNAME=$(kubectl get inferenceservice ${MODEL_NAME} -o jsonpath='{.status.url}' | cut -d "/" -f 3)

# Send the request
CMD="curl -v -H \"Host: ${SERVICE_HOSTNAME}\" -H \"Content-type: application/json\" http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/${MODEL_NAME}:predict -d @./yaml/inputs/${MODEL_NAME}-input.json"
echo $CMD
eval $CMD