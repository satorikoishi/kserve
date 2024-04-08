#!/bin/bash

# Usage: ./hey_inference.sh <MODEL_NAME>

MODEL_NAME=$1

# Get the service hostname
SERVICE_HOSTNAME=$(kubectl get inferenceservice ${MODEL_NAME} -o jsonpath='{.status.url}' | cut -d "/" -f 3)

# Send the request
CMD="hey -m POST -T \"application/json\" -host ${SERVICE_HOSTNAME} -D ./yaml/inputs/${MODEL_NAME}-input.json -c 10 http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/${MODEL_NAME}:predict"
echo $CMD
eval $CMD