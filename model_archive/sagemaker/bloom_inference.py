import os
import json
from transformers import AutoTokenizer, AutoModel

def model_fn(model_dir):
    """
    Load the model for inference
    """

    model_path = model_dir
    
    # Load tokenizer from disk.
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load model from disk.
    model = AutoModel.from_pretrained(model_path)

    model_dict = {'model': model, 'tokenizer':tokenizer}
    
    return model_dict

def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """
    
    tokenizer = model['tokenizer']
    auto_model = model['model']
    
    encoded_input = tokenizer(input_data, return_tensors='pt')
    
    return auto_model(**encoded_input)

def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """
    
    if request_content_type == "application/json":
        request = json.loads(request_body)
    else:
        request = request_body

    return request

def output_fn(prediction, response_content_type):
    """
    Serialize and prepare the prediction output
    """
    
    if response_content_type == "application/json":
        response = str(prediction)
    else:
        response = str(prediction)

    return response