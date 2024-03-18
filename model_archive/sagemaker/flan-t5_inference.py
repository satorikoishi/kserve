import os
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration

def model_fn(model_dir):
    """
    Load the model for inference
    """

    model_path = model_dir
    
    # Load t5 tokenizer from disk.
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    # Load t5 model from disk.
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    model_dict = {'model': model, 'tokenizer':tokenizer}
    
    return model_dict

def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """
    
    tokenizer = model['tokenizer']
    t5_model = model['model']
    
    encoded_input = tokenizer(input_data, return_tensors='pt')
    
    return t5_model.generate(input_ids=encoded_input['input_ids'], 
                      attention_mask=encoded_input['attention_mask'])

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