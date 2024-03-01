import transformers
import torch
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
from ts.torch_handler.base_handler import BaseHandler
import logging
import os

logger = logging.getLogger(__name__)
logger.propagate = False
# Create a console handler
handler = logging.StreamHandler()
# Create a formatter
formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')# Add the formatter to the handler
handler.setFormatter(formatter)
# Add the handler to the logger
logger.addHandler(handler)
logger.info("Transformers version %s", transformers.__version__)

class T5Handler(BaseHandler):
    """
    The handler class for the T5 transformer model.
    """
    def __init__(self):
        super(T5Handler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        self.model_dir = properties.get("model_dir")
        device_str = "cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() and properties.get("gpu_id") is not None else "cpu"
        logger.info(f"Device: {device_str}")
        self.device = torch.device(device_str)
        logger.info("Worker init settings finished")
        
        setup_config_path = os.path.join(self.model_dir, "setup_config.json")
        if os.path.isfile(setup_config_path):
            with open(setup_config_path) as setup_config_file:
                self.setup_config = json.load(setup_config_file)
        else:
            logger.warning("Missing the setup_config.json file.")
            
        # Load the tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_dir)
        logger.info("Loaded tokenizer from pretrained")
        if self.setup_config["use_torchload"]:
            self.model = torch.load(f"{self.model_dir}/model.pt")
            logger.info("Loaded model from pretrained (from torch load)")
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_dir)
            logger.info("Loaded model from pretrained")
        self.model.to(self.device)
        logger.info(f"Model loaded on device: {device_str}")
        
        self.model.eval()
        logger.info("Transformer model from path %s loaded successfully", self.model_dir)
        self.initialized = True

    def preprocess(self, data):
        """
        Preprocessing input data to the model. This involves tokenizing the text.
        """
        texts = [item['data'] for item in data]
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        return inputs.to(self.device)

    def inference(self, input_data):
        """
        Predict the output from the processed input data.
        """
        with torch.no_grad():
            outputs = self.model.generate(input_data['input_ids'], attention_mask=input_data['attention_mask'])
        return outputs

    def postprocess(self, inference_output):
        """
        Postprocessing the inference output to human-readable text.
        """
        results = [self.tokenizer.decode(output, skip_special_tokens=True) for output in inference_output]
        return results
