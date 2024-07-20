import json
import logging
import os
import zipfile
from abc import ABC

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from ts.torch_handler.base_handler import BaseHandler

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


TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


class TransformersSeqClassifierHandler(BaseHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super(TransformersSeqClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """In this initialize function, the BERT model is loaded and
        the Layer Integrated Gradients Algorithm for Captum Explanations
        is initialized here.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        """
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        # Loading the model and tokenizer from checkpoint and config files based on the user's choice of mode
        # further setup config can be added.
        # with zipfile.ZipFile(model_dir + "/model.zip", "r") as zip_ref:
        #     zip_ref.extractall(model_dir + "/model")

        # read configs for the mode, model_name, etc. from setup_config.json
        setup_config_path = os.path.join(model_dir, "setup_config.json")
        if os.path.isfile(setup_config_path):
            with open(setup_config_path) as setup_config_file:
                self.setup_config = json.load(setup_config_file)
        else:
            logger.warning("Missing the setup_config.json file.")

        if self.setup_config["use_torchload"]:
            self.model = torch.load(f"{model_dir}/model.pt")
            self.model.to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, return_tensors="pt"
        )
        self.model.to(self.device)

        self.model.eval()
        logger.info("Transformer model from path %s loaded successfully", model_dir)

        self.initialized = True

    def preprocess(self, requests):
        """Basic text preprocessing, based on the user's chocie of application mode.
        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of Tensor for the size of the word tokens.
        """
        texts = [item['data'] for item in requests]
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        return inputs.to(self.device)

    def inference(self, inputs):
        """Predict the class (or classes) of the received text using the
        serialized transformers checkpoint.
        Args:
            input_batch (list): List of Text Tensors from the pre-process function is passed here
        Returns:
            list : It returns a list of the predicted value for the input text
        """
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=50)
        
        output_text = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return output_text

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        return inference_output
