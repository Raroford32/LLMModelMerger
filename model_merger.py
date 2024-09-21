import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import gc
import os

class ModelMerger:
    def __init__(self):
        self.model1_name = "mlabonne/Hermes-3-Llama-3.1-70B-lorablated"
        self.model2_name = "kaizen9/DS-replit-3b-ternary-100B-star-27"
        self.merged_model_name = "merged_model"

    def load_model(self, model_name):
        print(f"Loading model: {model_name}")
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(model_name)
        model = load_checkpoint_and_dispatch(model, model_name, device_map="auto", no_split_module_classes=["OPTDecoderLayer"])
        return model

    def merge_models(self):
        print("Starting model merging process...")
        model1 = self.load_model(self.model1_name)
        model2 = self.load_model(self.model2_name)

        print("Merging models...")
        merged_state_dict = {}
        for key in model1.state_dict().keys():
            if key in model2.state_dict():
                merged_state_dict[key] = (model1.state_dict()[key] + model2.state_dict()[key]) / 2
            else:
                merged_state_dict[key] = model1.state_dict()[key]

        for key in model2.state_dict().keys():
            if key not in merged_state_dict:
                merged_state_dict[key] = model2.state_dict()[key]

        print("Creating merged model...")
        merged_model = AutoModelForCausalLM.from_pretrained(self.model1_name, state_dict=merged_state_dict)

        print("Saving merged model...")
        merged_model.save_pretrained(self.merged_model_name)

        # Save tokenizer from model1 (assuming it's compatible with both models)
        tokenizer = AutoTokenizer.from_pretrained(self.model1_name)
        tokenizer.save_pretrained(self.merged_model_name)

        print("Merged model saved successfully.")

        # Clean up to free memory
        del model1, model2, merged_model, merged_state_dict
        gc.collect()
        torch.cuda.empty_cache()

    def get_merged_model_path(self):
        return os.path.abspath(self.merged_model_name)
