import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from io import BytesIO
import base64

class InferlessPythonModel:
    def initialize(self):
        model_id = "stabilityai/stable-diffusion-2-1"
        snapshot_download(repo_id=model_id,allow_patterns=["*.safetensors"])
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id,use_safetensors=True,torch_dtype=torch.float16).to("cuda")
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        
    def infer(self, inputs):
        prompt = inputs["prompt"]
        image = self.pipe(prompt,negative_prompt="low quality",num_inference_steps=9).images[0]
        buff = BytesIO()
        image.save(buff, format="JPEG")
        img_str = base64.b64encode(buff.getvalue()).decode()
        
        return {"generated_image_base64" : img_str }
    
    def finalize(self):
        self.pipe = None
