import time
print("time1: ", time.time())
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
print("time2: ", time.time())
from huggingface_hub import snapshot_download
print("time3: ", time.time())
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
print("time4: ", time.time())
import torch
print("time5: ", time.time())
from io import BytesIO
print("time6: ", time.time())
import base64
print("time7: ", time.time())

class InferlessPythonModel:
    def initialize(self):
        print("time8: ", time.time())
        model_id = "stabilityai/stable-diffusion-2-1"
        snapshot_download(repo_id=model_id,allow_patterns=["*.safetensors"])
        print("time9: ", time.time())
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id,use_safetensors=True,torch_dtype=torch.float16).to("cuda")
        print("time10: ", time.time())
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        print("time11: ", time.time())
        
    def infer(self, inputs):
        print("time12: ", time.time())
        prompt = inputs["prompt"]
        image = self.pipe(prompt,negative_prompt="low quality",num_inference_steps=9).images[0]
        print("time13: ", time.time())
        buff = BytesIO()
        image.save(buff, format="JPEG")
        img_str = base64.b64encode(buff.getvalue()).decode()
        print("time14: ", time.time())
        
        return {"generated_image_base64" : img_str }
    
    def finalize(self):
        self.pipe = None
