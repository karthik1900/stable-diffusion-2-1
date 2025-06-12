print("time1: ", time.time())
from diffusers import StableDiffusionPipeline
print("time2: ", time.time())
import torch
print("time3: ", time.time())
from io import BytesIO
import base64
import os
print("time4: ", time.time())

class InferlessPythonModel:
    def initialize(self):
        print("time5: ", time.time())
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            use_safetensors=True,
            torch_dtype=torch.float16,
            device_map='balanced'
        )
        print("time6: ", time.time())


    def infer(self, inputs):
        print("time7: ", time.time())
        prompt = inputs["prompt"]
        image = self.pipe(prompt).images[0]
        print("time8: ", time.time())
        buff = BytesIO()
        image.save(buff, format="JPEG")
        img_str = base64.b64encode(buff.getvalue()).decode()
        print("time9: ", time.time())
        return { "generated_image_base64" : img_str }
        
    def finalize(self):
        self.pipe = None
