import torch
from diffusers import StableDiffusion3Pipeline

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

pipe = StableDiffusion3Pipeline.from_pretrained("/path/to/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

transformer_sd3 = pipe.transformer
transformer_sd3.save_pretrained("./sd3_transformer")


generator = torch.Generator("cuda").manual_seed(42)

image = pipe(
    "A capybara holding a sign that reads Hello World",
    num_inference_steps=40,
    guidance_scale=4.5,
    generator=generator
).images[0]
image.save("capybara.png")
