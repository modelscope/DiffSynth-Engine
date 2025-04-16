import torch

from diffsynth_engine import fetch_model, SDXLImagePipeline


if __name__ == "__main__":
    model_path = fetch_model("muse/sd_xl_base_1.0", revision="20240425120250", path="sd_xl_base_1.0.safetensors")
    pipe = SDXLImagePipeline.from_pretrained(
        model_path, device="cuda:0", dtype=torch.float16, offload_mode="cpu_offload"
    )
    image = pipe(
        prompt="A cat holding a sign that says hello world", height=1024, width=1024, num_inference_steps=20, seed=42
    )
    image.save("image.png")
