import torch.multiprocessing as mp
from diffsynth_engine import fetch_model, FluxImagePipeline


if __name__ == "__main__":
    mp.set_start_method("spawn")
    model_path = fetch_model("muse/flux-with-vae", path="flux1-dev-with-vae.safetensors")
    pipe = FluxImagePipeline.from_pretrained(model_path, parallelism=4, offload_mode="cpu_offload")
    image = pipe(prompt="a cat", seed=42)
    image.save("image.png")

    del pipe
