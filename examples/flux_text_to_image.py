from diffsynth_engine import fetch_model, FluxImagePipeline


if __name__ == "__main__":
    model_path = fetch_model("muse/flux-with-vae", path="flux_with_vae.safetensors")
    pipe = FluxImagePipeline.from_pretrained(model_path, device="cuda:0", offload_mode="cpu_offload")
    image = pipe(prompt="a cat")
    image.save("image.png")
