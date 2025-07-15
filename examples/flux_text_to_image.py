from diffsynth_engine import fetch_model, FluxImagePipeline, FluxPipelineConfig


if __name__ == "__main__":
    model_path = fetch_model("muse/flux-with-vae", path="flux1-dev-with-vae.safetensors")
    config = FluxPipelineConfig.basic_config(model_path=model_path, device="cuda:0", offload_mode="cpu_offload")
    pipe = FluxImagePipeline.from_pretrained(config)
    image = pipe(prompt="a cat")
    image.save("image.png")
