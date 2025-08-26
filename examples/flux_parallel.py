from diffsynth_engine import fetch_model, FluxImagePipeline, FluxPipelineConfig


if __name__ == "__main__":
    model_path = fetch_model("muse/flux-with-vae", path="flux1-dev-with-vae.safetensors")
    config = FluxPipelineConfig.basic_config(model_path=model_path, parallelism=4)
    pipe = FluxImagePipeline.from_pretrained(config)
    image = pipe(prompt="a cat", seed=42)
    image.save("image.png")

    del pipe
