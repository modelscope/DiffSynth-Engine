from diffsynth_engine import fetch_model, SDXLImagePipeline, SDXLPipelineConfig


if __name__ == "__main__":
    model_path = fetch_model("muse/sd_xl_base_1.0", revision="20240425120250", path="sd_xl_base_1.0.safetensors")
    config = SDXLPipelineConfig.basic_config(model_path=model_path, device="cuda:0", offload_mode="cpu_offload")
    pipe = SDXLImagePipeline.from_pretrained(config)
    image = pipe(
        prompt="A cat holding a sign that says hello world",
        height=1024,
        width=1024,
        num_inference_steps=20,
        seed=42,
    )
    image.save("image.png")
