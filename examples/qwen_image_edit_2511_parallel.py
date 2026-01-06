from diffsynth_engine import QwenImagePipeline, QwenImagePipelineConfig, fetch_model
from PIL import Image
import time

if __name__ == "__main__":
    # Configure pipeline with use_zero_cond_t=True for 2511 edit model
    config = QwenImagePipelineConfig.basic_config(
        model_path=fetch_model("Qwen/Qwen-Image-Edit-2511", revision="master", path="transformer/*.safetensors"),
        encoder_path=fetch_model("Qwen/Qwen-Image-Edit-2511", revision="master", path="text_encoder/*.safetensors"),
        vae_path=fetch_model("Qwen/Qwen-Image-Edit-2511", revision="master", path="vae/*.safetensors"),
        parallelism=2,
        use_zero_cond_t=True,  # Enable zero_cond_t for 2511 edit model
    )
    config.tp_degree = 1
    config.use_fsdp = False

    pipe = QwenImagePipeline.from_pretrained(config)

    lora_path = fetch_model("lightx2v/Qwen-Image-Edit-2511-Lightning", revision="master", path="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
    pipe.load_lora(lora_path, scale=1.0)

    prompt = "make the clothes to red"
    input_image = Image.open("input/768x1024.png")
    # input_image = Image.open("input/784x1024.png")

    input_image.load()
    input_images = [input_image]

    for i in range(0, 2):
        start = time.perf_counter()
        image = pipe(
            prompt=prompt,
            input_image=input_images,
            seed=42,
            num_inference_steps=8,
            cfg_scale=1.0,
        )
        if i > 0:
            print(f"time: {time.perf_counter() - start:.2f}s")
    image.save("image_edit_2511.png")
    del pipe
