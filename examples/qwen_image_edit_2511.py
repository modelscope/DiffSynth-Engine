from diffsynth_engine import QwenImagePipeline, QwenImagePipelineConfig, fetch_model
from PIL import Image

if __name__ == "__main__":
    # Configure pipeline with use_zero_cond_t=True for 2511 edit model
    config = QwenImagePipelineConfig.basic_config(
        model_path=fetch_model("Qwen/Qwen-Image-Edit-2511", revision="master", path="transformer/*.safetensors"),
        encoder_path=fetch_model("Qwen/Qwen-Image-Edit-2511", revision="master", path="text_encoder/*.safetensors"),
        vae_path=fetch_model("Qwen/Qwen-Image-Edit-2511", revision="master", path="vae/*.safetensors"),
        parallelism=1,
        use_zero_cond_t=True,  # Enable zero_cond_t for 2511 edit model
    )

    pipe = QwenImagePipeline.from_pretrained(config)

    prompt = "把'通义千问'替换成'muse平台'"
    input_images = [Image.open("input/qwen_image_edit_input.png")]
    image = pipe(
        prompt=prompt,
        input_image=input_images,
        seed=42,
    )
    image.save("image_edit_2511.png")
    del pipe
