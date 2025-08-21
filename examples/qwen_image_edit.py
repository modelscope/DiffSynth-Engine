from diffsynth_engine import QwenImagePipeline, QwenImagePipelineConfig, fetch_model
from PIL import Image

if __name__ == "__main__":
    config = QwenImagePipelineConfig.basic_config(
        model_path=fetch_model("Qwen/Qwen-Image-Edit", revision="v1", path="transformer/*.safetensors"),
        encoder_path=fetch_model("Qwen/Qwen-Image-Edit", revision="v1", path="text_encoder/*.safetensors"),
        vae_path=fetch_model("Qwen/Qwen-Image-Edit", revision="v1", path="vae/*.safetensors"),
        parallelism=1,
    )

    pipe = QwenImagePipeline.from_pretrained(config)

    prompt = "把'通义千问'替换成'muse平台'"
    image = pipe(
        prompt=prompt,
        input_image=Image.open("input/qwen_image_edit_inpput.png"),
        seed=42,
    )
    image.save("image.png")
    del pipe
