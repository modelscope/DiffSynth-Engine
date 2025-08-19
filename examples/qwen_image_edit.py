from diffsynth_engine import QwenImagePipeline, QwenImagePipelineConfig, fetch_model
from PIL import Image

if __name__ == "__main__":
    # TODO: upload edit model && replace model path
    image_path = "input_image_path"
    config = QwenImagePipelineConfig.basic_config(
        model_path=fetch_model("MusePublic/Qwen-image-edit", revision="v1", path="transformer/*.safetensors"),
        encoder_path=fetch_model("MusePublic/Qwen-image-edit", revision="v1", path="text_encoder/*.safetensors"),
        vae_path=fetch_model("MusePublic/Qwen-image-edit", revision="v1", path="vae/*.safetensors"),
        parallelism=1,
    )
    config.device = "cuda:1"
    pipe = QwenImagePipeline.from_pretrained(config)

    prompt = "把'通义千问'替换成'muse平台'"
    image = pipe(
        prompt=prompt,
        input_image=Image.open(image_path).resize((1024, 1024)),
        seed=42,
    )
    image.save("image.png")
    del pipe
