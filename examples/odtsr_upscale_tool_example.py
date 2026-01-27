from PIL import Image
from diffsynth_engine import QwenImagePipelineConfig, QwenImagePipeline, fetch_model
from diffsynth_engine.tools.qwen_image_upscaler_tool import QwenImageUpscalerTool

if __name__ == "__main__":
    config = QwenImagePipelineConfig.basic_config(
        model_path=fetch_model("Qwen/Qwen-Image", revision="master", path="transformer/*.safetensors"),
        encoder_path=fetch_model("Qwen/Qwen-Image", revision="master", path="text_encoder/*.safetensors"),
        vae_path=fetch_model("Qwen/Qwen-Image", revision="master", path="vae/*.safetensors"),
        parallelism=1,
        use_torch_compile=True,
        vae_tile_size=(256, 256),
        vae_tile_stride=(192, 192),
    )
    pipe = QwenImagePipeline.from_pretrained(config)
    tool = QwenImageUpscalerTool(pipe)

    input_image = Image.open("input/qwen_image_edit_input.png")
    scale = 2
    image = tool(
        image=input_image,
        scale=scale,
        fidelity=0.6,
        align_method="wavelet",
    )
    image.save("upscaled_image.png")
