import torch
from PIL import Image

from diffsynth_engine.pipelines.qwen_image import QwenImageLayeredPipeline
from diffsynth_engine.utils.download import fetch_model

if __name__ == "__main__":
    model_path = fetch_model("Qwen/Qwen-Image-Layered")
    pipe = QwenImageLayeredPipeline.from_pretrained(model_path_or_config=model_path)

    input_image = Image.open("examples/input/qwen_image_layered_input.png").convert("RGBA")

    prompt = ""

    output = pipe(
        image=input_image,
        prompt=prompt,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        layers=4,
        resolution=640,
        cfg_normalize=False,
        use_en_prompt=True,
        generator=torch.Generator(device="cpu").manual_seed(42),
    )

    images = output.images[0]
    for i, layer_image in enumerate(images):
        layer_image.save(f"{i}.out.png")
