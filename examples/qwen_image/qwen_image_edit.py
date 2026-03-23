import torch
from PIL import Image

from diffsynth_engine.pipelines.qwen_image import QwenImageEditPipeline
from diffsynth_engine.utils.download import fetch_model

if __name__ == "__main__":
    model_path = fetch_model("Qwen/Qwen-Image-Edit")
    pipe = QwenImageEditPipeline.from_pretrained(model_path_or_config=model_path)

    prompt = "Replace '通义千问' with '呜哩AI'"
    input_image = Image.open("examples/input/qwen_image_edit_input.png")

    image = pipe(
        image=input_image,
        prompt=prompt,
        negative_prompt=" ",
        true_cfg_scale=4.0,
        num_inference_steps=50,
        generator=torch.Generator(device="cpu").manual_seed(42),
    ).images[0]
    image.save("qwen_image_edit_example.png")
