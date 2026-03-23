import torch

from diffsynth_engine.pipelines.qwen_image import QwenImagePipeline
from diffsynth_engine.utils.download import fetch_model

if __name__ == "__main__":
    model_path = fetch_model("Qwen/Qwen-Image")
    pipe = QwenImagePipeline.from_pretrained(model_path_or_config=model_path)
    prompt = "A painting of a cat in a zen garden"
    negative_prompt = "ugly, blurry, low quality"
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        true_cfg_scale=4.0,
        width=1328,
        height=1328,
        num_inference_steps=28,
        generator=torch.Generator(device="cpu").manual_seed(42),
    ).images[0]
    image.save("qwen_image_example.png")
