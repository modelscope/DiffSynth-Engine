import torch
from PIL import Image

from diffsynth_engine.pipelines.qwen_image import QwenImageEditPlusPipeline
from diffsynth_engine.utils.download import fetch_model

if __name__ == "__main__":
    model_path = fetch_model("Qwen/Qwen-Image-Edit-2511")
    pipe = QwenImageEditPlusPipeline.from_pretrained(model_path_or_config=model_path)

    input_images = [
        Image.open("examples/input/qwen_image_edit_input_1.png").convert("RGB"),
        Image.open("examples/input/qwen_image_edit_input_2.png").convert("RGB"),
    ]
    prompt = "根据这图1中女性和图2中的男性，生成一组结婚照，并遵循以下描述：新郎穿着红色的中式马褂，新娘穿着精致的秀禾服，头戴金色凤冠。他们并肩站立在古老的朱红色宫墙前，背景是雕花的木窗。光线明亮柔和，构图对称，氛围喜庆而庄重。"

    image = pipe(
        image=input_images,
        prompt=prompt,
        negative_prompt=" ",
        true_cfg_scale=4.0,
        num_inference_steps=40,
        generator=torch.Generator(device="cpu").manual_seed(42),
    ).images[0]
    image.save("qwen_image_edit_plus_multi_example.png")
