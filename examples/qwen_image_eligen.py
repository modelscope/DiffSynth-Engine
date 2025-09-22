from PIL import Image
from diffsynth_engine import (
    fetch_model,
    QwenImagePipeline,
    QwenImagePipelineConfig,
    QwenImageControlNetParams,
    QwenImageControlType,
)


if __name__ == "__main__":
    config = QwenImagePipelineConfig.basic_config(
        model_path=fetch_model("MusePublic/Qwen-image", revision="v1", path="transformer/*.safetensors"),
        encoder_path=fetch_model("MusePublic/Qwen-image", revision="v1", path="text_encoder/*.safetensors"),
        vae_path=fetch_model("MusePublic/Qwen-image", revision="v1", path="vae/*.safetensors"),
    )
    pipe = QwenImagePipeline.from_pretrained(config)
    param = QwenImageControlNetParams(
        control_type=QwenImageControlType.eligen,
        image=None,
        scale=1.0,
        model=fetch_model("DiffSynth-Studio/Qwen-Image-EliGen-V2", path="model.safetensors"),
    )

    prompt = "写实摄影风格, 细节丰富。街头一位漂亮的女孩，穿着衬衫和短裤，手持写有“实体控制”的标牌，背景是繁忙的城市街道，阳光明媚，行人匆匆。"
    negative_prompt = "网格化，规则的网格，模糊, 低分辨率, 低质量, 变形, 畸形, 错误的解剖学, 变形的手, 变形的身体, 变形的脸, 变形的头发, 变形的眼睛, 变形的嘴巴"
    entity_prompts = ["一个漂亮的女孩", "标牌 '实体控制'", "短裤", "衬衫"]
    entity_masks = [Image.open(f"input/qwen_image_eligen/{i}.png").convert("RGB") for i in range(4)]
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        entity_prompts=entity_prompts,
        entity_masks=entity_masks,
        cfg_scale=4.0,
        width=1024,
        height=1024,
        num_inference_steps=40,
        seed=42,
        controlnet_params=param,
    )
    image.save("qwen_image_eligen.png")
    del pipe
