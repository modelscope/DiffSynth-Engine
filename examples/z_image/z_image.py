import torch

from diffsynth_engine.pipelines.z_image import ZImagePipeline
from diffsynth_engine.utils.download import fetch_model

if __name__ == "__main__":
    model_path = fetch_model("Tongyi-MAI/Z-Image")
    pipe = ZImagePipeline.from_pretrained(model_path_or_config=model_path)
    prompt = "两名年轻亚裔女性紧密站在一起，背景为朴素的灰色纹理墙面，可能是室内地毯地面。左侧女性留着长卷发，身穿藏青色毛衣，左袖有奶油色褶皱装饰，内搭白色立领衬衫，下身白色裤子；佩戴小巧金色耳钉，双臂交叉于背后。右侧女性留直肩长发，身穿奶油色卫衣，胸前印有“Tun the tables”字样，下方为“New ideas”，搭配白色裤子；佩戴银色小环耳环，双臂交叉于胸前。两人均面带微笑直视镜头。照片，自然光照明，柔和阴影，以藏青、奶油白为主的中性色调，休闲时尚摄影，中等景深，面部和上半身对焦清晰，姿态放松，表情友好，室内环境，地毯地面，纯色背景。"
    negative_prompt = ""
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=1280,
        width=720,
        cfg_normalization=False,
        num_inference_steps=50,
        guidance_scale=4,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]
    image.save("z_image_example.png")
