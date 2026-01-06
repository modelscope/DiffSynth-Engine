from PIL import Image

from diffsynth_engine import WanPipelineConfig
from diffsynth_engine.pipelines import WanDMDPipeline
from diffsynth_engine.utils.download import fetch_model
from diffsynth_engine.utils.video import save_video


if __name__ == "__main__":
    config = WanPipelineConfig.basic_config(
        model_path=fetch_model(
            "lightx2v/Wan2.2-Distill-Models",
            path=[
                "wan2.2_i2v_A14b_high_noise_lightx2v_4step_1030.safetensors",
                "wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors",
            ],
        ),
        parallelism=1,
    )
    pipe = WanDMDPipeline.from_pretrained(config)

    image = Image.open("input/wan_i2v_input.jpg").convert("RGB")
    video = pipe(
        prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        input_image=image,
        num_frames=81,
        width=480,
        height=832,
        seed=42,
        denoising_step_list=[1000, 750, 500, 250],
    )
    save_video(video, "wan_dmd_i2v.mp4", fps=pipe.get_default_fps())

    del pipe
