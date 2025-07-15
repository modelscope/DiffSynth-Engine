from PIL import Image

from diffsynth_engine import WanPipelineConfig
from diffsynth_engine.pipelines import WanVideoPipeline
from diffsynth_engine.utils.download import fetch_model
from diffsynth_engine.utils.video import save_video


if __name__ == "__main__":
    config = WanPipelineConfig.basic_config(
        model_path=fetch_model("MusePublic/wan2.1-i2v-14b-480p-bf16", path="dit.safetensors"),
        image_encoder_path=fetch_model(
            "muse/open-clip-xlm-roberta-large-vit-huge-14",
            path="open-clip-xlm-roberta-large-vit-huge-14.safetensors",
        ),
        parallelism=4,
        offload_mode="cpu_offload",
    )
    pipe = WanVideoPipeline.from_pretrained(config)

    image = Image.open("input/wan_i2v_input.jpg").convert("RGB")
    video = pipe(
        prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        negative_prompt="",
        input_image=image,
        num_frames=81,
        width=480,
        height=480,
        seed=42,
    )
    save_video(video, "wan_i2v.mp4", fps=15)

    del pipe
