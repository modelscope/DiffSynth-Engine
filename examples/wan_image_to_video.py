import torch.multiprocessing as mp
from PIL import Image

from diffsynth_engine.pipelines import WanVideoPipeline, WanModelConfig
from diffsynth_engine.utils.download import fetch_model
from diffsynth_engine.utils.video import save_video


if __name__ == "__main__":
    mp.set_start_method("spawn")
    config = WanModelConfig(
        model_path=fetch_model("muse/wan2.1-i2v-14b-480p-bf16", path="dit.safetensors"),
        t5_path=fetch_model("muse/wan2.1-umt5", path="umt5.safetensors"),
        vae_path=fetch_model("muse/wan2.1-vae", path="vae.safetensors"),
        image_encoder_path=fetch_model(
            "muse/open-clip-xlm-roberta-large-vit-huge-14",
            path="open-clip-xlm-roberta-large-vit-huge-14.safetensors",
        ),
        dit_fsdp=True,
    )
    pipe = WanVideoPipeline.from_pretrained(config, parallelism=4, use_cfg_parallel=True, offload_mode="cpu_offload")

    image = Image.open("i2v_input.jpg").convert("RGB")
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
