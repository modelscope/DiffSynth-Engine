import PIL.Image
import torch
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image

from diffsynth_engine.pipelines.wan import WanVACEPipeline
from diffsynth_engine.utils.download import fetch_model


def prepare_video_and_mask(
    first_img: PIL.Image.Image,
    last_img: PIL.Image.Image,
    height: int,
    width: int,
    num_frames: int,
):
    first_img = first_img.resize((width, height))
    last_img = last_img.resize((width, height))
    frames = [first_img]
    frames.extend([PIL.Image.new("RGB", (width, height), (128, 128, 128))] * (num_frames - 2))
    frames.append(last_img)
    mask_black = PIL.Image.new("L", (width, height), 0)
    mask_white = PIL.Image.new("L", (width, height), 255)
    mask = [mask_black, *[mask_white] * (num_frames - 2), mask_black]
    return frames, mask


if __name__ == "__main__":
    model_path = fetch_model("Wan-AI/Wan2.1-VACE-14B-diffusers")
    pipe = WanVACEPipeline.from_pretrained(model_path)

    # Set flow_shift to 5.0 for 720P (use 3.0 for 480P)
    flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)

    # Load the first and last frame images
    first_frame = load_image("examples/input/wan_vace_first_frame.png")
    last_frame = load_image("examples/input/wan_vace_last_frame.png")

    prompt = (
        "CG animation style, a small blue bird takes off from the ground, flapping its wings. "
        "The bird's feathers are delicate, with a unique pattern on its chest. "
        "The background shows a blue sky with white clouds under bright sunshine. "
        "The camera follows the bird upward, capturing its flight and the vastness of the sky "
        "from a close-up, low-angle perspective."
    )
    negative_prompt = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
        "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
        "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
        "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
        "in the background, walking backwards"
    )

    height = 512
    width = 512
    num_frames = 81
    video, mask = prepare_video_and_mask(first_frame, last_frame, height, width, num_frames)

    output = pipe(
        video=video,
        mask=mask,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=30,
        guidance_scale=5.0,
        generator=torch.Generator().manual_seed(42),
    )

    export_to_video(output.frames[0], "wan_vace_output.mp4", fps=16)
