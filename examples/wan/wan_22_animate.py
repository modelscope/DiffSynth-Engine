import torch
from diffusers.utils import export_to_video, load_video
from PIL import Image

from diffsynth_engine.pipelines.wan import WanAnimatePipeline
from diffsynth_engine.utils.download import fetch_model

if __name__ == "__main__":
    model_path = fetch_model("Wan-AI/Wan2.2-Animate-14B-Diffusers")
    pipe = WanAnimatePipeline.from_pretrained(model_path)

    # Load the reference character image
    image = Image.open("examples/input/wan_22_animate_input.png")

    # Load pose and face conditioning videos (preprocessed from a reference video)
    pose_video = load_video("examples/input/wan_22_animate_pose.mp4")
    face_video = load_video("examples/input/wan_22_animate_face.mp4")

    prompt = "People in the video are doing actions."

    # ---- Animate mode ----
    video = pipe(
        image=image,
        pose_video=pose_video,
        face_video=face_video,
        prompt=prompt,
        mode="animate",
        segment_frame_length=77,
        prev_segment_conditioning_frames=1,
        guidance_scale=1.0,
        num_inference_steps=20,
        generator=torch.Generator(device="cpu").manual_seed(42),
    )

    export_to_video(video.frames[0], "animated_output.mp4", fps=30)

    # ---- Replace mode (optional) ----
    # In replace mode, an additional background_video and mask_video are required.
    # background_video: the original video whose character will be replaced.
    # mask_video: grayscale masks indicating the region to replace (white = replace).
    #
    # background_video = load_video("examples/input/wan_22_animate_background.mp4")
    # mask_video = load_video("examples/input/wan_22_animate_mask.mp4")
    #
    # video_replace = pipe(
    #     image=image,
    #     pose_video=pose_video,
    #     face_video=face_video,
    #     background_video=background_video,
    #     mask_video=mask_video,
    #     prompt=prompt,
    #     mode="replace",
    #     segment_frame_length=77,
    #     prev_segment_conditioning_frames=1,
    #     guidance_scale=1.0,
    #     num_inference_steps=20,
    #     generator=torch.Generator(device="cpu").manual_seed(42),
    # )
    #
    # export_to_video(video_replace.frames[0], "animated_output_replace.mp4", fps=30)
