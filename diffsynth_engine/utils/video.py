from typing import List

import imageio
import imageio.v3 as iio
import numpy as np
from moviepy import AudioFileClip, ImageSequenceClip, VideoClip
from PIL import Image


class VideoReader:
    def __init__(self, path: str):
        self.reader = imageio.get_reader(path)

    def __len__(self):
        return self.reader.count_frames()

    def __getitem__(self, item):
        return Image.fromarray(np.array(self.reader.get_data(item))).convert("RGB")

    def __del__(self):
        self.reader.close()

    @property
    def frames(self) -> List[Image.Image]:
        return [self[i] for i in range(len(self))]


def load_video(path: str) -> VideoReader:
    return VideoReader(path)


def save_video(frames, save_path, fps=15):
    if save_path.endswith(".webm"):
        codec = "libvpx-vp9"
    elif save_path.endswith(".mp4"):
        codec = "libx264"

    converted_frames = []
    for img in frames:
        arr = np.array(img)
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        converted_frames.append(arr)

    with iio.imopen(save_path, "w", plugin="FFMPEG") as writer:
        writer.write(converted_frames, fps=fps, codec=codec)


def read_n_frames(
    frames: List[Image.Image], original_fps: int, n_frames: int, target_fps: int = 16
) -> List[Image.Image]:
    num_frames = len(frames)
    interval = max(1, round(original_fps / target_fps))
    sampled_frames: List[Image.Image] = []
    for i in range(n_frames):
        frame_idx = i * interval
        if frame_idx >= num_frames:
            break
        sampled_frames.append(frames[frame_idx])
    return sampled_frames


def save_video_with_audio(frames: List[Image.Image], audio_path: str, target_video_path: str, fps: int = 16):
    # combine all frames
    video = [np.array(frame) for frame in frames]  # shape: t* (b*h, w, c)
    video_clip = ImageSequenceClip(video, fps=fps)
    audio_clip = AudioFileClip(audio_path)
    if audio_clip.duration > video_clip.duration:
        audio_clip: AudioFileClip = audio_clip.subclipped(0, video_clip.duration)  # clip audio
    else:
        video_clip: VideoClip = video_clip.subclipped(0, audio_clip.duration)
    video_with_audio: VideoClip = video_clip.with_audio(audio_clip)
    video_with_audio.write_videofile(target_video_path, codec="libx264")
