import imageio
import imageio.v3 as iio
import numpy as np
from PIL import Image
from typing import List


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

    frames = [np.array(img) for img in frames]

    # 使用 imageio 写入 .webm 文件
    with iio.imopen(save_path, "w", plugin="FFMPEG") as writer:
        writer.write(frames, fps=fps, codec=codec)


def read_n_frames(video_path: str, n_frames: int, target_fps=16, last_n=False):
    from decord import VideoReader

    vr = VideoReader(video_path)
    original_fps = vr.get_avg_fps()
    total_frames = len(vr)
    interval = max(1, round(original_fps / target_fps))
    required_span = (n_frames - 1) * interval
    start_frame = max(0, total_frames - required_span - 1) if last_n else 0
    sampled_indices = []
    for i in range(n_frames):
        frame_idx = start_frame + i * interval
        if frame_idx >= total_frames:
            break
        else:
            sampled_indices.append(frame_idx)
    return vr.get_batch(sampled_indices).asnumpy()


def save_video_with_audio(frames: List[Image.Image], audio_path: str, target_video_path: str, fps: int=16):
    # combine all frames
    from moviepy import ImageSequenceClip, AudioFileClip, VideoClip
    video = [np.array(frame) for frame in frames]  # shape: t* (b*h, w, c)
    video_clip = ImageSequenceClip(video, fps=fps)
    audio_clip = AudioFileClip(audio_path)
    if audio_clip.duration > video_clip.duration:
        audio_clip: AudioFileClip = audio_clip.subclipped(0, video_clip.duration)  # clip audio
    else:
        video_clip: VideoClip = video_clip.subclipped(0, audio_clip.duration)
    video_with_audio: VideoClip = video_clip.with_audio(audio_clip)
    video_with_audio.write_videofile(target_video_path, codec='libx264')
