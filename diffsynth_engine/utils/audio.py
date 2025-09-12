from typing import List

import torch
import torchaudio


def save_audio(audios: List[torch.Tensor], save_path: str, sample_rate: int = 48000, format: str = "wav"):
    backend = "soundfile" if format != "ogg" else "sox"
    output_audio_paths = []
    for i, audio in enumerate(audios):
        output_audio_path = f"{save_path}_{i}.{format}"
        torchaudio.save(
            output_audio_path, audio, sample_rate=sample_rate, format=format, backend=backend
        )
        output_audio_paths.append(output_audio_path)
    return output_audio_paths