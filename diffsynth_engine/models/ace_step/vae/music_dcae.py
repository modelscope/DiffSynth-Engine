import torch
from torchvision import transforms
import torchaudio

from diffsynth_engine.models.base import PreTrainedModel

from .hifi_gan import ADaMoSHiFiGANV1
from .dcae import DCAE # TODO: rewrite above 2 files


class MusicDCAE(PreTrainedModel):
    def __init__(
        self,
        dcae: DCAE,
        vocoder: ADaMoSHiFiGANV1,
        source_sample_rate: int = 48000,
    ):
        super(MusicDCAE, self).__init__()

        self.dcae = dcae
        self.vocoder = vocoder
        self.resampler = torchaudio.transforms.Resample(source_sample_rate, 44100)
        self.transform = transforms.Compose([transforms.Normalize(0.5, 0.5)])
        self.min_mel_value = -11.0
        self.max_mel_value = 3.0
        self.time_dimention_multiple = 8
        self.scale_factor = 0.1786
        self.shift_factor = -1.9091

    def load_audio(self, audio_path):
        audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        return audio, sr

    def forward_mel(self, audios):
        mels = []
        for i in range(len(audios)):
            image = self.vocoder.mel_transform(audios[i])
            mels.append(image)
        mels = torch.stack(mels)
        return mels

    @torch.no_grad()
    def encode(self, audios: torch.Tensor, sr: int):
        audio_lengths = torch.tensor([audios.shape[2]] * audios.shape[0])
        audio_lengths = audio_lengths.to(audios.device)

        # audios: N x 2 x T, 48kHz
        resampler = torchaudio.transforms.Resample(sr, 44100)
        resampler = resampler.to(device=audios.device, dtype=audios.dtype)
        audio = resampler(audios)

        max_audio_len = audio.shape[-1]
        if max_audio_len % (8 * 512) != 0:
            audio = torch.nn.functional.pad(audio, (0, 8 * 512 - max_audio_len % (8 * 512)))

        mels = self.forward_mel(audio)
        mels = (mels - self.min_mel_value) / (self.max_mel_value - self.min_mel_value)
        mels = self.transform(mels)
        latents = []
        for mel in mels:
            latent = self.dcae.encoder(mel.unsqueeze(0))
            latents.append(latent)
        latents = torch.cat(latents, dim=0)
        latent_lengths = (audio_lengths / sr * 44100 / 512 / self.time_dimention_multiple).long()
        latents = (latents - self.shift_factor) * self.scale_factor
        return latents, latent_lengths

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, sr: int):
        latents = latents / self.scale_factor + self.shift_factor

        pred_wavs = []
        for latent in latents:
            mels = self.dcae.decoder(latent[None])
            mels = mels * 0.5 + 0.5
            mels = mels * (self.max_mel_value - self.min_mel_value) + self.min_mel_value

            # decode waveform for each channels to reduce vram footprint
            wav_ch1 = self.vocoder.decode(mels[:, 0, :, :]).squeeze(1).cpu()
            wav_ch2 = self.vocoder.decode(mels[:, 1, :, :]).squeeze(1).cpu()
            wav = torch.cat([wav_ch1, wav_ch2], dim=0)

            resampler = torchaudio.transforms.Resample(44100, sr)
            wav = resampler(wav.cpu().float())
            pred_wavs.append(wav)

        return pred_wavs
