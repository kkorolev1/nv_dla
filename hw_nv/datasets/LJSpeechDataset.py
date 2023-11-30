import time
from tqdm.auto import tqdm
import torchaudio
import os
from pathlib import Path
import torch
from hw_nv.model.mel_spectrogram import MelSpectrogramConfig, MelSpectrogram


class LJSpeechDataset:
    def __init__(self, data_path, limit=None, **kwargs):
        data_path = Path(data_path)
        self.paths = []

        for wav_path in data_path.iterdir():
            self.paths.append(wav_path)

        if limit is not None:
            self.paths = self.paths[:limit]

        mel_spec_config = MelSpectrogramConfig()
        self.mel_spec_transform = MelSpectrogram(mel_spec_config)

    def __getitem__(self, index):
        wav_gt, _ = torchaudio.load(self.paths[index])
        mel_gt = self.mel_spec_transform(wav_gt.detach()).squeeze(0)
        return {
            "wav_gt": wav_gt,
            "mel_gt": mel_gt
        }

    def __len__(self):
        return len(self.paths)