import time
from tqdm.auto import tqdm
import torchaudio
import os
from pathlib import Path
import torch


class LJSpeechDataset:
    def __init__(self, data_path, limit=None, **kwargs):
        data_path = Path(data_path)
        self.paths = []

        for wav_path in data_path.iterdir():
            self.paths.append(wav_path)

        if limit is not None:
            self.paths = self.paths[:limit]

    def __getitem__(self, index):
        wav, sr = torchaudio.load(self.paths[index])
        return {
            "wav": wav
        }

    def __len__(self):
        return len(self.paths)