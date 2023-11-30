import numpy as np
import torch
import torch.nn.functional as F


def pad_1D_tensor(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = F.pad(x, (0, length - x.shape[0]), value=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D_tensor(inputs, maxlen=None):

    def pad(x, max_len):
        if x.size(-1) > max_len:
            raise ValueError("not max_len")

        x_padded = F.pad(x, (0, max_len - x.size(-1), 0, 0))
        return x_padded

    if maxlen:
        output = torch.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(x.size(-1) for x in inputs)
        output = torch.stack([pad(x, max_len) for x in inputs])

    return output

def collate(batch):
    output = {}
    output["wav_gt"] = pad_1D_tensor([item["wav_gt"].squeeze(0) for item in batch]).unsqueeze(1)
    output["mel_gt"] = pad_2D_tensor([item["mel_gt"] for item in batch])
    return output