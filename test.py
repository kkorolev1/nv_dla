import argparse
import json
import os
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

import hw_nv.model as module_model
from hw_nv.trainer import Trainer
from hw_nv.utils import ROOT_PATH
from hw_nv.utils.parse_config import ConfigParser
from hw_nv.model.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig


DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, test_dir, output_dir, device):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device(device)

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    logger.info(f"Device {device}")
    model = model.to(device)
    model.eval()
    model.generator.remove_normalization()

    os.makedirs(output_dir, exist_ok=True)
    test_dir = Path(test_dir)
    output_dir = Path(output_dir)

    sampling_rate = 22050
    mel_spec_config = MelSpectrogramConfig()
    mel_spec_transform = MelSpectrogram(mel_spec_config).to(device)

    with torch.no_grad():
        for wav_path in tqdm(test_dir.iterdir(), "Processing wavs"):
            wav = torchaudio.load(wav_path)[0].to(device)
            mel_spec = mel_spec_transform(wav)
            wav_pred = model.generator(mel_spec).squeeze(0).cpu()
            torchaudio.save(output_dir / wav_path.name, wav_pred, sample_rate=sampling_rate)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-t",
        "--test-dir",
        default="test_audio",
        type=str,
        help="Directory with test audio wav files",
    )
    args.add_argument(
        "-o",
        "--output-dir",
        default="output",
        type=str,
        help="Output directory",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    # model_config = Path(args.resume).parent / "config_server.json"
    model_config = Path(args.config)
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    main(config, args.test_dir, args.output_dir, args.device)
