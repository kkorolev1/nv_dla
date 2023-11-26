import argparse
import collections
import warnings
import itertools

import numpy as np
import torch

import hw_nv.loss as module_loss
import hw_nv.metric as module_metric
import hw_nv.model as module_arch
from hw_nv.trainer import Trainer
from hw_nv.utils import prepare_device
from hw_nv.utils.object_loading import get_dataloaders
from hw_nv.utils.parse_config import ConfigParser
import hw_nv.utils.lr_scheduler


warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)
    
    model = config.init_obj(config["arch"], module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"], logger)
    logger.info(f"Device {device}")
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    # get function handles of loss and metrics
    loss_module = config.init_obj(config["loss"], module_loss).to(device)
    metrics = [
        config.init_obj(metric_dict, module_metric)
        for metric_dict in config["metrics"]
    ]

    logger.info(f'Len epoch {config["trainer"]["len_epoch"]}')
    logger.info(f'Epochs {config["trainer"]["epochs"]}')
    logger.info(f'Dataset size {len(dataloaders["train"].dataset)}')
    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    trainable_params_d = filter(lambda p: p.requires_grad, itertools.chain(model.mpd.parameters(), model.msd.parameters()))
    optimizer_d = config.init_obj(config["optimizer_d"], torch.optim, trainable_params_d)
    trainable_params_g = filter(lambda p: p.requires_grad, model.generator.parameters())
    optimizer_g = config.init_obj(config["optimizer_g"], torch.optim, trainable_params_g)
    lr_scheduler_d = config.init_obj(config["lr_scheduler_d"], hw_nv.utils.lr_scheduler, optimizer_d)
    lr_scheduler_g = config.init_obj(config["lr_scheduler_g"], hw_nv.utils.lr_scheduler, optimizer_g)

    trainer = Trainer(
        model,
        loss_module,
        metrics,
        optimizer_d,
        optimizer_g,
        lr_scheduler_d,
        lr_scheduler_g,
        config=config,
        device=device,
        dataloaders=dataloaders,
        len_epoch=config["trainer"].get("len_epoch", None)
    )

    trainer.train()


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
        default=None,
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
        "-k",
        "--wandb_key",
        default=None,
        type=str,
        help="WanDB API key",
    )
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--n_gpu"], type=int, target="n_gpu"
        ),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data;train;batch_size"
        ),
        CustomArgs(
            ["--reset_optimizer"], type=bool, target="trainer;reset_optimizer"
        ),
        CustomArgs(
            ["--epochs"], type=int, target="trainer;epochs"
        ),
        CustomArgs(
            ["--len_epoch"], type=int, target="trainer;len_epoch"
        ),
        CustomArgs(
            ["--wandb_run_name"], type=str, target="trainer;wandb_run_name"
        ),
        CustomArgs(
            ["--data_path"], type=str, target="data;train;datasets;[0];args;data_path"
        ),
        CustomArgs(
            ["--limit"], type=int, target="data;train;datasets;[0];args;limit"
        )
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
