import random
from pathlib import Path
from random import shuffle

import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import os

from hw_nv.base import BaseTrainer
from hw_nv.utils import inf_loop, MetricTracker

class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer_d,
            optimizer_g,
            lr_scheduler_d,
            lr_scheduler_g,
            config,
            device,
            dataloaders,
            len_epoch=None,
            skip_oom=True
    ):
        super().__init__(
            model,
            criterion,
            metrics,
            optimizer_d,
            optimizer_g,
            lr_scheduler_d,
            lr_scheduler_g,
            config,
            device
        )
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "mpd_d_loss",
            "msd_d_loss",
            "d_loss",
            "mpd_g_loss",
            "msd_g_loss",
            "mel_spec_g_loss",
            "mpd_features_g_loss",
            "msd_features_g_loss",
            "g_loss",
            "grad norm",
            *[m.name for m in self.metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", *[m.name for m in self.metrics], writer=self.writer
        )

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["wav"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self, module):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                module.parameters(), self.config["trainer"]["grad_norm_clip"]
            )
    
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()

        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    batch_idx,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} DLoss: {:.6f} GLoss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["d_loss"].item(),
                        batch["g_loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "discriminator learning rate", self.lr_scheduler_d.get_last_lr()[0]
                )
                self.writer.add_scalar(
                    "generator learning rate", self.lr_scheduler_g.get_last_lr()[0]
                )
                #self._log_predictions(**batch, is_train=True)
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        self.lr_scheduler_d.step()
        self.lr_scheduler_g.step()

        return log


    def process_batch(self, batch, batch_idx, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)

        # wav_gt: Bx1xL
        wav_gt = batch["wav"]
        mel_spec_gt = self.model.mel_spec_transform(wav_gt).squeeze(1)

        wav_pred = self.model.generator(mel_spec_gt)
        mel_spec_pred = self.model.mel_spec_transform(wav_pred).squeeze(1)

        # ---- Discriminator loss
        self.optimizer_d.zero_grad()

        # Don't need features for discriminator loss
        mpd_gt_outputs, _ = self.model.mpd(wav_gt)
        mpd_outputs, _ = self.model.mpd(wav_pred.detach())

        msd_gt_outputs, _ = self.model.msd(wav_gt)
        msd_outputs, _ = self.model.msd(wav_pred.detach())

        mpd_d_loss = self.criterion.discriminator_adv_loss(mpd_gt_outputs, mpd_outputs)
        msd_d_loss = self.criterion.discriminator_adv_loss(msd_gt_outputs, msd_outputs)

        d_loss = mpd_d_loss + msd_d_loss

        d_loss.backward()
        # TODO: clip_grad_norm
        self._clip_grad_norm(self.model.mpd)
        self._clip_grad_norm(self.model.msd)
        self.optimizer_d.step()

        # ---- Generator loss
        self.optimizer_g.zero_grad()

        # Don't need gt output for generator loss
        _, mpd_gt_features = self.model.mpd(wav_gt)
        mpd_outputs, mpd_features = self.model.mpd(wav_pred)

        _, msd_gt_features = self.model.msd(wav_gt)
        msd_outputs, msd_features = self.model.msd(wav_pred)

        mpd_g_loss = self.criterion.generator_adv_loss(mpd_outputs)
        msd_g_loss = self.criterion.generator_adv_loss(msd_outputs)

        mel_spec_g_loss = self.criterion.mel_spectrogram_loss(mel_spec_gt, mel_spec_pred)
        
        mpd_features_g_loss = self.criterion.feature_matching_loss(mpd_gt_features, mpd_features)
        msd_features_g_loss = self.criterion.feature_matching_loss(msd_gt_features, msd_features)

        g_loss = mpd_g_loss + msd_g_loss + mel_spec_g_loss + mpd_features_g_loss + msd_features_g_loss

        g_loss.backward()
        # TODO: clip_grad_norm
        self._clip_grad_norm(self.model.generator)
        self.optimizer_g.step()
        
        batch["mpd_d_loss"] = mpd_d_loss
        batch["msd_d_loss"] = msd_d_loss
        batch["d_loss"] = d_loss

        batch["mpd_g_loss"] = mpd_g_loss
        batch["msd_g_loss"] = msd_g_loss
        batch["mel_spec_g_loss"] = mel_spec_g_loss
        batch["mpd_features_g_loss"] = mpd_features_g_loss
        batch["msd_features_g_loss"] = msd_features_g_loss
        batch["g_loss"] = g_loss
        batch["grad norm"] = torch.tensor([self.get_grad_norm()])
    
        for metric_key in metrics.keys():
            metrics.update(metric_key, batch[metric_key].item())

        return batch

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_audio(self, audio, sr, name):
        self.writer.add_audio(f"Audio_{name}", audio, sample_rate=sr)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]

        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
