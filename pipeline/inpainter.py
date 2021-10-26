import logging
from pathlib import Path
from typing import Dict, Tuple, Any

import gdal
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from pytorch_lightning.core import LightningModule
from hydra.utils import instantiate
from PIL import Image
import numpy as np
from torchvision import transforms

logger = logging.getLogger(__name__)


class LitInpainter(LightningModule):

    def __init__(
        self, cfg: Dict[str, Any], **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.model = instantiate(self.cfg.model)
        self.loss = instantiate(self.cfg.loss)
        self.transforms = []   # for geotiff
        self.projections = []  # for geotiff
        self.dimensions = []   # for geotiff

    def forward(self, img_miss, mask):
        return self.model(img_miss, mask)

    def training_step(self, batch, batch_idx):
        img, mask = batch
        # todo: use smaller image dimension (512 -> 256)
        # todo: make this threshold a hydra config option
        mask[mask < 0.9] = 0.0  # to mitigate interpolation by resizing
        mask[mask >= 0.9] = 1.0  # to mitigate interpolation by resizing
        img_miss = img * mask
        output, _, _ = self(img_miss, mask)
        loss, loss_detail = self.loss(output, img, mask)

        self.log(
            'train_loss', loss, on_step=True, on_epoch=True)
        self.log(
            'train/mse', loss_detail['reconstruction_loss'],
            on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            'train/percep', loss_detail['perceptual_loss'],
            on_step=True, on_epoch=True)
        self.log(
            'train/style', loss_detail['style_loss'],
            on_step=True, on_epoch=True)
        self.log(
            'train/tv', loss_detail['total_variation_loss'],
            on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        # todo: make this threshold a hydra config option
        mask[mask < 0.9] = 0.0  # to mitigate interpolation by resizing
        mask[mask >= 0.9] = 1.0  # to mitigate interpolation by resizing
        img_miss = img * mask
        fulls, alphas, fills = self(img_miss, mask)
        loss, loss_detail = self.loss(fulls, img, mask)

        self.log(
            'val_loss', loss, on_epoch=True, prog_bar=True)
        self.log(
            'val/mse', loss_detail['reconstruction_loss'],
            on_epoch=True, prog_bar=True)
        self.log(
            'val/percep', loss_detail['perceptual_loss'], on_epoch=True)
        self.log(
            'val/style', loss_detail['style_loss'], on_epoch=True)
        self.log(
            'val/tv', loss_detail['total_variation_loss'], on_epoch=True)

        if self.trainer.is_global_zero:
            # To save n_save * n_image_per_batch samples into files every epoch.
            batch_interval = max(
                1, len(self.trainer.datamodule.val_dataloader()) // (
                    self.trainer.world_size * (self.cfg.val_save.n_save - 1)))
            if batch_idx % batch_interval == 0:
                save_dir = Path(f'result/epoch_{self.current_epoch}/')
                save_dir.mkdir(exist_ok=True, parents=True)
                n = self.cfg.val_save.n_image_per_batch
                full, alpha, fill = fulls[0], alphas[0], fills[0]

                # unnormalizing to [0, 255] to round to nearest integer
                save_image(
                    torch.cat((
                        img[:n], img_miss[:n],
                        alpha[:n], fill[:n], full[:n]), dim=0),
                    save_dir / f'{batch_idx:09d}.tif', nrow=n)
        return loss

    def test_step(self, batch, batch_idx):
        img, mask = batch
        # todo: make this threshold a hydra config option
        mask[mask < 0.90] = 0.0  # to mitigate interpolation by resizing
        mask[mask >= 0.90] = 1.0  # to mitigate interpolation by resizing
        img_miss = img * mask
        fulls, alphas, fills = self(img_miss, mask)
        loss, loss_detail = self.loss(fulls, img, mask)

        self.log(
            'test_loss', loss, on_epoch=True, prog_bar=True)
        self.log(
            'test/mse', loss_detail['reconstruction_loss'],
            on_epoch=True, prog_bar=True)
        self.log(
            'test/percep', loss_detail['perceptual_loss'], on_epoch=True)
        self.log(
            'test/style', loss_detail['style_loss'], on_epoch=True)
        self.log(
            'test/tv', loss_detail['total_variation_loss'], on_epoch=True)

        if self.trainer.is_global_zero:
            save_dir = Path(f'result/test/')
            save_dir.mkdir(exist_ok=True, parents=True)
            full, alpha, fill = fulls[0], alphas[0], fills[0]

            if self.cfg.custom.verbose:
                # save a stacked image of {input, masked, alpha, fill, full}
                # unnormalising to [0, 255] to round to nearest integer
                save_image(
                    torch.cat((
                        img[:], img_miss[:],
                        alpha[:], fill[:], full[:]), dim=0),
                    save_dir / f'{batch_idx:09d}.tif', nrow=self.cfg.dataset.batch_size)
            else:
                # save only the prediction, and convert it back to the height value
                # re-wrap to original resolution
                prediction = full[:]
                prediction = transforms.Resize((self.dimensions[batch_idx][1], self.dimensions[batch_idx][0]))(prediction)
                array = torch.squeeze(prediction).cpu().numpy()[0]  # retrieve arbitrary band

                # unnormalising to height field
                # todo: with hrdra config
                array = array * 100.0 - 10.0

                # write out geotiff
                driver = gdal.GetDriverByName('GTiff')
                out = driver.Create(str(save_dir / f'{batch_idx:09d}.tif'), self.dimensions[batch_idx][0], self.dimensions[batch_idx][1], 1, gdal.GDT_Float32)
                out.SetGeoTransform(self.transforms[batch_idx])  # sets the same geotransform as input
                out.SetProjection(self.projections[batch_idx])   # sets the same projection as input
                out.GetRasterBand(1).WriteArray(array)
                out.FlushCache()

        return loss

    def _log_metrics(self):
        if self.trainer.is_global_zero:
            str_metrics = ''
            for key, val in self.trainer.logged_metrics.items():
                str_metrics += f'\n\t{key}: {val}'
            logger.info(str_metrics)

    def on_validation_end(self):
        self._log_metrics()

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optim, self.parameters())
        scheduler = instantiate(self.cfg.scheduler, optimizer)
        return [optimizer], [scheduler]
