import logging
from pathlib import Path
from typing import Dict, Tuple, Any

import gdal
import torch
from torchvision.utils import save_image
from pytorch_lightning.core import LightningModule
from hydra.utils import instantiate
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
        self.transforms = []  # for geotiff
        self.dimensions = []  # for geotiff

    def forward(self, img_miss, mask):
        return self.model(img_miss, mask)

    def base_step(self, batch):
        img, mask = batch

        # binary mask
        mask[mask < self.cfg.dataset.mask.threshold] = 0.0   # to mitigate interpolation by resizing
        mask[mask >= self.cfg.dataset.mask.threshold] = 1.0  # to mitigate interpolation by resizing

        img_miss = img * mask
        fulls, alphas, fills = self(img_miss, mask)
        loss, loss_detail = self.loss(fulls, img, mask)
        return loss, loss_detail, img, img_miss, fulls, alphas, fills

    def training_step(self, batch, batch_idx):
        loss, loss_detail, _, _, _, _, _ = self.base_step(batch)
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
        loss, loss_detail, img, img_miss, fulls, alphas, fills = self.base_step(batch)

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

                # unnormalising to [0, 255] to round to nearest integer
                save_image(
                    torch.cat((
                        img[:n], img_miss[:n],
                        alpha[:n], fill[:n], full[:n]), dim=0),
                    save_dir / f'{batch_idx:09d}.tif', nrow=n)
        return loss

    def test_step(self, batch, batch_idx):
        loss, loss_detail, img, img_miss, fulls, alphas, fills = self.base_step(batch)

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

            if self.cfg.verbose:
                # save a stacked image of {input, masked, alpha, fill, full}
                # unnormalising to [0, 255] to round to nearest integer
                save_image(
                    torch.cat((
                        img[:], img_miss[:],
                        alpha[:], fill[:], full[:]), dim=0),
                    save_dir / f'{batch_idx:09d}.tif', nrow=self.cfg.dataset.batch_size)

            else:
                # re-wrap to original resolution
                prediction = transforms.Resize((self.dimensions[batch_idx][1], self.dimensions[batch_idx][0]))(full)
                array = torch.squeeze(prediction, 0).cpu().numpy()[0]  # retrieve arbitrary band (here index 0)

                # unnormalising to height field
                array = array / self.cfg.dataset.normaliser.scale - self.cfg.dataset.normaliser.gain

                # write out geotiff
                driver = gdal.GetDriverByName('GTiff')
                out = driver.Create(str(save_dir / f'{batch_idx:09d}.tif'), self.dimensions[batch_idx][0],
                                    self.dimensions[batch_idx][1], 1, gdal.GDT_Float32)
                out.SetGeoTransform(self.transforms[batch_idx])  # sets the same reference as input
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
