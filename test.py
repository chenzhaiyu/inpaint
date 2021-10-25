import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
from utils.log import print_config, save_lr_finder
from pipeline.inpainter import LitInpainter
from train import init_trainer
from dataset.image import ImageDataset
from dataset.inpaint import InpaintDataModule, InpaintDataset
from osgeo import gdal
import numpy as np
from PIL import Image
from pathlib import Path

logger = logging.getLogger(__name__)


@hydra.main(config_path='conf', config_name='config')
def test_split(cfg: DictConfig) -> None:
    """
    Run test on test split with random masks
    """
    print_config(cfg)
    pl._logger.handlers = []
    pl._logger.propagate = True

    if cfg.seed is not None:
        pl.seed_everything(cfg.seed)

    model = LitInpainter.load_from_checkpoint(cfg.custom.checkpoint)
    model.cfg.dataset.batch_size = cfg.dataset.batch_size

    trainer = init_trainer(cfg)
    datamodule = instantiate(cfg.dataset)

    trainer.test(model, datamodule=datamodule)


def nodata(image):
    """
    Return no-data flag, and mask of no-data pixels, if any.
    :param image: gdal image
    :return: whether given image has no-data pixels
    """
    band = image.GetRasterBand(1)
    value = band.GetNoDataValue()
    array = band.ReadAsArray()
    has_nodata = value in array
    mask_nodata = np.zeros(array.shape)
    if has_nodata:
        mask_nodata = value != array
    return has_nodata, mask_nodata


def create_nodata_mask(path_image, path_mask):
    """
    Create no-data masks for images with no-data pixels.
    No-data pixels are filled with 0s, and a no-data mask is created.
    """
    assert path_image.endswith('.tif')
    image_gdal = gdal.Open(path_image)
    has_nodata, mask_nodata = nodata(image_gdal)
    assert has_nodata
    mask = Image.fromarray(mask_nodata)
    mask.save(path_mask)


@hydra.main(config_path='conf', config_name='config')
def test_custom(cfg: DictConfig) -> None:
    """
    Run test on custom images with no-data pixels
    """
    print_config(cfg)
    pl._logger.handlers = []
    pl._logger.propagate = True

    if cfg.seed is not None:
        pl.seed_everything(cfg.seed)

    model = LitInpainter(cfg)
    model.load_from_checkpoint(cfg.custom.checkpoint)
    model.cfg.dataset.batch_size = cfg.dataset.batch_size

    # generate mask files, if they do not exist
    for sample_path in ImageDataset(cfg.custom.image_dir).samples:
        path_mask = (Path(cfg.custom.mask_dir) / Path(sample_path).name).with_suffix(cfg.custom.mask_suffix)
        if not path_mask.exists():
            # create mask file
            create_nodata_mask(sample_path, path_mask)

    trainer = init_trainer(cfg)

    # override dataset.data.test_dir to custom images
    # override dataset.mask.test_dir to custom masks
    cfg.dataset.data.test_dir = cfg.custom.image_dir
    cfg.dataset.mask.test_dir = cfg.custom.mask_dir
    datamodule = instantiate(cfg.dataset)

    trainer.test(model, datamodule=datamodule)


if __name__ == '__main__':
    # test_split()  # run test on test split data
    test_custom()  # run test on custom data
