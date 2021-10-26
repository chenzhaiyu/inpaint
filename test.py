import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
from utils.log import print_config, save_lr_finder
from pipeline.inpainter import LitInpainter
from train import init_trainer
from dataset.image import ImageDataset
from osgeo import gdal
import numpy as np
from PIL import Image
import os
import glob
from pathlib import Path
import subprocess

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
    assert has_nodata, 'encountered a complete image. consider do prepare() before test_custom()'
    mask = Image.fromarray(mask_nodata)
    mask.save(path_mask)


def prepare(input_dir, output_dir, suffix='.tif'):
    """
    Batch copy images to a specified directory.
    """
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    for input_path in glob.glob(os.path.join(input_dir, '*')):
        output_path = os.path.join(output_dir, Path(input_path).stem + suffix)
        image = gdal.Open(input_path)
        has_nodata, _ = nodata(image)
        if has_nodata:
            subprocess.run(["rsync"] + [input_path] + [output_path])


@hydra.main(config_path='conf', config_name='config')
def test_custom(cfg: DictConfig) -> None:
    """
    Run test on custom images with no-data pixels.
    """
    print_config(cfg)
    pl._logger.handlers = []
    pl._logger.propagate = True

    if cfg.seed is not None:
        pl.seed_everything(cfg.seed)

    # load retrained weights and configs
    model = LitInpainter.load_from_checkpoint(cfg.custom.checkpoint)
    model.cfg = cfg  # overwrite cfg
    model.cfg.dataset.batch_size = cfg.dataset.batch_size if cfg.custom.verbose else 1

    if model.cfg.mode == 'debug':
        logger.warning('Debug mode enabled. Set mode=run to avoid possible tensorboard logging exception.')

    transforms = []
    dimensions = []

    for sample_path in ImageDataset(cfg.custom.image_dir).samples:
        path_mask = (Path(cfg.custom.mask_dir) / Path(sample_path).name).with_suffix('.jpg')

        # load transform and projection for geotiff output
        if not cfg.custom.verbose:
            ds = gdal.Open(sample_path)
            transforms.append(ds.GetGeoTransform())
            dimensions.append((ds.RasterXSize, ds.RasterYSize))

        # generate mask files, if they do not exist
        if not path_mask.exists():
            create_nodata_mask(sample_path, path_mask)

    model.transforms = transforms
    model.dimensions = dimensions

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
