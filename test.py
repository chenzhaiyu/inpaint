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
from tqdm import trange, tqdm
from pathlib import Path
import subprocess

logger = logging.getLogger(__name__)


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
    path_mask.parent.mkdir(exist_ok=True, parents=True)
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
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    for input_path in tqdm(input_dir.rglob('*' + suffix)):
        output_path = output_dir / input_path.relative_to(input_dir)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        image = gdal.Open(str(input_path))
        has_nodata, _ = nodata(image)
        if has_nodata:
            subprocess.run(["rsync"] + [str(input_path)] + [str(output_path)])


@hydra.main(config_path='conf', config_name='config')
def test(cfg: DictConfig) -> None:
    """
    Run test on default/custom images with no-data pixels.
    """
    print_config(cfg)
    pl._logger.handlers = []
    pl._logger.propagate = True

    if cfg.seed is not None:
        pl.seed_everything(cfg.seed)

    # load parameters from checkpoint but overwrite cfg
    model = LitInpainter.load_from_checkpoint(cfg.dataset.data.checkpoint)
    model.cfg = cfg

    # if hasattr(model.cfg, 'verbose') and not model.cfg.verbose and not model.cfg.custom:
    #     logger.warning('"verbose=False" is only for custom data. Reset verbose=True.')
    #     model.cfg.verbose = True

    if model.cfg.custom:
        transforms = []
        dimensions = []

        logger.info('No-data mask validation/generation')
        for sample_path in tqdm(ImageDataset(cfg.dataset.data.tocomplete_dir).samples):
            path_mask = (Path(cfg.dataset.mask.tocomplete_dir) / Path(sample_path).relative_to(
                cfg.dataset.data.tocomplete_dir)).with_suffix('.jpg')

            # load transform and projection for geotiff output
            if not model.cfg.verbose:
                ds = gdal.Open(sample_path)
                transforms.append(ds.GetGeoTransform())
                dimensions.append((ds.RasterXSize, ds.RasterYSize))

            # generate mask files, if they do not exist
            if not path_mask.exists():
                create_nodata_mask(sample_path, path_mask)

        model.transforms = transforms
        model.dimensions = dimensions

        # override dataset.data.test_dir to custom images
        # override dataset.mask.test_dir to custom masks
        model.cfg.dataset.data.test_dir = cfg.dataset.data.tocomplete_dir
        model.cfg.dataset.mask.test_dir = cfg.dataset.mask.tocomplete_dir

    trainer = init_trainer(cfg)
    datamodule = instantiate(cfg.dataset)

    trainer.test(model, datamodule=datamodule)


if __name__ == '__main__':
    test()
