import os
from PIL import Image
from torchvision import datasets, transforms
from hydra.utils import get_original_cwd, to_absolute_path


class ImageDataset(datasets.DatasetFolder):

    def __init__(self, data_dir: str, transform=None, normaliser=None):
        self.data_dir = data_dir
        self.transform = transform
        self.normaliser = normaliser
        self.samples = []
        for r, _, filenames in sorted(
                os.walk(os.path.expanduser(self.data_dir), followlinks=True)):
            for filename in sorted(filenames):
                path = os.path.join(r, filename)
                if datasets.folder.is_image_file(path):
                    self.samples.append(path)

    def loader(self, path):
        """
        A replacement of the default (PIL) loader.
        """
        if path.endswith(".jpg"):
            # fallback to default loader for masks
            mask = datasets.folder.default_loader(path)
            return mask
        elif path.endswith(".tif"):
            with open(path, 'rb') as f:
                img = Image.open(f)
                """
                Converts a PIL Image or numpy.ndarray (H x W x C) in the range
                [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
                or if the numpy.ndarray has dtype = np.uint8
                In the other cases, tensors are returned without scaling.
                """
                tsr = transforms.ToTensor()(img)  # no scaling for tiff float (though in mode 'F')
                tsr = (tsr + self.normaliser.gain) * self.normaliser.scale  # manual scaling to [0-1]
                tsr[tsr < -1] = self.normaliser.nodata  # truncate no-data values (-3.4*10e+38) todo: robust catch
                tsr = tsr.expand(3, -1, -1)
                return tsr
        else:
            raise ValueError('Only JPEG and TIFF are supported')

    def __getitem__(self, index):
        """
        Reserved for train/val/test on full-fledged dataset, not on custom images.
        """
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
