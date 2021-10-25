import os
from PIL import Image
from torchvision import datasets, transforms


class ImageDataset(datasets.DatasetFolder):

    def __init__(self, data_dir: str = '/path/to/data', transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        for r, _, fnames in sorted(
                os.walk(os.path.expanduser(self.data_dir), followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(r, fname)
                if datasets.folder.is_image_file(path):
                    self.samples.append(path)

    @staticmethod
    def loader(path, fill_nodata=False):
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

                tsr = (tsr + 10) / 100.0  # manual scaling to [0-1]
                tsr[tsr < -1] = 0.0  # truncate no-data values (-3.4*10e+38)
                tsr = tsr.expand(3, -1, -1)
                return tsr

    def __getitem__(self, index):
        """
        Reserved for train/val/test on full-fledged dataset, not on custom images.
        """
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
