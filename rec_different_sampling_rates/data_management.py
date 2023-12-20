import glob
import os
import random
#import h5py
from os.path import join

import numpy as np
import torch

#from fastmri_utils.data import transforms
from tqdm import tqdm

from extra_utils import cut_to_01

from PIL import Image
# ----- Dataset creation, saving, and loading -----


def create_iterable_dataset(
    n, set_params, iterator, iter_params,
):
    """ Creates training, validation, and test data sets.

    Samples data signals from a data generator and stores them.

    Parameters
    ----------
    n : int
        Dimension of signals x.
    set_params : dictionary
        Must contain values for the following keys:
        path : str
            Directory path for storing the data sets.
        num_train : int
            Number of samples in the training set.
        num_val : int
            Number of samples in the validation set.
        num_test : int
            Number of samples in the validation set.
    generator : callable
        Generator function to create signal samples x. Will be called with
        the signature generator(n, **gen_params).
    gen_params : dictionary
        Additional keyword arguments passed on to the signal generator.
    """

    dataset_train = iterator(mode='train', **iter_params)
    dataset_val = iterator(mode='val', **iter_params)
    dataset_test = iterator(mode='test', **iter_params)
    iter_train = iter(dataset_train)
    iter_val = iter(dataset_val)
    iter_test = iter(dataset_test)
    os.makedirs(os.path.join(set_params["path"], "train"), exist_ok=True)
    os.makedirs(os.path.join(set_params["path"], "val"), exist_ok=True)
    os.makedirs(os.path.join(set_params["path"], "test"), exist_ok=True)

    for idx in tqdm(range(len(iter_train)), desc="generating training signals"):
        torch.save(
            next(iter_train),
            os.path.join(
                set_params["path"], "train", "sample_{:05d}.pt".format(idx)
            ),
        )

    for idx in tqdm(range(len(iter_val)), desc="generating validation signals"):
        torch.save(
            next(iter_val),
            os.path.join(
                set_params["path"], "val", "sample_{:05d}.pt".format(idx)
            ),
        )

    for idx in tqdm(range(len(iter_test)), desc="generating test signals"):
        torch.save(
            next(iter_test),
            os.path.join(
                set_params["path"], "test", "sample_{:05d}.pt".format(idx)
            ),
        )


class Load_fastMRI_dataset:
    """ Loads a dataset of fastMRI images.


    Parameters
    ----------
    mode : str 
        One of 'train', 'test', 'val'. Decides which dataset to create.
    path_train : str
        Path to training images. 
    path_val : str
        Path to validation images. 
    path_test : str
        Path to testing images. 
    """
    def __init__(self, mode, path_train = None, path_val = None, path_test = None):
        
        self.mode = mode
        
        self.path_train = path_train
        self.path_val = path_val
        self.path_test = path_test
        
        if self.mode.lower() == 'train':
            self.path = path_train
        elif self.mode.lower() == 'val':
            self.path = path_val
        elif self.mode.lower() == 'test':
            self.path = path_test
        else:
            self.path = None
        if self.path is not None:
            self.filenames = glob.glob(join(self.path, '*.h5'))
        else:
            self.filenames = [];
   
        self.data_size = len(self.filenames)

    def __len__(self):
        return self.data_size

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.path is None:
            raise StopIteration
        if self.count < self.data_size:
            fname = self.filenames[self.count]
            self.count += 1
            with h5py.File(fname, 'r') as hf:
                rec = np.array(hf['reconstruction_rss'])
                rec = rec[int(rec.shape[0]/2), ...]
                amax = np.amax(rec)
                rec *= (255/amax)
                im = Image.fromarray(np.uint8(rec))
                im = im.resize((256,256))
                np_im = np.array(im).astype(np.float32)/255;
                
                return torch.tensor(np_im, dtype=torch.float)
        else:
            raise StopIteration

class IPDataset(torch.utils.data.Dataset):
    """ Datasets for imaging inverse problems.

    Loads image signals created by `create_iterable_dataset` from a directory.

    Implements the map-style dataset in `torch`.

    Attributed
    ----------
    subset : str
        One of "train", "val", "test".
    path : str
        The directory path. Should contain the subdirectories "train", "val",
        "test" containing the training, validation, and test data respectively.
    """

    def __init__(self, subset, path, transform=None, device=None):
        self.path = path
        self.files = glob.glob(os.path.join(path, subset, "*.pt"))
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # load image and add channel dimension
        out = torch.load(self.files[idx])
        if out.dim() == 2:
            out = out.unsqueeze(0)
        if self.device is not None:
            out.to(self.device)
        out = (out,)
        return self.transform(out) if self.transform is not None else out

# ----- data transforms -----

class JointRandomCrop(object):
    """ Joint random cropping transform for (input, target) image pairs. """

    def __init__(self, size):
        self.size = size

    def __call__(self, imgs):
        iw, ih = imgs[0].shape[-2:]  # image width and height
        cw, ch = self.size  # crop width and height

        # sample random corner of cropping area
        w0 = random.randint(0, iw - cw) if iw > cw else 0
        h0 = random.randint(0, ih - ch) if ih > ch else 0
        return tuple(img[..., w0 : w0 + cw, h0 : h0 + ch] for img in imgs)


class Inversion(object):
    """ Inverse transform on (meas, target) tuples.

    Inverts meas to image domain and returns (inv, target) pair.

    Parameters
    ----------
    inverter : callable
        The inversion operation to use.

    """

    def __init__(self, inverter):
        self.inverter = inverter

    def __call__(self, inputs):
        meas, target = inputs
        inv = self.inverter(meas)
        return inv, target


class SimulateMeasurements(object):
    """ Forward operator on target samples.

    Computes measurements and returns (measurement, target) pair.

    Parameters
    ----------
    operator : callable
        The measurement operation to use.

    """

    def __init__(self, operator):
        self.operator = operator

    def __call__(self, target):
        (target,) = target
        meas = self.operator(target)
        return meas, target


class ComplexMagnitude(object):
    """ Removes a complex channel from (input, target) image pairs.

    Returns the magnitude of an image as a single channel. If the image has no
    complex channel, then it is passed on unaltered.

    """

    def __init__(self):
        pass

    def __call__(self, imgs):
        return tuple(
            [
                rotate_real(img)[..., 0:1, :, :]
                if img.shape[-3] == 2
                else torch.abs(img)
                for img in imgs
            ]
        )


class ToComplex(object):
    """ Adds a complex channel to images.

    Transforms images of shape [..., 1, W, H] to shape [..., 2, W, H]
    by concatenating an empty channel for the imaginary part.

    """

    def __init__(self):
        pass

    def __call__(self, imgs):
        return tuple([to_complex(img) for img in imgs])


class CenterCrop(object):
    """ Crops (input, target) image pairs to have matching size. """

    def __init__(self, shape):
        self.shape = shape

    def __call__(self, imgs):
        return tuple([transforms.center_crop(img, self.shape) for img in imgs])


class Flatten(object):
    """ Flattens selected dimensions of tensors. """

    def __init__(self, start_dim, end_dim):
        self.start_dim = start_dim
        self.end_dim = end_dim

    def __call__(self, inputs):
        return tuple(
            [torch.flatten(x, self.start_dim, self.end_dim) for x in inputs]
        )


class Normalize(object):
    """ Normalizes (input, target) pairs with respect to target or input. """

    def __init__(self, p=2, reduction="sum", use_target=True):
        self.p = p
        self.reduction = reduction
        self.use_target = use_target

    def __call__(self, inputs):
        inp, tar = inputs
        norm = torch.norm(tar if self.use_target else inp, p=self.p)
        if self.reduction == "mean" and not self.p == "inf":
            norm /= np.prod(tar.shape) ** (1 / self.p)
        return inputs[0] / norm, inputs[1] / norm


class Jitter(object):
    """ Adds random pertubations to the input of (input, target) pairs.
    """

    def __init__(self, eta, scale_lo, scale_hi, n_seed=None, t_seed=None):
        self.eta = eta
        self.scale_lo = scale_lo
        self.scale_hi = scale_hi
        self.rng = np.random.RandomState(n_seed)
        self.trng = torch.Generator()
        if t_seed is not None:
            self.trng.manual_seed(t_seed)

    def __call__(self, inputs):
        meas, target = inputs
        m = meas.shape[-1]  # number of sampled measurements
        scale = (
            self.scale_lo + (self.scale_hi - self.scale_lo) * self.rng.rand()
        )
        noise = torch.randn(meas.shape, generator=self.trng).to(meas.device)
        meas_noisy = meas + self.eta / np.sqrt(m) * noise * scale
        return meas_noisy, target


# ---- run data generation -----
if __name__ == "__main__":
    import config
    
    np.random.seed(config.numpy_seed)
    torch.manual_seed(config.torch_seed)
    create_iterable_dataset(
        config.n, config.set_params, config.data_gen, config.data_params,
    )




