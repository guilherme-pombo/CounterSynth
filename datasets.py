import numpy as np

from torch.utils.data import Dataset
from monai.transforms import *
from monai import transforms as t
from monai.data.dataloader import DataLoader as MonaiDataLoader


class ExampleDataset(Dataset):
    '''
    This is an example Dataset class that you should adapt to use your own data
    here we produce just random tensors, to illustrate the augmentation pipeline and how it would fit in
    the train_model.py code. For the paper we used UK Biobank data which is freely avaible, but not shareable
    '''

    def __init__(self, subset, aug_p=0.8):
        '''
        Recommended to add oversampling to ensure the generator learns the task properly
        :param aug_p: The augmentation probability. Due to the difficulty of the task and the lack of brain imaging
        available it is recommended that this be set to a fairly high value.
        '''

        self.aug_p = aug_p
        self.subset = subset
        self.test_transforms = t.Compose([t.NormalizeIntensity()])
        self.monai_transforms = [ToTensor(),
                                 t.Rand3DElastic(sigma_range=(0.01, 1), magnitude_range=(0, 1),
                                                 prob=aug_p, rotate_range=(0.18, 0.18, 0.18),
                                                 translate_range=(4, 4, 4), scale_range=(0.10, 0.10, 0.10),
                                                 spatial_size=None, padding_mode="border", as_tensor_output=False),
                                 t.RandHistogramShift(num_control_points=(5, 15), prob=aug_p),
                                 t.RandAdjustContrast(prob=aug_p),
                                 t.RandGaussianNoise(prob=aug_p),
                                 t.NormalizeIntensity()]
        self.monai_transforms = t.Compose(self.monai_transforms)

        # Where your brains would go
        self.brain_volumes = np.random.random((1000, 64, 64, 64))
        # Classification labels
        self.class_labels = np.random.random((1000, 1))
        # Regression labels
        self.reg_labels = np.random.random((1000, 1))

    def __len__(self):
        return len(self.brain_volumes)

    def __getitem__(self, index):
        image = np.expand_dims(self.brain_volumes[index], axis=0)

        if self.subset == 'train' and self.aug_p > 0:
            image = self.monai_transforms(image)
        else:
            image = self.test_transforms(image)

        return image, self.class_labels[index], self.reg_labels[index]


class MultiEpochsDataLoader(MonaiDataLoader):
    '''
    Override the default dataloader so that it doesn't spawn loads of processes at the start of every epoch
    This saves a huge amount of time for 3D
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Nicer for tqdm
    Adapted from: https://github.com/pytorch/examples/blob/master/imagenet/train.py
    """
    def __init__(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.
        self.tracker = []

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def track(self):
        """
        @rtype: object
        """
        self.tracker.append(self.avg)

    def save(self, fn):
        np.save(fn, np.array(self.tracker))
