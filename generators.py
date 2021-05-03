import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
import warnings
warnings.filterwarnings('ignore')


class DataGen3D:
    def __init__(self, batch_size, labels, ch0,
                 ch1=None, ch2=None, rand=True, out_size=None):
        self.labels = labels
        self.ch_arr = np.rollaxis(np.array([x for x in [ch0, ch1, ch2]
                                            if x is not None]), 0, 5)
        self.batch_size = batch_size
        self.rand = rand
        self.out_size = out_size
        self._index_list = []

    def __iter__(self):
        return self

    def __next__(self):
        return self._group_data()

    def __call__(self):
        for _ in range(2**32-1):
            yield self._group_data()

    def _index_tracker(self):
        if not self._index_list:
            self._index_list = list(range(self.labels.shape[0]))
        if self.rand:
            random.shuffle(self._index_list)
        return self._index_list.pop(0)

    def _crop(self, array):
        in_size = self.ch_arr.shape[1:-1]
        diffs = [a-b for a, b in zip(in_size, self.out_size)]
        empty = [slice(None)]  # Used to pad the slice tuple
        slices = empty + [slice(d//2, d-d//2) for d in diffs] + empty
        return array[tuple(slices)]

    def _get_data(self):
        index = self._index_tracker()
        return self.ch_arr[index], self.labels[index]

    def _group_data(self):
        train_array = np.zeros((self.batch_size,
                                *self.ch_arr.shape[1:]))
        truth_array = np.zeros((self.batch_size,
                                *self.labels.shape[1:]))
        for i in range(self.batch_size):
            train_array[i], truth_array[i] = self._get_data()
        if not self.out_size:
            return train_array, truth_array
        return self._crop(train_array), self._crop(truth_array)


class AugmentedDataGen3D(DataGen3D):
    def __init__(self, batch_size, labels, ch0,
                 ch1=None, ch2=None, out_size=None, **kwargs):
        super().__init__(batch_size, labels, ch0, ch1, ch2)
        self.out_size = out_size
        self._IDG = IDG(**kwargs)
        self._seed = random.randint(0, 2**32-1)
        self._truth_IDG = self._init_generator(self.labels)
        self._train_IDG = self._init_generator(self.ch_arr)

    def _init_generator(self, arr):
        indx = range(arr.shape[-1])
        flow_dict = {'seed': self._seed,
                     'batch_size': 1}
        return zip(*[self._IDG.flow(x=arr[..., i], **flow_dict) for i in indx])

    def _get_data(self):
        train = np.rollaxis(np.array(next(self._train_IDG))[:, 0], 0, 4)
        test = np.rollaxis(np.array(next(self._truth_IDG))[:, 0], 0, 4)
        return train, test


if __name__ == "__main__":
    img0 = np.zeros((10, 270, 270, 75))
    lbl0 = np.zeros((10, 7, 270, 270, 75))

    auggen = AugmentedDataGen3D(2, lbl0, img0, img0, img0, **{'rotation_range': 45})(max=2**31)

    for _ in range(100):
        t0, l0 = next(auggen)

    print(t0.shape, l0.shape)

    tf_img_shape = [None, t0.shape]

    ds_fgen = tf.data.Dataset.from_generator(auggen, output_types=(tf.float32, tf.float32),
                                             output_shapes=(tf.TensorShape(img0.shape),
                                                            tf.TensorShape(lbl0.shape)))
'''
For this to intergrate into the library, we would want to say:

generator = cohort.generator.tensorflow(*args, **kwargs) -> formatted for tensorflow
generator = cohort.generator.pytorch(*args, **kwargs) -> formatted for pytorch
generator = cohort.generator(*args, **kwargs) -> non-optimized generator function

This will create a generator, like above and then will format it into tensorflow or pytorch

required parameters:
    n_channels

optional parameters:
    shuffle: bool. Default = True
    augmentations: dict. Default = None
    filter_: function. Default = None
    iterator: function. Default = None
    modalities: list. Default = None
'''

class generator:
    def __init__(self, n_channels: int, augmentations: dict = None, shuffle: bool = False, filter_: object = None,
                 modalities: list = None, iterator: object = None):
        self.n_channels = n_channels
        self.augmentations = augmentations
        self.shuffle = shuffle
        self.filter_ = filter_
        self.modalities = modalities
        self.iterator = iterator