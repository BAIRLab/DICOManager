import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
import warnings
from groupings import ReconstructedFile
warnings.filterwarnings('ignore')

'''
Workflow process:
- iterate through frame of reference groups
- filter appropriately
- crop and process if necessary
    - automatically crop / pad
    - we can also have a processing function input
- compile images into hd5f array or numpy array
    - dependent upon in_memory parameter
- yield the volumes
    - shuffle if desired
    - augment if specified
'''


class BasicGenerator:
    """
    This follows the given workflow:
    - iterate through the tree or iterator
    - for each volume, filter
    - process the volumes
    - compile into an array

    Then, when called:
    - yield data
    - if shuffle, track indicies
    """
    def __init__(self, n_channels: int, output_dims: tuple, train_filter: object,
                 augmentations: dict = None, shuffle: bool = False,
                 inclusion_filter: object = None, modalities: list = None,
                 iterator: object = None, in_memory: bool = False,
                 struct_names: list = None, processing_func: object = None):
        self.n_channels = n_channels
        self.output_dims = output_dims
        self.shuffle = shuffle
        self.train_filter = train_filter
        self.inclusion_filter = inclusion_filter
        self.modalities = modalities
        self.augmentations = augmentations
        self.iterator = iterator
        self.in_memory = in_memory
        self.struct_names = struct_names
        self._index_list = []
        if not self.iterator:
            self.iterator = self.iter_frames()

    def _yield_data(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        return self._group_data()

    def __call__(self):
        yield next(self)

    def _to_gen(self):
        while True:
            yield self._group_data()

    def _compile(self):
        for vol in self.iterator:
            if self._filter(vol):
                processed = self._process(vol)
                self._compile(processed)
        return True

    def _tree_iter(self):
        for frame in self.iterator:
            group = self._filter_frame(frame)
            train, truth = self._build_arrays(group)

    def _build_array(self, group):
        fill_array_train = np.zeros(self.output_dims)
        fill_array_truth = np.zeros(self.output_dims)

        for modality, volume in group.items():
            cropped = self._crop(volume)
            if self.inclusion_filter(modality):
                index = self._fill_index(modality, train=True)
                fill_array_train[index] = cropped
            else:
                index = self._fill_index(modality, train=False)
                fill_array_truth[index] = cropped
        return fill_array_train, fill_array_truth

    def _filter_frame(self, frame):
        vol_group = {}
        for vol in frame.iter_volumes():
            if type(vol) is ReconstructedFile:
                vol.load_array()
            if self.inclusion_filter:
                if not self.inclusion_filter(vol):
                    continue
            if self.modalities:
                if vol.Modality not in self.modalities:
                    continue
            if self.struct_names and vol.Modality == 'RTSTRUCT':
                temp = {x: y for x, y in vol.struct.items() if x in self.struct_names}
                vol_group.update({'RTSTRUCT': temp})
            else:
                vol_group.update({vol.Modality: vol.volumes})
        return vol_group

    def _process(self):
        if self.processing_func:
            array = self.processing_func(array)
        if array.shape != self.output_dims:
            array = self._crop(array)
        return array

    def _crop(self, array):
        in_size = self.ch_arr.shape[1:-1]
        diffs = [a-b for a, b in zip(in_size, self.output_dims)]
        empty = [slice(None)]  # Used to pad the slice tuple
        slices = empty + [slice(d//2, d-d//2) for d in diffs] + empty
        return array[tuple(slices)]

    def _index_tracker(self):
        if not self._index_list:
            n_groups = len(self.iterator)
            self._index_list = list(range(n_groups))
        if self.shuffle:
            random.shuffle(self._index_list)
        return self._index_list.pop(0)

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


class AugmentedGenerator(BasicGenerator):
    def __init__(self, n_channels: int, output_dims: tuple, augmentations: dict = None,
                 shuffle: bool = False, filter_: object = None, modalities: list = None,
                 iterator: object = None, in_memory: bool = False, struct_names: list = None):
        super().__init__(n_channels, output_dims, augmentations, shuffle, filter_,
                         modalities, iterator, in_memory, struct_names)
        self.out_size = output_dims
        self._IDG = IDG(self.augmentations)
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


def basic(n_channels: int, output_dims: tuple, augmentations: dict = None,
          shuffle: bool = False, filter_: object = None, modalities: list = None,
          iterator: object = None, in_memory: bool = False, struct_names: list = None):
    if augmentations:
        generator = AugmentedGenerator(n_channels, output_dims, augmentations, shuffle,
                                       filter_, modalities, iterator, in_memory, struct_names)
    else:
        generator = BasicGenerator(n_channels, output_dims, augmentations, shuffle,
                                   filter_, modalities, iterator, in_memory, struct_names)
    return generator


def pytorch(n_channels: int, output_dims: tuple, augmentations: dict = None,
            shuffle: bool = False, filter_: object = None, modalities: list = None,
            iterator: object = None, in_memory: bool = False, struct_names: list = None):
    if augmentations:
        generator = AugmentedGenerator(n_channels, output_dims, augmentations, shuffle,
                                       filter_, modalities, iterator, in_memory, struct_names)
    else:
        generator = BasicGenerator(n_channels, output_dims, augmentations, shuffle,
                                   filter_, modalities, iterator, in_memory, struct_names)
    return generator


def tensorflow(n_channels: int, output_dims: tuple, augmentations: dict = None,
               shuffle: bool = False, filter_: object = None, modalities: list = None,
               iterator: object = None, in_memory: bool = False, struct_names: list = None):
    if augmentations:
        generator = AugmentedGenerator(n_channels, output_dims, augmentations, shuffle,
                                       filter_, modalities, iterator, in_memory, struct_names)
    else:
        generator = BasicGenerator(n_channels, output_dims, augmentations, shuffle,
                                   filter_, modalities, iterator, in_memory, struct_names)

    tfgen = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.float32),
                                           output_shapes=(output_dims, output_dims))
    return tfgen
