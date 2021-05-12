import numpy as np
import tensorflow as tf
from helper_funcs import *

# data generator to load labels and mfcc features
class data_generator():
    """ For 2 input model """
    def __init__(self, speaker_map, seg_loader, M=8, batch_size=256, steps_per_epoch=20, infinite=False):
        """ seg_loader: function that takes segment array and returns features (list of array shape T X D)
            batch_size: power of 2,  
            M: should divide batch_size, 
            steps_per_epoch: number of batches per epoch"""
        self.data = speaker_map
        self.spk_index = {speaker: i for i, speaker in enumerate(self.data.keys())}
        self.M = min(M, len(self.spk_index))
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.infinite_generation = infinite
        self.continue_generation = True
        self.count = 0
        self.seg_loader = seg_loader

        self.D = self._make_batch()['feats'].shape[-1]

    def _make_batch(self):
        batch = {'feats':[], 'labels':[]}
        # sample M speakers
        speakers = np.random.choice([*self.data.keys()], size= self.M, replace=False)
        # sample batch_size/M segments
        for speaker in speakers:
            spk_segments = np.random.choice(range(len(self.data[speaker])), self.batch_size//self.M, replace=False) 
            spk_segments = [self.data[speaker][i] for i in spk_segments]
            spk_feats = self.seg_loader(spk_segments)
            batch['labels'].extend([self.spk_index[speaker]]*len(spk_segments))
            batch['feats'].extend(spk_feats)
        # print([f.shape for f in batch['feats']])
        batch['feats']  = np.stack(batch['feats'])
        batch['labels'] = np.asarray(batch['labels'])
        return batch

    def dict_generator(self):
        """ yeilds a dict {"feats":... , "labels":...} for each batch"""
        while self.continue_generation:
            yield self._make_batch()
            self.count += 1
            if not self.infinite_generation \
                and self.count == self.steps_per_epoch:
                self.continue_generation = False

    def usual_generator(self):
        """ yeilds (feats, labels) for each batch"""
        while self.continue_generation:
            batch = self._make_batch()
            yield batch['feats'], batch['labels']
            self.count += 1
            if not self.infinite_generation \
                and self.count == self.steps_per_epoch:
                self.continue_generation = False

    def reset(self):
        self.continue_generation = True
        self.count = 0

    def get_dict_tf_dataset(self):
        """ for 2 input model with inputs "feats", "labels" """
        tf_dataset = tf.data.Dataset.from_generator(
            generator = self.dict_generator, output_types= {'feats':tf.float32, 'labels':tf.int32}, 
            output_shapes= {'feats':tf.TensorShape((None, None, self.D)), 'labels':tf.TensorShape((None,))}
            ).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return tf_dataset

    def get_usual_tf_dataset(self):
        """ for single input model """
        tf_dataset = tf.data.Dataset.from_generator(
            generator = self.usual_generator, output_types= (tf.float32, tf.int32), 
            output_shapes= (tf.TensorShape((None, None, self.D)), tf.TensorShape((None,)))
            ).prefetch(tf.data.experimental.AUTOTUNE)
        return tf_dataset
