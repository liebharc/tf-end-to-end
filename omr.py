import tensorflow as tf
from tensorflow import keras
import numpy as np


class OMR_model(keras.Model):
    def __init__(self, model_params):
        model_params['vocabulary_size'] = len(open(model_params['voc_path'],'r').read().splitlines())

        super().__init__(self)
        self.model_params = model_params
        self.directory = model_params['corpus_path']

        width_reduction = 1
        height_reduction = 1
        
        # Set up the CNN
        filter_n = model_params['conv_filter_n'].split(',')
        kernels = model_params['conv_filter_size'][i].split(',')
        conv_pooling_size = model_params['conv_pooling_size'][i].split(',')
        for i in range(len(filter_n)):
            # Add the convolutional layer
            self.add(keras.layers.Conv2D(filters=filter_n[i],
                                         kernel_size=kernels[i].split('x'),
                                         padding='same',
                                         activation=None))
            self.add(keras.layers.BatchNormalization())
            self.add(keras.layers.LeakyReLU(alpha=model_params['leaky_relu_alpha']))
            self.add(keras.layers.MaxPooling2D(pool_size=conv_pooling_size[i].split('x'),
                                               strides=conv_pooling_size[i].split('x')))
            
            width_reduction = width_reduction * conv_pooling_size[i].split('x')[1]
            height_reduction = height_reduction * conv_pooling_size[i].split('x')[0]

        # Reshape the output of the CNN to a 2D tensor to feed into the RNN
        input_shape=(None, model_params['img_height'],
            model_params['img_width'],
            model_params['img_channels'])
        
        features = tf.compat.v1.transpose(x, perm=[2, 0, 3, 1])  # -> [width, batch, height, channels] (time_major=True)
        feature_dim = model_params['conv_filter_n'][-1] * (model_params['img_height'] / height_reduction)
        feature_width = input_shape[2] / width_reduction
        features = tf.compat.v1.reshape(features, tf.compat.v1.stack([tf.compat.v1.cast(feature_width,'int32'), input_shape[0], tf.compat.v1.cast(feature_dim,'int32')]))  # -> [width, batch, features]

        
        self.add(keras.layers.Reshape(target_shape=(feature_width, feature_dim * filter_n[-1])))
        # self.add(keras.layers.Dense(model_params['rnn_units'], activation='relu'))
        # self.add(keras.layers.Dropout(model_params['rnn_dropout']))
        # self.add(keras.layers.Reshape(target_shape=(model_params['img_height'] // height_reduction, model_params['img_width'] // width_reduction * filter_n[-1])))

        # Set up the RNN
        for i in range(model_params['rnn_layers']):
            self.add(keras.layers.Bidirectional(keras.layers.LSTM(model_params['rnn_units'],
                                                                  return_sequences=True,
                                                                  dropout=model_params['rnn_dropout'])))
        self.add(keras.layers.Dense(model_params['vocabulary_size'], activation='softmax'))
    
    def call(self, inputs, training=False):
        x = inputs
        for layer in self.layers:
            x = layer(x, training=training)
        
        return x
    
    
