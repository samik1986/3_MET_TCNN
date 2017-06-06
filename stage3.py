import numpy as np
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.models
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, merge, UpSampling2D, Reshape, BatchNormalization
from keras.layers import Input, TimeDistributed
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras import backend as K
from keras.engine.topology import Layer
from tensorflow.python.ops import array_ops
from scipy.linalg._expm_frechet import vec
from tensorflow.python.framework import ops
from tensorflow.python.framework.op_def_library import _Flatten, _IsListValue
from keras.callbacks import TensorBoard, ModelCheckpoint
#
model_tm = load_model('models/stg2_ckpt711.hdf5')
model_bcm = load_model('models/bcm.hdf5')
# model_tm.trainable = False
# print model_bcm.summary()
model_stg1 = load_model('models/stg1_ckpt760.hdf5')
# print model_stg1.summary()

def create_network(input_dim):

    input_source = Input(input_dim)

    stg1_l = model_bcm(input_source)
    # stg1_l.trainable = False
    stg2_l = model_tm(stg1_l)
    # stg2_l.trainable = False

    # n_layers = model_stg1.__sizeof__()
    # print n_layers
    #
    # x = Conv2D(10, (3, 3), activation='relu', padding='same')(input_source)
    # x.trainable= False
    # x = MaxPooling2D((3, 3), padding='same')(x)
    # x.trainable = False
    # x = Conv2D(15, (3, 3), activation='relu', padding='same')(x)
    # x.trainable = False
    # x = BatchNormalization()(x)
    # x.trainable = False
    # x = Conv2D(20, (3, 3), activation='relu', padding='same')(x)
    # x.trainable = False
    # bcm = MaxPooling2D((3, 3), padding='same')(x)
    # bcm.trainable=False
    #
    # x = Conv2D(15, (3, 3), activation='relu', padding='same')(bcm)
    # x.trainable = False
    # x = Conv2D(15, (3, 3), activation='relu', padding='same')(x)
    # x.trainable = False
    # x = MaxPooling2D((2, 2), padding='same')(x)
    # x.trainable = False
    # x = Conv2D(10, (3, 3), activation='relu', padding='same')(x)
    # x.trainable = False
    # x = BatchNormalization()(x)
    # x.trainable = False
    # x = MaxPooling2D((2, 2), padding='same')(x)
    # x.trainable = False
    # shape1 = K.int_shape(x)
    # # print shape1[0]
    # x = Flatten()(x)
    # shape2 = K.int_shape(x)
    # # print shape2[0]
    # x = Dense(2048, activation='relu')(x)
    # x.trainable = False
    # x = Dense(512, activation='relu')(x)
    # x.trainable = False
    # encoded = Dropout(0.25)(x)
    # encoded.trainable = False
    #
    # # print encoded
    # x = Dense(512, activation='relu')(encoded)
    # x.trainable = False
    # x = Dense(2048, activation='relu')(x)
    # x.trainable = False
    # x = Dense(shape2[1], activation='relu')(x)
    # x.trainable = False
    # x = Reshape([shape1[1], shape1[2], shape1[3]])(x)
    # x.trainable = False
    # x = UpSampling2D((2, 2))(x)
    # x.trainable = False
    # x = BatchNormalization()(x)
    # x.trainable = False
    # x = Conv2D(15, (3, 3), activation='relu', padding='same')(x)
    # x.trainable = False
    # x = UpSampling2D((2, 2))(x)
    # x.trainable = False
    # x = Conv2D(15, (3, 3), activation='relu', padding='same')(x)
    # x.trainable = False
    # decoded = Conv2D(20, (3, 3), activation='relu', padding='same')(x)
    # decoded.trainable = False
    #
    #
    flat1 = model_stg1.get_layer('flatten_1')(stg2_l)
    y = model_stg1.get_layer('dense_1')(flat1)
    y = model_stg1.get_layer('dense_2')(y)
    y = model_stg1.get_layer('dropout_1')(y)
    lm = model_stg1.get_layer('dense_3')(y)
    classifier1 = model_stg1.get_layer('dense_4')(lm)
    #


    final = Model(inputs=[input_source],
                  outputs=[classifier1])
    # print len(final.layers)

    for layer in final.layers[:3]:
        layer.trainable = False

    return final

model_stg3=create_network([100,100,3])
print model_stg3.summary()
print model_stg3.get_config()