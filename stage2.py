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

model_stg1 = load_model('models/stg1_ckpt760.hdf5')

model_bcm = Model(inputs=model_stg1.input,
                  outputs=model_stg1.get_layer('max_pooling2d_2').output)

model_bcm.save('models/bcm')

print model_bcm.summary()

x_train = np.load('ae_gxTrain.npy')
print 'Read Gallery'
x_train = model_bcm.predict(x_train)
print 'Gallery Features'
x_aux_train = np.load('ae_pxTrain.npy')
print 'Read Probe'
x_aux_train = model_bcm.predict(x_aux_train)
print 'Probe Features'


input_dim =  [100,100,20]

sess = tf.InteractiveSession()

def create_network(input_dim):
    input_target = Input(input_dim)

    #---Autoencoder----
    x = Conv2D(15, (3, 3), activation='relu', padding='same')(input_target)
    x1 = Conv2D(15, (3, 3), activation='relu', padding='same')(x)
    x2 = MaxPooling2D((2, 2), padding='same')(x1)
    x3 = Conv2D(10, (3, 3), activation='relu', padding='same')(x2)
    x4 = BatchNormalization()(x3)
    x5 = MaxPooling2D((2, 2), padding='same')(x4)
    shape1 = K.int_shape(x5)
    # print shape1[0]
    x6 = Flatten()(x5)
    shape2 = K.int_shape(x6)
    # print shape2[0]
    x7 = Dense(2048,activation='relu')(x6)
    x71 = Dense(512,activation='relu')(x7)

    encoded = Dropout(0.5)(x71)


    # print encoded
    x8 = Dense(512,activation='relu')(encoded)
    x9 = Dense(2048,activation='relu')(x8)
    x10 = Dense(shape2[1],activation='relu')(x9)
    x11 = Reshape([shape1[1],shape1[2],shape1[3]])(x10)
    x12 = UpSampling2D((2, 2))(x11)
    x13 = BatchNormalization()(x12)
    x14 = Conv2D(15, (3, 3), activation='relu', padding='same')(x13)
    x15 = UpSampling2D((2, 2))(x14)
    x16 = Conv2D(15, (3, 3), activation='relu', padding='same')(x15)
    decoded = Conv2D(20, (3, 3), activation='relu', padding='same')(x16)
    # decoded = Conv2D(10, (3, 3), activation='sigmoid', padding='same')(x17)
    # print K.int_shape(decoded)
    # print input_source


    final = Model(inputs=input_target,
                  outputs=decoded)

    return final


model = create_network([100, 100, 3])
# model.save("model.h5",overwrite=True)
print(model.summary())


# datagen = ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True)

adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-05)
model.compile(loss='kullback_leibler_divergence',
              optimizer=adam)

# datagen.fit(x_aux_train)
# print 'D'
filepath="models/stg2_ckpt{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# model.fit_generator(datagen.flow(x_aux_train, x_train, batch_size=200),
#                     steps_per_epoch=len(x_aux_train) / 200, epochs=10000,
#                     callbacks=[TensorBoard(log_dir='models/',
#                                  write_images=True, write_grads=True),checkpoint])

model.fit(x_aux_train,
          x_train,
          batch_size=200, epochs=100000,
          callbacks=[TensorBoard(log_dir='models/',
                                 write_images=True, write_grads=True),
                     checkpoint])

model.save("model.h5",overwrite=True)

