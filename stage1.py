import numpy as np
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.models
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import Model, load_model, save_model
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

x_train = np.load('trainImagesIITM.npy')
y_train = np.load('trainLabelIITM.npy')

print len(x_train)

input_dim =  [100,100,3]
sess = tf.InteractiveSession()

def create_network(input_dim):

    input_source = Input(input_dim)

    x = Conv2D(10, (3, 3), activation='relu', padding='same')(input_source)
    x1 = MaxPooling2D((3, 3), padding='same')(x)
    x2 = Conv2D(15, (3, 3), activation='relu', padding='same')(x1)
    x3 = BatchNormalization()(x2)
    x4 = Conv2D(20, (3, 3), activation='relu', padding='same')(x3)
    bcm = MaxPooling2D((3, 3), padding='same')(x4)

    flat1 = Flatten()(bcm)
    y = Dense(4096,activation='relu')(flat1)
    y1 = Dense(512, activation='relu')(y)
    y2 = Dropout(0.25)(y1)
    lm = Dense(100, activation='relu')(y2)
    classifier1 = Dense(51, activation='softmax')(lm)

    final = Model(inputs=[input_source],
                  outputs=[classifier1])



    return final

model = create_network([100, 100, 3])
print(model.summary())
# plot_model(model, to_file='model.png')

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.7, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(x_train)
filepath="models/stg1_ckpt{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
model.fit_generator(datagen.flow(x_train, y_train, batch_size=300),
                    steps_per_epoch=len(x_train) / 300, epochs=1000,
                    callbacks=[TensorBoard(log_dir='models/',
                                 write_images=True, write_grads=True),checkpoint])

model.save("model_stage1.hdf5",overwrite=True)
model.save_weights("model_stage1_weights.hdf5", overwrite=True)
