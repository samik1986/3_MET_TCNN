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
print model_stg1.summary()

x_train = np.load('targetImagesIITM.npy')
x_train = model_bcm.predict(x_train)
x_train = model_tm.predict(x_train)
y_train = np.load('targetLabelIITM.npy')

print len(x_train)

sz_1 = len(model_bcm.layers)
print sz_1

del model_tm, model_bcm
# input_dim =  [12,12,20]
# sess = tf.InteractiveSession()


def create_network(input_dim):

    input_source = Input(input_dim)


    sz_3 = len(model_stg1.layers)
    print sz_3

    x = model_stg1.layers[sz_3-sz_1+1](input_source)

    for i in range(sz_3-sz_1-2):
        x = model_stg1.layers[i+sz_1+1](x)

    classifier1 = model_stg1.layers[sz_3-1](x)




    final = Model(inputs=[input_source],
                  outputs=[classifier1])

    sz_4= len(final.layers)
    for i in range(sz_4-1):
        w = model_stg1.layers[sz_3-sz_1+i+1].get_weights()
        final.layers[i+1].set_weights(w)

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.7, nesterov=True)

    final.compile(loss='categorical_crossentropy', optimizer=sgd)

    return final


model_stg3=create_network([12,12,20])
print model_stg3.summary()



filepath="models/stg3_ckpt{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')


model_stg3.fit(x_train, y_train,
          batch_size=200, epochs=100000,
          callbacks=[TensorBoard(log_dir='models/',
                                 write_images=True, write_grads=True),
                     checkpoint])

