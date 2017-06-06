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

x_test = np.load('testImagesIITM.npy')
y_test = np.load('testLabelIITM.npy')

model_bcm = load_model('bcm.hdf5')
model_tm = load_model('tm.hdf5')
model_lm = load_model('lm.hdf5')

x_pred = model_bcm.predict(x_test)
x_pred = model_tm.predict(x_pred)
y_pred = model_lm.predict(x_pred)


