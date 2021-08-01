from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense , BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf


def BAD(x):
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(rate=0.25)(x)
    return x


# generate model
input_dim = (28, 28, 1)
encoder_input = Input(shape=input_dim, name='encoder_input')

layer = encoder_input
layer = Conv2D(filters=32, kernel_size=5, strides=1, padding='same')(layer)
layer = BAD(layer)

layer = Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(layer)
layer = BAD(layer)

layer = Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(layer)
layer = BAD(layer)

layer = Flatten()(layer)

# generate code for normal distribution
z_dim = 2
mu = Dense(z_dim, name='mu')(layer)
log_var = Dense(z_dim, name='log_var')(layer)
mu_var_model = Model(encoder_input, (mu, log_var))

#
epsilon = tf.random.normal(shape=K.shape(mu), mean=0., stddev=1.)
code = mu + tf.exp(log_var / 2) * epsilon
