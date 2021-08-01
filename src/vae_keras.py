from tensorflow.keras.layers import Input, Flatten, BatchNormalization, LeakyReLU, Dropout, Reshape, Activation
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import KLDivergence, MeanSquaredError
from tensorflow.keras import backend as K
import tensorflow as tf

from src.utils import plot_images


def BAD(x):
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(rate=0.25)(x)
    return x


# Generate encoder model
input_dim = (28, 28, 1)
encoder_input = Input(shape=input_dim, name='encoder_input')
layer = encoder_input
layer = Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(layer)
layer = BAD(layer)
layer = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(layer)
layer = BAD(layer)
layer = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(layer)
layer = BAD(layer)
layer = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(layer)
layer = BAD(layer)
top_conv = layer
layer = Flatten()(top_conv)

z_dim = 2
mu = Dense(z_dim, name='mu')(layer)
log_var = Dense(z_dim, name='log_var')(layer)
epsilon = tf.random.normal(shape=K.shape(mu), mean=0., stddev=1.)
encoder_output = mu + tf.exp(log_var / 2) * epsilon
encoder_model = Model(encoder_input, encoder_output)

# Generate decoder
decoder_input = Input(shape=(z_dim,))
layer = Dense(units=tf.reduce_prod(top_conv.get_shape()[1:]))(decoder_input)
layer = Reshape(top_conv.get_shape()[1:])(layer)

layer = Conv2DTranspose(filters=64, kernel_size=3, strides=1, padding='same')(layer)
layer = BAD(layer)
layer = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same')(layer)
layer = BAD(layer)
layer = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same')(layer)
layer = BAD(layer)
layer = Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')(layer)
layer = Activation('sigmoid')(layer)
decoder_output = layer
decoder_model = Model(decoder_input, decoder_output)

# Generate variational auto-encoder model
vae_model = Model(encoder_input, decoder_model(encoder_output))


def vae_mse_loss(y_true, y_pred):
    mse_factor = 1000
    return MeanSquaredError()(y_true, y_pred) * mse_factor


def vae_kl_loss(y_true, y_pred):
    return -0.5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=1)


def vae_loss(y_true, y_pred):
    return vae_kl_loss(y_true, y_pred) + vae_mse_loss(y_true, y_pred)


opt = Adam()
vae_model.compile(opt, loss=vae_loss, metrics=[vae_mse_loss, vae_kl_loss])

# load dataset
(train_xs, train_ys), (test_xs, test_ys) = tf.keras.datasets.mnist.load_data()
train_xs = train_xs[..., None] / 255.
test_xs = test_xs[..., None] / 255.

sample_imgs = train_xs[:25]
pred = vae_model.predict(sample_imgs)[..., 0]

# plot_images(sample_imgs)
# plot_images(pred)


# encoder_model.compile('adam', loss=vae_kl_loss)
# encoder_model.evaluate(x=train_xs)

vae_model.compile('adam', loss=vae_loss)
vae_model.train(x=train_xs)
