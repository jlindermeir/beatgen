import numpy as np

from tensorflow.keras.layers import Lambda, Input, Dense, Flatten, Reshape, GaussianNoise, Add, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.utils import plot_model
import tensorflow as tf

class VAE():
    def __init__(self, latent_dim, input_shape, weights=None, debug=True):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.debug = debug

        self.input = Input(shape=self.input_shape, name='encoder_input')
        self.latent_input = Input(shape=(self.latent_dim,), name = 'latent_input')
        self.make_encoder()
        self.make_decoder()

        encoder_output = self.encoder(self.input)
        vae_output = self.decoder(encoder_output[2])
        self.VAE = Model(self.input, vae_output, name='VAE_DNN')

        #recon_loss = binary_crossentropy(self.input, vae_output) * float(np.prod(self.input_shape))
        
        #@tf.function
        def KL_loss (mu, sigma):
            tf.math.reduce_sum(tf.math.log(sigma) + (1 + tf.math.square(mu)) / (2 * tf.math.square(sigma)) - .5, axis = -1)
        #kl_loss = tf.math.reduce_sum(kl_loss, axis=-1)
        #kl_loss *= -0.5
        loss = binary_crossentropy(self.input, vae_output)# + KL_loss(encoder_output[0], encoder_output[1])
        
        #self.VAE.add_loss(tf.math.reduce_mean(recon_loss) + kl_loss)

        self.VAE.compile(optimizer='adam', loss = 'binary_crossentropy')
        if self.debug:
            self.VAE.summary()

        if weights: self.VAE.load_weights(weights)

    def make_encoder(self):
        x = self.input
        x = Dense(128, activation = 'relu')(x)
        x = Flatten(name = 'flatten')(x)
        
        z_mean = Dense(self.latent_dim, name = 'z_mean') (x)
        z_std = Dense(self.latent_dim, name = 'z_log_var') (x)
        eps = GaussianNoise(1, name = 'normal_sample') (z_std)
        normal_sample = Multiply() ([eps, z_std])
        z = Add(name = 'z')([z_mean, normal_sample])
        
        encoder = Model(self.input, [z_mean, z_std, z], name='encoder')
        if self.debug: encoder.summary()
        self.encoder = encoder

    def make_decoder(self):
        x = self.latent_input
        x = Dense(128, activation = 'relu') (x)
        x = Dense(tf.math.reduce_prod(self.input_shape), activation = 'sigmoid') (x)
        x = Reshape(self.input_shape) (x)

        decoder = Model(self.latent_input, x, name='decoder')
        if self.debug: decoder.summary()
        self.decoder = decoder

    def train(self, train_data, validation_data, epochs, batchsize):
        self.VAE.fit(train_data, epochs=epochs, batch_size = batchsize, validation_data=validation_data)
        self.VAE.save_weights('pokevae.h5') 
