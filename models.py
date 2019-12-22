import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Lambda, Input, Dense, Flatten, Reshape, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy, mean_squared_error

from time import gmtime, strftime

class VAE():
    def __init__(self, latent_dim, input_shape, weights=None, debug=True):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.debug = debug
        self.weightdir = 'weights/'

        bar_input = Input(shape = self.input_shape, name='encoder_input')
        latent_input = Input(shape = self.latent_dim, name = 'latent_input')
        self.encoder = self.make_encoder(bar_input)
        self.decoder = self.make_decoder(latent_input)

        encoder_output = self.encoder(bar_input)
        vae_output = self.decoder(encoder_output[2])
        
        self.VAE = Model(bar_input, vae_output, name='VAE_DNN')
        
        z_mean, z_log_var, z = encoder_output

        kl_loss = tf.math.reduce_sum(1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var), axis = -1)
        kl_loss *= -.5
        
        recon_loss = mean_squared_error(tf.reshape(bar_input, [-1]), tf.reshape(vae_output, [-1]))
        recon_loss *= np.prod(self.input_shape, dtype = float)
        
        vae_loss = tf.math.reduce_mean(kl_loss + recon_loss)
        self.VAE.add_loss(vae_loss)

        self.VAE.compile(optimizer='adam')
        if self.debug:
            self.VAE.summary()

        if weights: self.VAE.load_weights(weights)

    def make_encoder(self, bar_input):
        x = bar_input
        x = Conv1D(128, 3, activation = 'relu') (x)
        x = Conv1D(128, 3, activation = 'relu') (x)
        x = Flatten(name = 'flatten')(x)
                
        z_mean = Dense(self.latent_dim, name = 'z_mean') (x)
        z_log_var = Dense(self.latent_dim, name = 'z_log_var') (x)
        normal_sample_f = lambda y : tf.random.normal(tf.shape(y))
        eps = Lambda(normal_sample_f) (z_log_var)
        z = z_mean + 0.5 * tf.math.exp(z_log_var) * eps
        
        encoder = Model(bar_input, [z_mean, z_log_var, z], name='encoder')
        if self.debug: encoder.summary()
        return encoder

    def make_decoder(self, latent_input):
        x = latent_input
        x = Dense(128, activation = 'relu') (x)
        x = Dense(tf.math.reduce_prod(self.input_shape), activation = 'sigmoid') (x)
        x = Reshape(self.input_shape) (x)

        decoder = Model(latent_input, x, name='decoder')
        if self.debug: decoder.summary()
        return decoder

    def train(self, train_data, epochs, batchsize, validation_split=0.0):
        history = self.VAE.fit(train_data, epochs=epochs, batch_size = batchsize, validation_split = validation_split)
        name = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
        self.VAE.save_weights(self.weightdir + name) 
        return history
