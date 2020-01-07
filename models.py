# models.py
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Lambda, Input, Dense, Flatten, Reshape, ThresholdedReLU
from tensorflow.keras.layers import Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy, mean_squared_error

from time import gmtime, strftime

class VAE():
    '''
    A class representing the variational autoencoder used to encode the bar arrays.
    The VAE encodes a batch of bars of shape input_shape into a vector with latent_dim dimensions.

    Parameters
    ----------
    latent_dim : int
        The dimensionality of the latent space.
    input_shape : list of int with lenght 2
        The shape of the input space without batch size.
        The first element specifies the number of timesteps in a bar, the second one the number of instruments.
    weights : str, optional
        Path to a weights file that was previously saved by Keras.
        If specified, the weights are loaded. If None, the weights are initialized by Keras.
    debug : bool, optional
        If true print debug information, such as model summaries during class construction.
    '''
    def __init__(self, latent_dim, input_shape, weights=None, debug=True):
        # set main class attributes
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.debug = debug
        self.weightdir = 'weights/'
        self.callbacks = [tf.keras.callbacks.TensorBoard(log_dir='./logs')] # saves the necassary data to be viewed by the Tensorboard application

        # create input tensors and instantiate the encoder and decoder models
        bar_input = Input(shape = self.input_shape, name='encoder_input')
        latent_input = Input(shape = self.latent_dim, name = 'latent_input')
        self.encoder = self.make_encoder(bar_input)
        self.decoder = self.make_decoder(latent_input)

        # create the output tensors
        encoder_output = self.encoder(bar_input)
        vae_output = self.decoder(encoder_output[2])

        # instantiate the VAE model
        self.VAE = Model(bar_input, vae_output, name='VAE_DNN')

        # calcuate the loss functions and add them to the model
        z_mean, z_log_var, z = encoder_output
        kl_loss = -.5 * tf.math.reduce_sum(1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var), axis = -1)
        recon_loss = mean_squared_error(tf.reshape(bar_input, [-1]), tf.reshape(vae_output, [-1]))
        recon_loss *= np.prod(self.input_shape, dtype = float)
        vae_loss = tf.math.reduce_mean(0.1 * kl_loss + recon_loss)
        self.VAE.add_loss(vae_loss)
        self.VAE.add_metric(recon_loss, name = 'recon_loss', aggregation='mean') # add the reconstruction loss as an additional viewable metric for performance analysis

        # compile the model and load the weights if specified
        self.VAE.compile(optimizer='adam')
        if self.debug:
            self.VAE.summary()
        if weights: self.VAE.load_weights(weights)

    def make_encoder(self, bar_input, n_conv = 4):
        '''
        Helper function to instatiate the encoder model.

        Parameters
        ----------
        bar_input : Tensorflow symbolic tensor
            The symbolic tensor which represents the encoder input.
        n_conv : int, optional
            The number of convolution and pooling layers applied.

        Returns:
        --------
        Keras model
            The created encoder model.
        '''

        x = bar_input
        # add the specified number of convolution/pooling layers
        for i in range(n_conv):
          x = Conv1D(64 * 2 ** i, 3, activation = 'relu', padding = 'valid') (x)
          x = MaxPooling1D(2) (x)
        x = Flatten(name = 'flatten')(x)
        #x = Dense(self.latent_dim * 2, activation = 'relu') (x) # intermediate dense layer

        z_mean = Dense(self.latent_dim, name = 'z_mean') (x)
        z_log_var = Dense(self.latent_dim, name = 'z_log_var') (x)
        # function to sample from a standard normal distribution, which is wrapped as a Lambda layer
        normal_sample_f = lambda y : tf.random.normal(tf.shape(y))
        eps = Lambda(normal_sample_f) (z_log_var)
        # Reparamerisation trick, rescale the sampled value to the actual distribution
        z = z_mean + tf.math.exp(0.5 * z_log_var) * eps

        # model returns the mean and logartihm of the variance as well as the sampled value
        encoder = Model(bar_input, [z_mean, z_log_var, z], name='encoder')
        if self.debug: encoder.summary()
        return encoder

    def make_decoder(self, latent_input, n_deconv = 4):
        '''
        Helper function to instatiate the decoder model.

        Parameters
        ----------
        latent_input : Tensorflow symbolic tensor
            The symbolic tensor which represents the latent input.
        n_deconv : int, optional
            The number of upsampling and convolution layers applied.

        Returns:
        --------
        Keras model
            The created encoder model.
        '''
        x = latent_input
        #x = Dense(self.latent_dim * 2, activation = 'relu') (x)
        x = Dense(4 * 128, activation = 'relu') (x)
        x = Reshape((4, 128)) (x)
        for i in range(n_deconv - 1):
          x = UpSampling1D(2) (x)
          x = Conv1D(64 * 2, 3, activation = 'relu', padding = 'same') (x)
        x = UpSampling1D(2) (x)
        x = Conv1D(22, 5, activation = 'relu', padding = 'same') (x)

        decoder = Model(latent_input, x, name='decoder')
        if self.debug: decoder.summary()
        return decoder

    def train(self, train_data, epochs, validation_split=0.0, **kwargs):
        '''
        Train the VAE model on a given dataset for a number of epochs.
        The weights are saved in the weightdir folder with a timestamp after training is completed.

        Parameters
        ----------
        train_data : array_like
            An array with shape (batch, *input_shape) which contains data to train the VAE on.
        epochs : int
            The number of epochs to train the model.
        validation_split : float, optional
            A number in the range [0,1].
            It corresponds to the fraction of the samples withheld from the training data set to use for validation purposes.
        **kwargs
            Further keyword arguments to be passed to the train method of the VAE.
        '''
        history = self.VAE.fit(train_data, epochs=epochs, validation_split = validation_split, callbacks = self.callbacks, **kwargs)
        name = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
        self.VAE.save_weights(self.weightdir + name)
        return history

    def interpolate(self, bar1, bar2, n_steps = 10):
        '''
        Interpolate between the specified bars.
        To achive this, each bar is encoded as a latent vector. Between those latent vectors, the specified number of intemediate vectors are generated.
        The entire sequence is then decoded into a sequence of bars again.

        Parameters
        ----------
        bar1, bar2 : array_like
            Arrays of shape input_shape which represent the bars to be interpolated.
        n_steps : int, optional
            The number of steps in the intepolqation inculding the two original bars.

        Returns
        -------
        array_like
            Array of shape (n_steps, *input:_shape) representing the generated sequence.
        '''
        lv1 = self.encoder.predict(bar1[None,:,:])[2]
        lv2 = self.encoder.predict(bar2[None,:,:])[2]

        lvs = np.concatenate([lv1 + (lv2-lv1)*i/(n_steps - 1) for i in range(n_steps)])
        bars_pred = self.decoder.predict(lvs)

        return bars_pred

    def sample(self, bar, n_samples):
        '''
        Sample from the distribtion of a given input bar.

        Parameters
        ----------
        bar : array_like
            Array of shape input_shape around which the distribution is to be sampled.

        Return
        ------
        array_like
            Array of shape (n_samples, *input_shape) which contians the samples.
        '''
        return self.VAE.predict(np.array([bar]*n_samples))
