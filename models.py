from tensorflow.keras.layers import Lambda, Input, Dense, Flatten, Reshape, Conv2D, MaxPooling2D, ThresholdedReLU, Conv2DTranspose, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy, mean_squared_error

from time import gmtime, strftime

class VAE():
    def __init__(self, latent_dim, input_shape, weights=None, debug=True):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.debug = debug
        self.weightdir = 'weights/'
        self.callbacks = [tf.keras.callbacks.TensorBoard(log_dir='./logs')]

        bar_input = Input(shape = self.input_shape, name='encoder_input')
        latent_input = Input(shape = self.latent_dim, name = 'latent_input')
        self.encoder = self.make_encoder(bar_input)
        self.decoder = self.make_decoder(latent_input)

        encoder_output = self.encoder(bar_input)
        vae_output = self.decoder(encoder_output[2])
        
        self.VAE = Model(bar_input, vae_output, name='VAE_DNN')
        
        z_mean, z_log_var, z = encoder_output

        kl_loss = -.5 * tf.math.reduce_sum(1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var), axis = -1)
        
        recon_loss = mean_squared_error(tf.reshape(bar_input, [-1]), tf.reshape(vae_output, [-1]))
        recon_loss *= np.prod(self.input_shape, dtype = float)
        
        vae_loss = tf.math.reduce_mean(0.1 * kl_loss + recon_loss)
        self.VAE.add_loss(vae_loss)
        self.VAE.add_metric(recon_loss, name = 'recon_loss', aggregation='mean')
        
        self.VAE.compile(optimizer='adam')
        if self.debug:
            self.VAE.summary()
        if weights: self.VAE.load_weights(weights)

    def make_encoder(self, bar_input):
        n_conv = 4
        x = bar_input
        x = Reshape((*self.input_shape, 1)) (x)
        for _ in range(n_conv):
          x = Conv2D(32, (3, 1), activation = 'relu', padding = 'valid') (x)
          x = MaxPooling2D((2,1)) (x)
        x = Flatten(name = 'flatten')(x)
        x = Dense(self.latent_dim * 2, activation = 'relu') (x)
                
        z_mean = Dense(self.latent_dim, name = 'z_mean') (x)
        z_log_var = Dense(self.latent_dim, name = 'z_log_var') (x)
        normal_sample_f = lambda y : tf.random.normal(tf.shape(y))
        eps = Lambda(normal_sample_f) (z_log_var)
        z = z_mean + 0.5 * tf.math.exp(z_log_var) * eps
        
        encoder = Model(bar_input, [z_mean, z_log_var, z], name='encoder')
        if self.debug: encoder.summary()
        return encoder

    def make_decoder(self, latent_input):
        n_deconv = 4
        x = latent_input
        x = Dense(self.latent_dim * 2, activation = 'relu') (x)
        x = Dense(1408, activation = 'relu') (x)
        x = Reshape((2, 22, 32)) (x)
        for _ in range(n_deconv - 1):
          x = UpSampling2D((2,1)) (x)
          x = Conv2DTranspose(32, (3,1), activation = 'relu', padding = 'valid') (x)
        x = UpSampling2D((2,1)) (x)
        x = Conv2DTranspose(1, (5,1), activation = 'relu', padding = 'valid') (x)
        x = Reshape(self.input_shape) (x)

        decoder = Model(latent_input, x, name='decoder')
        if self.debug: decoder.summary()
        return decoder

    def train(self, train_data, epochs, batchsize, validation_split=0.0, **kwargs):
        history = self.VAE.fit(train_data, epochs=epochs, batch_size = batchsize, validation_split = validation_split, callbacks = self.callbacks, **kwargs)
        name = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
        self.VAE.save_weights(self.weightdir + name) 
        return history
