# This is a sample Python script.
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt

from tensorflow import keras

class PlotLosses(K.callbacks.Callback):
    def __init__(self):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = None
        self.logs = []

    def on_train_begin(self, logs={}):
        self.fig = plt.figure()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        #clear_output(wait=True) # To Use in Notebooks
        plt.clf()
        plt.plot(self.x, self.losses, '-o', label="loss")
        plt.plot(self.x, self.val_losses, '-o', label="val_loss")
        plt.legend()
        plt.show()


class ConvAutoEncoder(K.Model):

    def __init__(self, hidden_dim=128, latent_dim=12, downsample=3):
        super(ConvAutoEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.downsample = downsample

        self.encoder = K.Sequential()
        self.encoder.add(K.layers.Conv2D(self.hidden_dim, kernel_size=3, padding='same', strides=1))
        self.encoder.add(K.layers.LeakyReLU())
        for _ in range(self.downsample):
            self.encoder.add(K.layers.Conv2D(self.hidden_dim, kernel_size=3, padding='same', strides=1))
            self.encoder.add(K.layers.LeakyReLU())
            self.encoder.add(K.layers.Conv2D(self.hidden_dim, kernel_size=3, padding='same', strides=2))
            self.encoder.add(K.layers.LeakyReLU())
        self.encoder.add(K.layers.Conv2D(self.latent_dim, kernel_size=3, padding='same', strides=1))
        self.encoder.add(K.layers.LeakyReLU())

        self.decoder = K.Sequential()
        self.decoder.add(K.layers.Conv2DTranspose(self.hidden_dim, kernel_size=3, padding='same', strides=1))
        self.decoder.add(K.layers.LeakyReLU())
        for _ in range(self.downsample):
            self.decoder.add(K.layers.Conv2DTranspose(self.hidden_dim, kernel_size=3, padding='same', strides=1))
            self.decoder.add(K.layers.LeakyReLU())
            self.decoder.add(K.layers.Conv2DTranspose(self.hidden_dim, kernel_size=3, padding='same', output_padding=1, strides=2))
            self.decoder.add(K.layers.LeakyReLU())
        self.decoder.add(K.layers.Conv2DTranspose(3, kernel_size=3, padding='same', strides=1))
        self.decoder.add(K.layers.LeakyReLU())

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def kl_divergence(rho, rho_hat):
    return rho * tf.math.log(rho) - rho * tf.math.log(rho_hat) + (1 - rho) * tf.math.log(1 - rho) - (1 - rho) * tf.math.log(1 - rho_hat)


@tf.function
def Loss(y_true, y_pred):
    z = autoencoder.encoder(y_true)
    rho_hat = tf.reduce_mean(z, axis=[1, 2, 3])
    kl = kl_divergence(0.05, rho_hat)

    diff = y_true - y_pred

    cost = 0.5 * tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=[1, 2, 3])) # + 3 * tf.reduce_sum(kl)
    return cost


if __name__ == '__main__':
    #tf.config.run_functions_eagerly(True)
    plot_losses = PlotLosses()

    (x_train, _), (x_test, _) = keras.datasets.cifar100.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test  = x_test.astype('float32') / 255.

    autoencoder = ConvAutoEncoder()

    autoencoder.compile(optimizer='adam', loss=Loss)

    autoencoder.fit(x_train, x_train,
                    epochs=10,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[plot_losses])

    encoded_imgs = autoencoder.encoder(x_test).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

    n = 10
    plt.figure(figsize=(30, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        plt.title("reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + n)
        z = encoded_imgs[i].reshape((8, 8))
        plt.imshow(z)
        plt.title("encoded")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        psnr = float(tf.image.psnr(tf.expand_dims(x_test[i], 0), tf.expand_dims([decoded_imgs[i]], 0), max_val=1.0))
        spar = (len(encoded_imgs[i]) - np.count_nonzero(encoded_imgs[i])) / len(encoded_imgs[i])*100
        print('PSNR: \t {} \t \t Sparsity: \t {}'.format(round(psnr, 2), round(spar, 2)))
    plt.show()


