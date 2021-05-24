# This is a sample Python script.
import os
import random
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt

from mdutils.mdutils import MdUtils

sns.set()

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)


class PlotLosses(K.callbacks.Callback):
    def __init__(self):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.EMSE = []
        self.KLD = []
        self.fig = None
        self.logs = []

    def on_train_begin(self, logs={}):
        self.fig = plt.figure()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        #self.EMSE.append(logs.get('val_EMSE'))
        self.EMSE.append(logs.get('val_mean_squared_error'))
        self.KLD.append(logs.get('val_QKLD'))
        self.i += 1

        # clear_output(wait=True) # To Use in Notebooks
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(self.x, self.losses, '-o', label="loss")
        plt.plot(self.x, self.val_losses, '-o', label="val_loss")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(self.x, self.EMSE, '-o', label="val_MSE")
        plt.plot(self.x, self.KLD, '-o', label="val_KLdiv")
        plt.legend()
        plt.show()

    def on_train_end(self, logs={}):
        #plt.figure(figsize=(10, 20))
        plt.subplot(2, 1, 1)
        plt.title('Loss Plot')
        plt.plot(self.x, self.losses, '-o', label="loss")
        plt.plot(self.x, self.val_losses, '-o', label="val_loss")
        plt.ylabel('Epoch')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.title('MSE and KL Divergence')
        plt.plot(self.x, self.EMSE, '-o', label="val_MSE")
        plt.plot(self.x, self.KLD, '-o', label="val_KLdiv")
        plt.ylabel('Epoch')
        plt.legend()
        plt.savefig('20210520_loss_plot.png')


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
        #
        self.encoder.add(K.layers.Flatten())
        self.encoder.add(K.layers.Dense(14*14, activation='relu'))

        self.decoder = K.Sequential()
        #
        self.decoder.add(K.layers.Dense(14*14, activation='sigmoid'))
        self.decoder.add(K.layers.Reshape((14, 14, 1)))
        #
        self.decoder.add(K.layers.Conv2DTranspose(self.hidden_dim, kernel_size=3, padding='same', strides=1))
        self.decoder.add(K.layers.LeakyReLU())
        for _ in range(self.downsample):
            self.decoder.add(K.layers.Conv2DTranspose(self.hidden_dim, kernel_size=3, padding='same', strides=1))
            self.decoder.add(K.layers.LeakyReLU())
            self.decoder.add(
                K.layers.Conv2DTranspose(self.hidden_dim, kernel_size=3, padding='same', output_padding=1, strides=2))
            self.decoder.add(K.layers.LeakyReLU())
        self.decoder.add(K.layers.Conv2DTranspose(1, kernel_size=3, padding='same', strides=1))
        self.decoder.add(K.layers.LeakyReLU())

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def kl_divergence(rho, rho_hat):
    return tf.abs(rho * tf.math.log(rho) - rho * tf.math.log(tf.abs(rho_hat)) + (1 - rho) * tf.math.log(1 - rho) - (
                1 - rho) * tf.math.log(tf.abs(1 - rho_hat)))

def EMSE(y_true, y_pred):
    diff = y_true - y_pred
    return tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=[1, 2, 3]))

def KLD(y_true, ae):
    z = ae.encoder(y_true)
    #rho_hat = tf.reduce_mean(z, axis=[1, 2, 3])
    rho_hat = tf.reduce_mean(z, axis=1)
    kl = kl_divergence(0.05, rho_hat)
    cost = tf.reduce_sum(kl)
    return cost

def QKLD(y_true, y_pred):
    return KLD(y_true, autoencoder)

mse = K.losses.MeanSquaredError()

@tf.function
def Loss(y_true, y_pred):
    #a = EMSE(y_true, y_pred)
    a = mse(y_true, y_pred)
    #b = KLD(y_true, autoencoder)

    #cost = 0.5 * a #+ 3 * b
    cost = a #+ b
    return cost


if __name__ == '__main__':
    #tf.config.run_functions_eagerly(True)

    mdFile = MdUtils(file_name='Report_20210520', title='ARoundVision Meeting')
    mdFile.new_header(level=1, title='AutoEncoder with KL Divergence as Sparsity Constraint')

    plot_losses = PlotLosses()

    (x_train, _), (x_test, _) = K.datasets.fashion_mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = tf.expand_dims(x_train, 3)
    x_test  = tf.expand_dims(x_test, 3)

    autoencoder = ConvAutoEncoder(hidden_dim=28, latent_dim=1, downsample=1)

    lr_schedule = K.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = K.optimizers.Adam(learning_rate=lr_schedule)

    #autoencoder.compile(optimizer=optimizer, loss=Loss, metrics=[EMSE, QKLD])
    #autoencoder.compile(optimizer=optimizer, loss=K.losses.MeanSquaredError(), metrics=[mse, QKLD])
    autoencoder.compile(optimizer='adam', loss=Loss, metrics=[mse, QKLD])

    autoencoder.fit(x_train, x_train,
                    batch_size=512,
                    epochs=20,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[plot_losses])

    encoded_imgs = autoencoder.encoder(x_test).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

    table = ['Image', 'PSNR (dB)', 'Sparcity (% of zeros)']
    n = 10
    plt.figure(figsize=(12, 8))
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(x_test[i])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        plt.title("reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + n*2)
        z = encoded_imgs[i].reshape((14, 14))
        plt.imshow(z)
        plt.title("encoded")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        psnr = float(tf.image.psnr(tf.expand_dims(x_test[i], 0), tf.expand_dims([decoded_imgs[i]], 0), max_val=1.0))
        spar = (encoded_imgs[i].size - np.count_nonzero(encoded_imgs[i])) / encoded_imgs[i].size * 100
        table.extend([str(i), str(round(psnr, 2)), str(round(spar, 2))])
        print('PSNR: \t {} \t \t Sparsity: \t {}'.format(round(psnr, 2), round(spar, 2)))
    plt.tight_layout()
    fig = plt.gcf()
    fig.show()
    fig.savefig('20210520_orig_rec_eno.png')

    mdFile.new_line()
    mdFile.new_table(columns=3, rows=n+1, text=table, text_align='center')
    mdFile.new_line(mdFile.new_inline_image(text='Loss Plot', path='./20210520_loss_plot.png'))
    mdFile.new_line(mdFile.new_inline_image(text='Original, Reconstructed, Encoded', path='./20210520_orig_rec_eno.png'))
    mdFile.create_md_file()
