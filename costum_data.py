# This is a sample Python script.
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt

from tensorflow import keras


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

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

        # clear_output(wait=True) # To Use in Notebooks
        plt.clf()
        plt.plot(self.x, self.losses, '-o', label="loss")
        plt.plot(self.x, self.val_losses, '-o', label="val_loss")
        plt.legend()
        plt.show()


class Autoencoder(K.Model):

    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = K.Sequential([
            K.layers.Flatten(),
            K.layers.Flatten(),
            K.layers.Dense(latent_dim, activation='relu'),
        ])

        self.decoder = K.Sequential([
            K.layers.Dense(49152, activation='sigmoid'),
            K.layers.Reshape((128, 128, 3))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


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
            self.decoder.add(
                K.layers.Conv2DTranspose(self.hidden_dim, kernel_size=3, padding='same', output_padding=1, strides=2))
            self.decoder.add(K.layers.LeakyReLU())
        self.decoder.add(K.layers.Conv2DTranspose(3, kernel_size=3, padding='same', strides=1))
        self.decoder.add(K.layers.LeakyReLU())

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def Loss(y_true, y_pred, ae):
    # sl = ae.encoder(y_pred)
    sl = y_pred
    return tf.math.count_nonzero(sl) / (28 * 28)


@tf.function
def L1(y_true, y_pred):
    # return tf.math.count_nonzero(tf.math.subtract(y_true, y_pred)) / (28 * 28)
    # return tf.math.abs(tf.math.subtract(y_true, y_pred))
    aux = tf.math.abs(tf.math.subtract(y_true, y_pred))
    aux = tf.math.reduce_sum(aux, axis=(1, 2))
    # aux = tf.equal(aux, 0)
    # aux = tf.math.reduce_sum(aux)
    # aux = tf.math.count_nonzero(aux)
    aux = tf.cast(aux, tf.float32)
    # aux = tf.math.scalar_mul(aux, tf.convert_to_tensor(1/784, dtype=tf.float32))
    return aux


@tf.function
def L2(y_true, y_pred):
    mse = K.losses.mean_squared_error(y_true, y_pred)
    mse1 = K.losses.MeanSquaredError()
    mse2 = mse1(y_true, y_pred)
    aux = autoencoder.encoder(y_true)
    aux = tf.reshape(aux, (aux.shape[0], 8, 8))
    print(aux.numpy())
    # aux = tf.math.round(aux)
    # aux = tf.math.count_nonzero(aux)
    aux = tf.equal(aux, 0)
    aux = tf.math.count_nonzero(aux, axis=-1)
    aux = tf.cast(aux, tf.float32) / 100
    # aux = autoencoder.decoder(aux)/10
    # aux = tf.math.abs(tf.math.subtract(y_true, aux))
    print(aux.numpy())
    return aux


@tf.function
def L3(y_true, y_pred):
    aux = autoencoder.encoder(y_true)

    def enttrop(aux):
        _, _, count = tf.unique_with_counts(tf.constant(aux))
        prob = count / tf.reduce_sum(count)
        tf_res = -tf.reduce_sum(prob * tf.math.log(prob))
        tf_res = tf.cast(tf_res, tf.float32)
        return tf_res

    aux = tf.map_fn(enttrop, aux)
    return aux


def KLD(z):
    p = 0.05
    beta = 3
    # p_hat = tf.math.mean(z) # average over the batch samples
    p_hat = tf.reduce_mean(z, axis=1)
    # KLD = p*(K.log(p)-K.log(p_hat)) + (1-p)*(K.log(1-p)-K.log(1-p_hat))
    KLD = p * (tf.math.log(p / p_hat)) + (1 - p) * (tf.math.log(1 - p / 1 - p_hat))
    # return beta * tf.math.sum(KLD)
    # return beta * tf.reduce_sum
    l = beta * KLD
    print(l)
    return l


def kl_loss(z):
    def logfunc(x1, x2):
        return tf.multiply(x1, tf.math.log(tf.divide(x1, x2)))

    def kl_div(rho, rho_hat):
        t_num = tf.constant(1.) - rho
        t_den = tf.constant(1.) - rho_hat
        kl = logfunc(rho, rho_hat) + logfunc(t_num, t_den)
        return kl

    return tf.reduce_sum(kl_div(0.02, tf.reduce_mean(z, 0)))


# def kl_divergence(p, p_hat):
#    return tf.reduce_mean(p * tf.math.log(p) - p * tf.math.log(p_hat) + (1 - p) * tf.math.log(1 - p) - (1 - p) * tf.math.log(1 - p_hat))

def kl_divergence(rho, rho_hat):
    return rho * tf.math.log(rho) - rho * tf.math.log(rho_hat) + (1 - rho) * tf.math.log(1 - rho) - (
                1 - rho) * tf.math.log(1 - rho_hat)


@tf.function
def L4(y_true, y_pred):
    z = autoencoder.encoder(y_true)
    rho_hat = tf.reduce_mean(z, axis=1)
    kl = kl_divergence(0.05, rho_hat)

    diff = y_true - y_pred

    cost = 0.5 * tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=1))  # + 3 * tf.reduce_sum(kl)
    return cost


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tf.config.run_functions_eagerly(True)
    seed = 2162024
    plot_losses = PlotLosses()

    (x_train, _), (x_test, _) = K.datasets.fashion_mnist.load_data()

    (x_train, _), (x_test, _) = keras.datasets.cifar100.load_data()

    # x_train = K.preprocessing.image_dataset_from_directory('./Dataset', label_mode=None, color_mode='rgb',
    #                                                            image_size=(128, 128), shuffle=True, seed=seed,
    #                                                            validation_split=0.2, subset='training', batch_size=4)
    #
    # x_test = K.preprocessing.image_dataset_from_directory('./Dataset',  label_mode=None, color_mode='rgb',
    #                                                           image_size=(128, 128), shuffle=True, seed=seed,
    #                                                           validation_split=0.2, subset='validation', batch_size=4)

    # x_train = x_train.map(lambda x: tf.cast(x, tf.float32)/255.)
    # x_test  = x_test.map(lambda x: tf.cast(x, tf.float32) / 255.)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    # print(x_train.shape)
    # print(x_test.shape)

    # autoencoder = Autoencoder(latent_dim)
    autoencoder = ConvAutoEncoder()

    autoencoder.compile(optimizer='adam', loss=K.losses.MeanSquaredError())
    # autoencoder.compile(optimizer='adam', loss=L4)

    # autoencoder.fit(x_train,
    #                 epochs=10,
    #                 shuffle=True,
    #                 validation_data=x_test,
    #                 callbacks=[plot_losses])

    autoencoder.fit(x_train, x_train,
                    epochs=10,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[plot_losses])

    encoded_imgs = autoencoder.encoder(x_test).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

    # x_test.take(1).get_single_element()
    # x_test.take(1).as_numpy_iterator()

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
        spar = (len(encoded_imgs[i]) - np.count_nonzero(encoded_imgs[i])) / len(encoded_imgs[i]) * 100
        print('PSNR: \t {} \t \t Sparsity: \t {}'.format(round(psnr, 2), round(spar, 2)))
    plt.show()


