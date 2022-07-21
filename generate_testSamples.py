from email import utils
import tensorflow as tf
import os
import utils as utils
from PhIREGANs import *
from glob import glob
from PIL import Image
import copy


def _parse_test_(serialized_example, mu_sig=None):
    """
        Parser data from TFRecords for the models to read in for testing

        inputs:
            serialized_example - batch of data drawn from tfrecord
            mu_sig             - mean, standard deviation if known

        outputs:
            idx     - array of indicies for each sample
            data_LR - array of LR images in the batch
    """

    feature = {
        'index': tf.FixedLenFeature([], tf.int64),
        'data_LR': tf.FixedLenFeature([], tf.string),
        'h_LR': tf.FixedLenFeature([], tf.int64),
        'w_LR': tf.FixedLenFeature([], tf.int64),
        'c': tf.FixedLenFeature([], tf.int64),
    }
    example = tf.parse_single_example(serialized_example, feature)

    idx = example['index']

    h_LR, w_LR = example['h_LR'], example['w_LR']

    c = example['c']

    data_LR = tf.decode_raw(example['data_LR'], tf.float64)

    data_LR = tf.reshape(data_LR, (h_LR, w_LR, c))

    if mu_sig is not None:
        data_LR = (data_LR - mu_sig[0]) / mu_sig[1]

    return idx, data_LR


def calc_mu_sig_for_test(data_path, batch_size=1):
    """
        Compute mean (mu) and standard deviation (sigma) for each data channel
        inputs:
            data_path - (string) path to the tfrecord for the training data
            batch_size - number of samples to grab each interation

        outputs:
            sets self.mu_sig
    """
    print('Loading data ...', end=' ')
    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(_parse_test_).batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    _, LR_out = iterator.get_next()

    with tf.Session() as sess:
        N, mu, sigma = 0, 0, 0
        try:
            while True:
                data_LR = sess.run(LR_out)

                N_batch, h, w, c = data_LR.shape
                N_new = N + N_batch

                mu_batch = np.mean(data_LR, axis=(0, 1, 2))
                sigma_batch = np.var(data_LR, axis=(0, 1, 2))

                sigma = (
                    (N / N_new) * sigma
                    + (N_batch / N_new) * sigma_batch
                    + (N * N_batch / N_new ** 2) * (mu - mu_batch) ** 2
                )
                mu = (N / N_new) * mu + (N_batch / N_new) * mu_batch

                N = N_new

        except tf.errors.OutOfRangeError:
            pass

    mu_sig = [mu, np.sqrt(sigma)]
    print("mu_sig: ", mu_sig)
    print('Done.')
    return mu_sig


def generate_test_dataset(data_type, lr=10, hr=500):
    real_npy_path = 'example_data/' + data_type + '_test50_real.npy'
    lr_npy_path = 'example_data/' + data_type + '_test50_lr.npy'
    lr_tfrecord_path = 'example_data/' + data_type + '_test50_lr.tfrecord'

    if not (
        os.path.exists(real_npy_path)
        or os.path.exists(lr_npy_path)
        or os.path.exists(lr_tfrecord_path)
    ):
        np.random.seed(666)
        sample_indices = np.random.choice(range(1, 209), 50, replace=False).tolist()
        print(sample_indices)
        if data_type == 'solar':
            dataset_path = (
                '/home/guiyli/Documents/DataSet/NSRDB/500X500/2013/grid1/dni_dhi'
            )
        elif data_type == 'wind':
            dataset_path = '/home/guiyli/Documents/DataSet/Wind/2010/u_v'
        images_path = glob(dataset_path + '/*.npy')

        data_out_lr = None
        data_out_real = None
        for i in sample_indices:
            data_real = np.load(images_path[i]).astype(np.float64)
            data_lr = utils.downscale_image(data_real, hr // lr)
            if data_out_lr is None:
                data_out_lr = data_lr
                data_out_real = data_real[np.newaxis, :, :, :]
            else:
                data_out_lr = np.concatenate((data_out_lr, data_lr), axis=0)
                data_out_real = np.concatenate(
                    (data_out_real, data_real[np.newaxis,]), axis=0
                )

        np.save(real_npy_path, data_out_real)
        np.save(lr_npy_path, data_out_lr)

        utils.generate_TFRecords(
            lr_tfrecord_path, data_HR=data_out_lr, data_LR=data_out_lr, mode='test',
        )
    else:
        print(data_type, "test dataset already exist")


def generate_test_samples(
    data_type, model_name='wind_07-10_bs4_epoch10', r_lr_mr=[2, 5], r_mr_hr=[5]
):
    phiregans = PhIREGANs(data_type=data_type)
    # -------------------------------------------------------------
    # LR-MR wind (10, 10, 2) solar (20,20,2)--> (100, 100, 2)
    data_path = 'example_data/' + data_type + '_test50_lr.tfrecord'
    model_path = 'models/' + model_name + '/' + data_type + '_lr-mr/trained_gan/gan'
    phiregans.mu_sig = calc_mu_sig_for_test(data_path)

    data_out_path = phiregans.test(
        r=r_lr_mr,
        data_path=data_path,
        model_path=model_path,
        plot_data=False,
        batch_size=8,
    )
    lr_mr_result = np.load(data_out_path + '/dataSR.npy')
    os.rename(
        data_out_path + '/dataSR.npy',
        data_out_path + '/' + data_type + '_test50_result_mr.npy',
    )

    utils.generate_TFRecords(
        data_out_path + '/' + data_type + '_test50_result_mr.tfrecord',
        data_HR=lr_mr_result,
        data_LR=lr_mr_result,
        mode='test',
    )

    # -------------------------------------------------------------
    # MR-HR (100, 100, 2) --> (500, 500, 2)
    data_path = data_out_path + '/' + data_type + '_test50_result_mr.tfrecord'
    model_path = 'models/' + model_name + '/' + data_type + '_mr-hr/trained_gan/gan'
    phiregans.mu_sig = calc_mu_sig_for_test(data_path)
    phiregans.test(
        r=r_mr_hr,
        data_path=data_path,
        model_path=model_path,
        plot_data=False,
        batch_size=8,
    )
    os.rename(
        data_out_path + '/dataSR.npy',
        data_out_path + '/' + data_type + '_test50_result_hr.npy',
    )
    return data_out_path + '/' + data_type + '_test50_result_hr.npy'


def images_to_grid(images):
    """Converts a grid of images (NNCHW 5D tensor) to a single image in HWC.

    Args:
    images: 5D tensor (count_y, count_x, colors, height, width), grid of images.

    Returns:
      a 3D tensor image of shape (count_y * height, count_x * width, colors).
    """
    ny, nx, c, h, w = images.shape
    images = images.transpose(0, 3, 1, 4, 2)
    images = images.reshape([ny * h, nx * w, c])
    return images


def to_png(x):
    """Convert a 3D tensor to png.

    Args:
    x: Tensor, 01C formatted input image.

    Returns:
    Tensor, 1D string representing the image in png format.
    """
    with tf.Graph().as_default():
        with tf.Session() as sess_temp:
            x = tf.constant(x, dtype=tf.float32)
            y = tf.image.encode_png(
                tf.cast(
                    tf.clip_by_value(tf.round(127.5 + 127.5 * x), 0, 255), tf.uint8
                ),
                compression=-1,
            )
            return sess_temp.run(y)


def stretch_min_max_grids(image, min_value=-1.0, max_value=1.0):
    image = np.asarray(image)
    n, h, w, c = image.shape
    for i in range(n):
        pixel_min = image[i, :, :, :].min()
        pixel_max = image[i, :, :, :].max()
        image[i, :, :, :] = (
            (image[i, :, :, :] - pixel_min) / max((pixel_max - pixel_min), 1e-5)
        ) * (max_value - min_value) + min_value
    return image


def plot_test_samples(fake_path, data_type, batch_size=4):
    """
    plot first 4 images in the test sample
    """
    real = np.load('example_data/' + data_type + '_test50_real.npy')
    low = np.load('example_data/' + data_type + '_test50_lr.npy')
    low = tf.image.resize_nearest_neighbor(low, (500, 500))
    sess = tf.Session()
    low = sess.run(low)
    fake = np.load(fake_path)

    # stretch to [-1, 1] for `to_png` to generate images at [0,255] data range for better visual effects
    real = stretch_min_max_grids(real)
    low = stretch_min_max_grids(low)
    fake = stretch_min_max_grids(fake)

    # 50,500,500,2
    real = real[:batch_size, np.newaxis, :, :, :]  # 4,1,500,500,2
    low = low[:batch_size, np.newaxis, :, :, :]  # 4,1,500,500,2
    fake = fake[:batch_size, np.newaxis, :, :, :]  # 4,1,500,500,2
    grids = np.concatenate((real, low, fake), axis=1).transpose(
        0, 1, 4, 2, 3
    )  # 4,3,2. 500,500
    images = images_to_grid(grids)

    _, _, c = images.shape
    for i in range(c):
        output_file = os.path.abspath('test' + str(i) + '.png')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        open(output_file, 'wb').write(to_png(images[:, :, i][:, :, np.newaxis]))
        print('Saved', output_file)


if __name__ == '__main__':
    # WIND Total Scale: 50X
    # ----------------------------------
    data_type = 'wind'
    model_name = 'wind_07-10_bs8_epoch10'
    generate_test_dataset(data_type, lr=10, hr=500)
    hr_test_samples = generate_test_samples(
        data_type, model_name=model_name, r_lr_mr=[2, 5]
    )
    plot_test_samples(fake_path=hr_test_samples, data_type=data_type)

    # WIND Total Scale: 25X
    # ----------------------------------
    # data_type = 'solar'
    # model_name = 'solar_09-13_bs8_epoch10'
    # generate_test_dataset(data_type, lr=20, hr=500)
    # hr_test_samples = generate_test_samples(
    #     data_type, model_name=model_name, r_lr_mr=[5]
    # )
    # plot_test_samples(fake_path=hr_test_samples, data_type=data_type)

