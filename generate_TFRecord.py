import os
from glob import glob

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import utils as tool

# WIND - Total Scale: 50X
# LR-MR (10, 10, 2) --> (100, 100, 2) - 10X
# MR-HR (100, 100, 2) --> (500, 500, 2) - 5X
# -------------------------------------------------------------

# SOLAR - Total Scale: 20X
# LR-MR (20, 20, 2) --> (100, 100, 2) - 5X
# MR-HR (100, 100, 2) --> (500, 500, 2) - 5X
# -------------------------------------------------------------


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class TFRecordExporter:
    def __init__(
        self,
        tfrecord_dir,
        expected_images,
        dataset_name,
        print_progress=True,
        progress_interval=100,
    ):
        self.tfrecord_dir = tfrecord_dir
        self.tfr_prefix = os.path.join(self.tfrecord_dir, dataset_name)
        self.expected_images = expected_images
        self.cur_images = 0
        self.shape = None
        self.resolution_log2 = None
        self.tfr_writer = None
        self.print_progress = print_progress
        self.progress_interval = progress_interval
        if self.print_progress:
            print('Creating dataset "%s"' % self.tfr_prefix)
        if not os.path.isdir(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)
        assert os.path.isdir(self.tfrecord_dir)

    def close(self):
        if self.print_progress:
            print('%-40s\r' % 'Flushing data...', end='', flush=True)
        self.tfr_writer.close()
        if self.print_progress:
            print('%-40s\r' % '', end='', flush=True)
            print('Added %d images.' % self.cur_images)

    def add_image(self, data_HR, data_LR, index, label, mode="train"):
        if self.tfr_writer is None:
            tfr_opt = tf.io.TFRecordOptions(compression_type="")
            tfr_file = self.tfr_prefix + '.tfrecord'
            self.tfr_writer = tf.io.TFRecordWriter(tfr_file, tfr_opt)

        if self.print_progress and self.cur_images % self.progress_interval == 0:
            print(
                '%d / %d\r' % (self.cur_images, self.expected_images),
                end='',
                flush=True,
            )
        if mode == "train":
            h_HR, w_HR, c = data_HR.shape
            h_LR, w_LR, c = data_LR.shape
            features = tf.train.Features(
                feature={
                    'index': _int64_feature(index),
                    'data_LR': _bytes_feature(data_LR.tostring()),
                    'h_LR': _int64_feature(h_LR),
                    'w_LR': _int64_feature(w_LR),
                    'data_HR': _bytes_feature(data_HR.tostring()),
                    'h_HR': _int64_feature(h_HR),
                    'w_HR': _int64_feature(w_HR),
                    'c': _int64_feature(c),
                    'label': _bytes_feature(label),
                }
            )
        elif mode == "test":
            h_LR, w_LR, c = data_LR.shape
            features = tf.train.Features(
                feature={
                    'index': _int64_feature(index),
                    'data_LR': _bytes_feature(data_LR.tostring()),
                    'h_LR': _int64_feature(h_LR),
                    'w_LR': _int64_feature(w_LR),
                    'c': _int64_feature(c),
                }
            )
        example = tf.train.Example(features=features)
        self.tfr_writer.write(example.SerializeToString())
        self.cur_images += 1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def wind_dataset(years=[2007, 2008], lr=10, mr=100, hr=500, mode="train"):
    data_list = []
    for year in years:
        data_list += sorted(
            glob('/home/guiyli/Documents/DataSet/Wind/' + str(year) + '/u_v/*.npy')
        )

    # TFRecord generate method by the author, but cannot handle large dataset.
    # TFRecord_path_mr_hr = (
    #     'example_data/wind_' + ','.join([str(e) for e in years]) + '_MR-HR.tfrecord'
    # )
    # TFRecord_path_lr_mr = (
    #     'example_data/wind_' + ','.join([str(e) for e in years]) + '_LR-MR.tfrecord'
    # )

    # channel = 2
    # data_batches_hr = np.empty((len(data_list), hr, hr, channel), dtype=np.float32)
    # data_batches_mr = np.empty((len(data_list), mr, mr, channel), dtype=np.float32)
    # data_batches_lr = np.empty((len(data_list), lr, lr, channel), dtype=np.float32)

    # for idx, image in enumerate(tqdm(data_list)):
    #     img_hr = np.load(image)
    #     if not img_hr.shape[0] == hr:
    #         img_hr = tool.downscale_image(img_hr, img_hr.shape[0] // hr).squeeze()

    #     img_lr = tool.downscale_image(img_hr, hr // lr).squeeze()
    #     img_mr = tool.downscale_image(img_hr, hr // mr).squeeze()

    #     data_batches_hr[idx] = img_hr
    #     data_batches_mr[idx] = img_mr
    #     data_batches_lr[idx] = img_lr
    # tool.generate_TFRecords(
    #     TFRecord_path_mr_hr, data_batches_hr, data_batches_mr, 'train'
    # )
    # tool.generate_TFRecords(
    #     TFRecord_path_lr_mr, data_batches_mr, data_batches_lr, 'train'
    # )

    # --------------------------------
    # MR-HR
    tfrecord_path = 'example_data/'
    tfrecord_name = 'wind_' + ','.join([str(e) for e in years]) + '_MR-HR-' + mode
    with TFRecordExporter(tfrecord_path, len(data_list), tfrecord_name) as tfr:
        for idx, image in enumerate(tqdm(data_list)):
            img_hr = np.load(image).astype('float64')
            if not img_hr.shape[0] == hr:
                img_hr = tool.downscale_image(img_hr, img_hr.shape[0] // hr).squeeze()

            img_mr = tool.downscale_image(img_hr, hr // mr).squeeze()

            label = tf.compat.as_bytes(
                data_list[idx].split('/')[-1][:-4], encoding='utf-8'
            )
            tfr.add_image(img_hr, img_mr, idx, label, mode=mode)

    # --------------------------------
    # LR-MR
    tfrecord_path = 'example_data/'
    tfrecord_name = 'wind_' + ','.join([str(e) for e in years]) + '_LR-MR-' + mode
    with TFRecordExporter(tfrecord_path, len(data_list), tfrecord_name) as tfr:
        for idx, image in enumerate(tqdm(data_list)):
            img_hr = np.load(image).astype('float64')
            if not img_hr.shape[0] == hr:
                img_hr = tool.downscale_image(img_hr, img_hr.shape[0] // hr).squeeze()

            img_mr = tool.downscale_image(img_hr, hr // mr).squeeze()
            img_lr = tool.downscale_image(img_hr, hr // lr).squeeze()

            label = tf.compat.as_bytes(
                data_list[idx].split('/')[-1][:-4], encoding='utf-8'
            )
            tfr.add_image(img_mr, img_lr, idx, label, mode=mode)


def solar_dataset(years=[2009, 2010, 2011], lr=20, mr=100, hr=500, mode="train"):
    # LR-MR (20, 20, 2) --> (100, 100, 2)
    # MR-HR (100, 100, 2) --> (500, 500, 2)
    data_list = []
    for year in years:
        data_list += sorted(
            glob(
                '/home/guiyli/Documents/DataSet/NSRDB/500X500/'
                + str(year)
                + '/grid1/dni_dhi/*.npy'
            )
        )
    # --------------------------------
    # MR-HR
    tfrecord_path = 'example_data/'
    tfrecord_name = 'solar_' + ','.join([str(e) for e in years]) + '_MR-HR-' + mode
    with TFRecordExporter(tfrecord_path, len(data_list), tfrecord_name) as tfr:
        for idx, image in enumerate(tqdm(data_list)):
            img_hr = np.load(image).astype('float64')
            if not img_hr.shape[0] == hr:
                img_hr = tool.downscale_image(img_hr, img_hr.shape[0] // hr).squeeze()

            img_mr = tool.downscale_image(img_hr, hr // mr).squeeze()

            label = tf.compat.as_bytes(
                data_list[idx].split('/')[-1][:-4], encoding='utf-8'
            )
            tfr.add_image(img_hr, img_mr, idx, label, mode=mode)

    # --------------------------------
    # LR-MR
    tfrecord_path = 'example_data/'
    tfrecord_name = 'solar_' + ','.join([str(e) for e in years]) + '_LR-MR-' + mode
    with TFRecordExporter(tfrecord_path, len(data_list), tfrecord_name) as tfr:
        for idx, image in enumerate(tqdm(data_list)):
            img_hr = np.load(image).astype('float64')
            if not img_hr.shape[0] == hr:
                img_hr = tool.downscale_image(img_hr, img_hr.shape[0] // hr).squeeze()

            img_mr = tool.downscale_image(img_hr, hr // mr).squeeze()
            img_lr = tool.downscale_image(img_hr, hr // lr).squeeze()

            label = tf.compat.as_bytes(
                data_list[idx].split('/')[-1][:-4], encoding='utf-8'
            )
            tfr.add_image(img_mr, img_lr, idx, label, mode=mode)


if __name__ == '__main__':
    # wind_dataset(years=[2007, 2008], lr=10, mr=100, hr=500, mode="train")
    # wind_dataset(years=[2010], lr=10, mr=100, hr=500, mode="test")
    solar_dataset(years=[2009, 2010, 2011], lr=20, mr=100, hr=500, mode="train")
    solar_dataset(years=[2013], lr=20, mr=100, hr=500, mode="test")
