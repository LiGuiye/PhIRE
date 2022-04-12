import tensorflow as tf
import numpy as np

class loadPhIRE:
    def __init__(self, mu_sig=None):
        self.mu_sig = mu_sig
        self.LR_data_shape = None

    def _parse_test_(self, serialized_example, mu_sig=None):
        '''
            Parser data from TFRecords for the models to read in for testing

            inputs:
                serialized_example - batch of data drawn from tfrecord
                mu_sig             - mean, standard deviation if known

            outputs:
                idx     - array of indicies for each sample
                data_LR - array of LR images in the batch
        '''

        feature = {'index': tf.FixedLenFeature([], tf.int64),
                 'data_LR': tf.FixedLenFeature([], tf.string),
                    'h_LR': tf.FixedLenFeature([], tf.int64),
                    'w_LR': tf.FixedLenFeature([], tf.int64),
                       'c': tf.FixedLenFeature([], tf.int64)}
        example = tf.parse_single_example(serialized_example, feature)

        idx = example['index']

        h_LR, w_LR = example['h_LR'], example['w_LR']

        c = example['c']

        data_LR = tf.decode_raw(example['data_LR'], tf.float64)

        data_LR = tf.reshape(data_LR, (h_LR, w_LR, c))

        if mu_sig is not None:
            data_LR = (data_LR - mu_sig[0])/mu_sig[1]

        return idx, data_LR


    def _parse_train_(self, serialized_example, mu_sig=None):
        '''
        Parser data from TFRecords for the models to read in for (pre)training

        inputs:
            serialized_example - batch of data drawn from tfrecord
            mu_sig             - mean, standard deviation if known

        outputs:
            idx     - array of indicies for each sample
            data_LR - array of LR images in the batch
            data_HR - array of HR images in the batch
        '''

        feature = {
            'index': tf.FixedLenFeature([], tf.int64),
            'data_LR': tf.FixedLenFeature([], tf.string),
            'h_LR': tf.FixedLenFeature([], tf.int64),
            'w_LR': tf.FixedLenFeature([], tf.int64),
            'data_HR': tf.FixedLenFeature([], tf.string),
            'h_HR': tf.FixedLenFeature([], tf.int64),
            'w_HR': tf.FixedLenFeature([], tf.int64),
            'c': tf.FixedLenFeature([], tf.int64),
        }
        example = tf.parse_single_example(serialized_example, feature)

        idx = example['index']

        h_LR, w_LR = example['h_LR'], example['w_LR']
        h_HR, w_HR = example['h_HR'], example['w_HR']

        c = example['c']

        data_LR = tf.decode_raw(example['data_LR'], tf.float64)
        data_HR = tf.decode_raw(example['data_HR'], tf.float64)

        data_LR = tf.reshape(data_LR, (h_LR, w_LR, c))
        data_HR = tf.reshape(data_HR, (h_HR, w_HR, c))

        if mu_sig is not None:
            data_LR = (data_LR - mu_sig[0]) / mu_sig[1]
            data_HR = (data_HR - mu_sig[0]) / mu_sig[1]

        return idx, data_LR, data_HR

    def set_LR_data_shape(self, data_path):
        '''
            Get size and shape of LR input data
            inputs:
                data_path - (string) path to the tfrecord of the data

            outputs:
                sets self.LR_data_shape
        '''
        print('Loading data ...', end=' ')
        dataset = tf.data.TFRecordDataset(data_path)
        dataset = dataset.map(self._parse_test_).batch(1)

        iterator = dataset.make_one_shot_iterator()
        _, LR_out = iterator.get_next()

        with tf.Session() as sess:
            data_LR = sess.run(LR_out) #(1,10,10,2)
        
        self.LR_data_shape = data_LR.shape[1:] # (10,10,2)

    def set_mu_sig(self, data_path, batch_size=1):
        '''
            Compute mean (mu) and standard deviation (sigma) for each data channel
            inputs:
                data_path - (string) path to the tfrecord for the training data
                batch_size - number of samples to grab each interation

            outputs:
                sets self.mu_sig
        '''
        print('Loading data ...', end=' ')

        dataset = tf.data.TFRecordDataset(data_path)
        dataset = dataset.map(self._parse_train_).batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
        _, _, HR_out = iterator.get_next()

        with tf.Session() as sess:
            N, mu, sigma = 0, 0, 0
            try:
                while True:
                    data_HR = sess.run(HR_out)  # (5,100,100,2)

                    N_batch, h, w, c = data_HR.shape
                    N_new = N + N_batch

                    mu_batch = np.mean(data_HR, axis=(0, 1, 2))
                    sigma_batch = np.var(data_HR, axis=(0, 1, 2))

                    sigma = (
                        (N / N_new) * sigma
                        + (N_batch / N_new) * sigma_batch
                        + (N * N_batch / N_new**2) * (mu - mu_batch) ** 2
                    )
                    mu = (N / N_new) * mu + (N_batch / N_new) * mu_batch

                    N = N_new

            except tf.errors.OutOfRangeError:
                pass

        self.mu_sig = [mu, np.sqrt(sigma)]

        print('Done.')


    def load(self, data_path, batch_size=100):
        tf.reset_default_graph()

        if self.mu_sig is None:
            self.set_mu_sig(data_path, batch_size)
        
        # self.set_LR_data_shape(data_path)
        # h, w, C = self.LR_data_shape # 10, 10, 2
        
        print('Building data pipeline ...', end=' ')
        ds = tf.data.TFRecordDataset(data_path)
        ds = (
            ds.map(lambda xx: self._parse_train_(xx, self.mu_sig))
            .shuffle(1000)
            .batch(batch_size)
        )

        iterator = tf.data.Iterator.from_structure(ds.output_types, ds.output_shapes)
        idx, LR_out, HR_out = iterator.get_next()

        init_iter = iterator.make_initializer(ds)
        print('Done.')
        with tf.Session() as sess:
            sess.run(init_iter)
            batch_idx, batch_LR, batch_HR = sess.run([idx, LR_out, HR_out])
            print(batch_idx, batch_LR, batch_HR)


if __name__ == '__main__':
    # there are only 5 images for wind
    # (5, 10, 10, 2) --> (5, 100, 100, 2) --> (5, 500, 500, 2)
    # data_path = 'example_data/wind_LR-MR.tfrecord'
    # data_path = 'example_data/wind_MR-HR.tfrecord'

    # also only 5 images for solar
    # (5, 20, 20, 2) --> (5, 100, 100, 2) --> (5, 500, 500, 2)
    # data_path = 'example_data/solar_LR-MR.tfrecord'
    data_path = 'example_data/solar_MR-HR.tfrecord'


    load = loadPhIRE()
    load.load(data_path, batch_size=16)
