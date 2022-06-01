from glob import glob

import numpy as np
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


if __name__ == '__main__':
    wind_train = glob('/home/guiyli/Documents/DataSet/Wind/2007/u_v/train/*.npy')
    channel = 2

    # TFRecord_path = 'example_data/wind_2007_MR-HR.tfrecord'
    # hr_size = 500
    # lr_size = 100
    # scale = 5

    TFRecord_path = 'example_data/wind_2007_LR-MR.tfrecord'
    hr_size = 100
    lr_size = 10
    scale = 10

    # NHWC
    data_batches_hr = np.empty(
        (len(wind_train), hr_size, hr_size, channel), dtype=np.float32
    )
    data_batches_low = np.empty(
        (len(wind_train), lr_size, lr_size, channel), dtype=np.float32
    )

    for idx, image in enumerate(tqdm(wind_train)):
        img_hr = np.load(image)
        if not img_hr.shape[0] == hr_size:
            img_hr = tool.downscale_image(img_hr, img_hr.shape[0] // hr_size)
        img_low = tool.downscale_image(img_hr, scale)
        data_batches_hr[idx] = img_hr
        data_batches_low[idx] = img_low
    tool.generate_TFRecords(TFRecord_path, data_batches_hr, data_batches_low, 'train')
