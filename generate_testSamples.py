import os
from email import utils
from glob import glob

import cv2
import tensorflow as tf
from tqdm import tqdm

import utils as utils
from PhIREGANs import *


def generate_test_dataset(dataset_name, lr=10, hr=500, sample_name = None, hpcc = True):
    real_npy_path = 'example_data/' + dataset_name + '_test'+str(sample_name)+'_real.npy'
    lr_npy_path = 'example_data/' + dataset_name + '_test'+str(sample_name)+'_lr.npy'
    lr_tfrecord_path = 'example_data/' + dataset_name + '_test'+str(sample_name)+'_lr.tfrecord'

    if dataset_name == 'wind_2014':
        if hpcc:
            dataset_path = '/lustre/scratch/guiyli/Dataset_WIND/DIP/Wind2014_removed/u_v'
        else:
            dataset_path = '/home/guiyli/Documents/DataSet/Wind/2014/u_v'
    elif dataset_name == 'solar_2014':
        if hpcc:
            dataset_path = '/lustre/scratch/guiyli/Dataset_NSRDB/DIP/Solar2014_removed'
        else:
            dataset_path = '/home/guiyli/Documents/DataSet/Solar/npyFiles/dni_dhi/2014'
    image_path = dataset_path+'/'+sample_name+'.npy'
    np.random.seed(666)

    images_channel = 2
    data_lr = np.empty((lr,lr,images_channel))
    data_hr = np.empty((hr,hr,images_channel))

    img_hr = np.load(image_path).astype(np.float64)
    if img_hr.shape[0] == hr:
        data_hr = img_hr
    for c in range(images_channel):
        if not img_hr.shape[0] == hr:
            data_hr[:,:,c] = cv2.resize(img_hr[:,:,c], (hr,hr), interpolation=cv2.INTER_NEAREST)
        data_lr[:,:,c] = cv2.resize(data_hr[:,:,c], (lr,lr), interpolation=cv2.INTER_AREA)

    np.save(real_npy_path, data_hr[None,:,:,:])
    np.save(lr_npy_path, data_lr[None,:,:,:])

    utils.generate_TFRecords(
        lr_tfrecord_path, data_HR=data_hr[None,:,:,:], data_LR=data_lr[None,:,:,:], mode='test',
    )
    return lr_tfrecord_path


def generate_test_dataset_all(dataset_name="solar_2009", lr=10, mr=100, hr=500, hpcc=True):
    if hpcc:
        if dataset_name.split('_')[0]=='wind':
            real_npy_path = '/lustre/scratch/guiyli/Dataset_WIND/PhIRE/' + dataset_name + '_testAll_real.npy'
            lr_npy_path = '/lustre/scratch/guiyli/Dataset_WIND/PhIRE/' + dataset_name + '_testAll_lr.npy'
            mr_npy_path = '/lustre/scratch/guiyli/Dataset_WIND/PhIRE/' + dataset_name + '_testAll_mr.npy'
            lr_tfrecord_path = '/lustre/scratch/guiyli/Dataset_WIND/PhIRE/' + dataset_name + '_testAll_lr.tfrecord'
            mr_tfrecord_path = '/lustre/scratch/guiyli/Dataset_WIND/PhIRE/' + dataset_name + '_testAll_mr.tfrecord'
        elif dataset_name.split('_')[0] == 'solar':
            real_npy_path = '/lustre/scratch/guiyli/Dataset_NSRDB/PhIRE/' + dataset_name + '_testAll_real.npy'
            lr_npy_path = '/lustre/scratch/guiyli/Dataset_NSRDB/PhIRE/' + dataset_name + '_testAll_lr.npy'
            mr_npy_path = '/lustre/scratch/guiyli/Dataset_NSRDB/PhIRE/' + dataset_name + '_testAll_mr.npy'
            lr_tfrecord_path = '/lustre/scratch/guiyli/Dataset_NSRDB/PhIRE/' + dataset_name + '_testAll_lr.tfrecord'
            mr_tfrecord_path = '/lustre/scratch/guiyli/Dataset_NSRDB/PhIRE/' + dataset_name + '_testAll_mr.tfrecord'
    else:
        real_npy_path = 'example_data/' + dataset_name + '_testAll_real.npy'
        lr_npy_path = 'example_data/' + dataset_name + '_testAll_lr.npy'
        lr_tfrecord_path = 'example_data/' + dataset_name + '_testAll_lr.tfrecord'

    if dataset_name == 'solar_2009':
        if hpcc:
            dataset_path = '/home/guiyli/DataSet/Solar/2009/dni_dhi/test'
        else:
            dataset_path = '/home/guiyli/Documents/DataSet/NSRDB/500X500/2009/grid1/dni_dhi/test'
    elif dataset_name == 'wind_2014':
        if hpcc:
            dataset_path = '/lustre/scratch/guiyli/Dataset_WIND/DIP/Wind2014_removed/u_v'
        else:
            dataset_path = '/home/guiyli/Documents/DataSet/Wind/2014/u_v'
    elif dataset_name == 'solar_2014':
        if hpcc:
            dataset_path = '/lustre/scratch/guiyli/Dataset_NSRDB/DIP/Solar2014_removed'
        else:
            dataset_path = '/home/guiyli/Documents/DataSet/Solar/npyFiles/dni_dhi/2014'
    images_path = glob(dataset_path + '/*.npy')
    images_channel = 2
    images_num = len(images_path)
    data_out_lr = np.empty((images_num, lr, lr, images_channel))
    data_out_mr = np.empty((images_num, mr, mr, images_channel))
    data_out_real = np.empty((images_num, hr, hr, images_channel))

    data_lr = np.empty((lr,lr,images_channel))
    data_mr = np.empty((mr,mr,images_channel))
    data_hr = np.empty((hr,hr,images_channel))

    for i in tqdm(range(images_num)):
        img_hr = np.load(images_path[i]).astype(np.float64)
        if img_hr.shape[0] == hr:
            data_hr = img_hr

        for c in range(images_channel):
            if not img_hr.shape[0] == hr:
                data_hr[:,:,c] = cv2.resize(img_hr[:,:,c], (hr,hr), interpolation=cv2.INTER_NEAREST)
            data_lr[:,:,c] = cv2.resize(data_hr[:,:,c], (lr,lr), interpolation=cv2.INTER_AREA)
            data_mr[:,:,c] = cv2.resize(data_hr[:,:,c], (mr,mr), interpolation=cv2.INTER_AREA)

        data_out_lr[i] = data_lr
        data_out_mr[i] = data_mr
        data_out_real[i] = data_hr

    np.save(real_npy_path, data_out_real) # N, 500, 500, 2
    np.save(mr_npy_path, data_out_mr) # N, 100, 100, 2
    np.save(lr_npy_path, data_out_lr) # N, 10, 10, 2
    del data_out_real

    utils.generate_TFRecords(
        lr_tfrecord_path, data_HR=data_out_lr, data_LR=data_out_lr, mode='test',
    )
    utils.generate_TFRecords(
        mr_tfrecord_path, data_HR=data_out_mr, data_LR=data_out_mr, mode='test',
    )


def generate_test_samples_all(
    data_type="solar", dataset_name = "solar_2009", model_name='wind_07-10_bs4_epoch10', r_lr_mr=[2, 5], r_mr_hr=[5], hpcc=False, data_path=None
):
    phiregans = PhIREGANs(data_type=data_type)
    # -------------------------------------------------------------
    # LR-MR wind (10, 10, 2) solar (20,20,2)--> (100, 100, 2)
    if data_path is None:
        if hpcc:
            if data_type == 'wind':
                data_path = '/lustre/scratch/guiyli/Dataset_WIND/PhIRE/' + dataset_name + '_testAll_lr.tfrecord'
            elif data_type == 'solar':
                data_path = '/lustre/scratch/guiyli/Dataset_NSRDB/PhIRE/' + dataset_name + '_testAll_lr.tfrecord'
        else:
            data_path = 'example_data/' + dataset_name + '_testAll_lr.tfrecord'
    model_path = 'models/' + model_name + '/' + data_type + '_lr-mr/trained_gan/gan'
    if data_type == 'wind':
        phiregans.mu_sig = [[0.7684, -0.4575], [4.9491, 5.8441]]
    elif data_type == 'solar':
        phiregans.mu_sig = [[344.3262, 113.7444], [370.8409, 111.1224]]

    data_out_path = phiregans.test(
        r=r_lr_mr,
        data_path=data_path,
        model_path=model_path,
        plot_data=False,
        batch_size=8,
    )
    lr_mr_result = np.load(data_out_path + '/dataSR.npy') # original range [-314.7676617413293, 1269.250537298715]
    os.rename(
        data_out_path + '/dataSR.npy',
        data_out_path + '/' + data_type + '_testAll_result_mr.npy',
    )
    # fakr mr images
    data_path = data_out_path + '/' + data_type + '_testAll_result_mr.tfrecord'

    utils.generate_TFRecords(
        data_path,
        data_HR=lr_mr_result,
        data_LR=lr_mr_result,
        mode='test',
    )
    del lr_mr_result

    # -------------------------------------------------------------
    # MR-HR (100, 100, 2) --> (500, 500, 2)
    # real mr images
    # data_path = '/lustre/scratch/guiyli/Dataset_WIND/PhIRE/' + dataset_name + '_testAll_mr.tfrecord'
    model_path = 'models/' + model_name + '/' + data_type + '_mr-hr/trained_gan/gan'
    if data_type == 'wind':
        phiregans.mu_sig = [[0.7684, -0.4575], [5.02455, 5.9017]]
    elif data_type == 'solar':
        phiregans.mu_sig = [[344.3262, 113.7444], [386.9283, 117.9627]]
    data_out_path = phiregans.test(
        r=r_mr_hr,
        data_path=data_path,
        model_path=model_path,
        plot_data=False,
        batch_size=8,
    )
    os.rename(
        data_out_path + '/dataSR.npy',
        data_out_path + '/' + data_type + '_testAll_result_hr.npy',
    )
    return data_out_path


if __name__ == '__main__':
    # across the entire dataset

    # # WIND Total Scale: 50X
    # # ----------------------------------
    # # generate test dataset for Wind_2014
    # generate_test_dataset_all('wind_2014', 10, 100, 500, hpcc=True)
    # # generate mr and hr output for entire Wind_2014 dataset
    # data_out_path = generate_test_samples_all(
    #     'wind', 'wind_2014', model_name='pretrained_wind', r_lr_mr=[2, 5], r_mr_hr=[5], hpcc=True
    # )
    # print('data_out_path:', data_out_path)

    # Solar Total Scale: 25X
    # ----------------------------------
    # # generate test dataset for solar_2014
    # generate_test_dataset_all('solar_2014', 20, 100, 500, hpcc=True)
    # # generate mr and hr output for entire solar_2014 dataset
    # data_out_path = generate_test_samples_all(
    #     'solar', 'solar_2014', model_name='pretrained_solar', r_lr_mr=[5], r_mr_hr=[5], hpcc=True
    # )
    # print('data_out_path:', data_out_path)

    # for a single choosen image
    # ----------------------------------
    dataset_name = 'wind_2014'
    sample_name = 'wtk_grid2_2014-07-19-12:00:00'
    lr_tfrecord_path = generate_test_dataset(dataset_name, lr=10, hr=500, sample_name = sample_name, hpcc = True)
    data_out_path = generate_test_samples_all(
        'wind', 'wind_2014', model_name='pretrained_wind', r_lr_mr=[2, 5], r_mr_hr=[5], hpcc=True, data_path=lr_tfrecord_path
    )
    print('data_out_path:', data_out_path)

    dataset_name = 'solar_2014'
    sample_name = 'month1_day14_hour16_minute30'
    lr_tfrecord_path = generate_test_dataset(dataset_name, lr=20, hr=500, sample_name = sample_name, hpcc = True)
    data_out_path = generate_test_samples_all(
        'solar', 'solar_2014', model_name='pretrained_solar', r_lr_mr=[5], r_mr_hr=[5], hpcc=True, data_path=lr_tfrecord_path
    )
    print('data_out_path:', data_out_path)


    