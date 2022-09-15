import argparse
from time import time

from PhIREGANs import *


def train(
    data_type='solar', dataset_name='solar_2009,2010,2011', batch_size=8, epoch=10
):
    # -------------------------------------------------------------
    # LR-MR  solar (20, 20, 2) wind (10, 10, 2) --> (100, 100, 2)
    if dataset_name == "solar_2007,2008,2009,2010,2011,2012":
        # this one is too large
        data_path = '/lustre/scratch/guiyli/Dataset_NSRDB/PhIRE/solar_2007,2008,2009,2010,2011,2012_LR-MR-train.tfrecord'
    elif dataset_name == "wind_07-12":
        data_path = "/lustre/scratch/guiyli/Dataset_WIND/PhIRE/wind_2007,2008,2009,2010,2011,2012_LR-MR-train.tfrecord"
    else:
        data_path = 'example_data/' + dataset_name + '_LR-MR-train.tfrecord'
    model_path = None
    r = [5] if data_type == 'solar' else [2, 5]
    phiregans = PhIREGANs(
        data_type=data_type, mu_sig=None, print_every=4000, N_epochs=epoch
    )
    # train cnn
    phiregans.N_epochs = 2000
    model_dir = phiregans.pretrain(
        r=r, data_path=data_path, model_path=model_path, batch_size=batch_size
    )
    # train gan
    phiregans.N_epochs = 200
    model_dir = phiregans.train(
        r=r, data_path=data_path, model_path=model_dir, batch_size=batch_size
    )

    # -------------------------------------------------------------
    # MR-HR (100, 100, 2) --> (500, 500, 2)
    if dataset_name == "solar_2007,2008,2009,2010,2011,2012":
        # this one is too large
        data_path = '/lustre/scratch/guiyli/Dataset_NSRDB/PhIRE/solar_2007,2008,2009,2010,2011,2012_MR-HR-train.tfrecord'
    elif dataset_name == "wind_07-12":
        data_path = "/lustre/scratch/guiyli/Dataset_WIND/PhIRE/wind_2007,2008,2009,2010,2011,2012_MR-HR-train.tfrecord"
    else:
        data_path = 'example_data/' + dataset_name + '_MR-HR-train.tfrecord'
    model_path = None
    r = [5]
    phiregans = PhIREGANs(
        data_type=data_type, mu_sig=None, print_every=4000, N_epochs=epoch
    )
    # train cnn
    phiregans.N_epochs = 200
    model_dir = phiregans.pretrain(
        r=r, data_path=data_path, model_path=model_path, batch_size=batch_size
    )
    # train gan
    phiregans.N_epochs = 20
    model_dir = phiregans.train(
        r=r, data_path=data_path, model_path=model_dir, batch_size=batch_size
    )
    return model_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LAG-Pytorch')
    parser.add_argument('--data_type', type=str, default='solar', help='solar or wind')
    parser.add_argument('--dataset_name', type=str, default='solar_2009', help='solar_2007,2008,2009,2010,2011,2012 or solar_2009 or wind_2007,2008')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=10)
    # parse config
    args = parser.parse_args()

    start_time = time()
    # -----------------------------------------------
    # model_dir = train(
        # data_type='solar', dataset_name='solar_2007,2008,2009,2010,2011,2012', batch_size=16, epoch=10
    # )

    # model_dir = train(
    #     data_type='wind', dataset_name='wind_2007,2008', batch_size=8, epoch=10
    # )

    # model_dir = train(
    #     data_type='solar', dataset_name='solar_2009', batch_size=16, epoch=10
    # )

    model_dir = train(
        data_type=args.data_type, dataset_name=args.dataset_name, batch_size=args.batch_size, epoch=args.epoch
    )
    # -----------------------------------------------
    finish_time = time() - start_time
    total_time_print = 'Training Finished. Took {:.4f} minutes or {:.4f} hours to complete.'.format(
        finish_time / 60, finish_time / 3600
    )
    print(total_time_print)
    os.makedirs(model_dir, exist_ok=True)
    text_file = open(os.path.join(model_dir, "total_time.txt"), "w")
    text_file.write(total_time_print)
    text_file.close()
