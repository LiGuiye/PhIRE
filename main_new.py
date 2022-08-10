from PhIREGANs import *
import time


def train(
    data_type='solar', dataset_name='solar_2009,2010,2011', batch_size=8, epoch=10
):
    # -------------------------------------------------------------
    # LR-MR  solar (20, 20, 2) wind (10, 10, 2) --> (100, 100, 2)
    data_path = 'example_data/' + dataset_name + '_LR-MR-train.tfrecord'
    model_path = None
    r = [5] if data_type == 'solar' else [2, 5]
    phiregans = PhIREGANs(
        data_type=data_type, mu_sig=None, print_every=400, N_epochs=epoch
    )
    # train cnn
    model_dir = phiregans.pretrain(
        r=r, data_path=data_path, model_path=model_path, batch_size=batch_size
    )
    # train gan
    model_dir = phiregans.train(
        r=r, data_path=data_path, model_path=model_dir, batch_size=batch_size
    )

    # -------------------------------------------------------------
    # MR-HR (100, 100, 2) --> (500, 500, 2)
    data_path = 'example_data/' + dataset_name + '_MR-HR-train.tfrecord'
    model_path = None
    r = [5]
    phiregans = PhIREGANs(
        data_type=data_type, mu_sig=None, print_every=400, N_epochs=epoch
    )
    # train cnn
    model_dir = phiregans.pretrain(
        r=r, data_path=data_path, model_path=model_path, batch_size=batch_size
    )
    # train gan
    model_dir = phiregans.train(
        r=r, data_path=data_path, model_path=model_dir, batch_size=batch_size
    )
    return model_dir


if __name__ == '__main__':
    start_time = time.time()
    # -----------------------------------------------

    # model_dir = train(
    #     data_type='solar', dataset_name='solar_2009,2010,2011', batch_size=8, epoch=10
    # )

    # model_dir = train(
    #     data_type='wind', dataset_name='wind_2007,2008', batch_size=8, epoch=10
    # )

    model_dir = train(
        data_type='solar', dataset_name='solar_2009', batch_size=8, epoch=10
    )
    # -----------------------------------------------
    finish_time = time.time() - start_time
    total_time_print = 'Training Finished. Took {:.4f} minutes or {:.4f} hours to complete.'.format(
        finish_time / 60, finish_time / 3600
    )
    print(total_time_print)
    os.makedirs(model_dir, exist_ok=True)
    text_file = open(os.path.join(model_dir, "total_time.txt"), "w")
    text_file.write(total_time_print)
    text_file.close()
