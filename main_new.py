from PhIREGANs import *


def train_Wind(batch_size = 8, epoch = 10):
    # WIND Total Scale: 50X
    data_type = 'wind'

    # -------------------------------------------------------------
    # LR-MR (10, 10, 2) --> (100, 100, 2)
    data_path = 'example_data/wind_2007,2008_LR-MR-train.tfrecord'
    model_path = None  # (string) path of previously trained model to load in if continuing training
    r = [2, 5]
    phiregans = PhIREGANs(data_type=data_type, mu_sig=None, print_every=400, N_epochs=epoch)
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
    data_path = 'example_data/wind_2007,2008_MR-HR-train.tfrecord'
    model_path = None
    r = [5]
    phiregans = PhIREGANs(data_type=data_type, mu_sig=None, print_every=400, N_epochs=epoch)
    # train cnn
    model_dir = phiregans.pretrain(
        r=r, data_path=data_path, model_path=model_path, batch_size=batch_size
    )
    # train gan
    model_dir = phiregans.train(
        r=r, data_path=data_path, model_path=model_dir, batch_size=batch_size
    )

def train_Solar(batch_size = 8, epoch = 10):
    # WIND Total Scale: 25X
    data_type = 'solar'

    # -------------------------------------------------------------
    # LR-MR (20, 20, 2) --> (100, 100, 2)
    data_path = 'example_data/solar_2009,2010,2011_LR-MR-train.tfrecord'
    model_path = None
    r = [5]
    phiregans = PhIREGANs(data_type=data_type, mu_sig=None, print_every=400, N_epochs=epoch)
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
    data_path = 'example_data/wind_2007,2008_MR-HR-train.tfrecord'
    model_path = None
    r = [5]
    phiregans = PhIREGANs(data_type=data_type, mu_sig=None, print_every=400, N_epochs=epoch)
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

    model_dir = train_Solar(batch_size = 8, epoch = 10)

    finish_time = time.time() - start_time
    total_time_print = 'Training Finished. Took {:.4f} minutes or {:.4f} hours to complete.'.format(
        finish_time / 60, finish_time / 3600
    )
    print(total_time_print)
    text_file = open(os.path.join(model_dir, "total_time.txt"), "w")
    text_file.write(total_time_print)
    text_file.close()