from PhIREGANs import *

# # WIND Total Scale: 50X
# data_type = 'wind'
# batch_size = 4
# # -------------------------------------------------------------
# # LR-MR (10, 10, 2) --> (100, 100, 2)
# data_path = 'example_data/wind_2007,2008_LR-MR-train.tfrecord'
# model_path = None # (string) path of previously trained model to load in if continuing training
# r = [2, 5]
# mu_sig = [[-1.74402694, -1.74402694], [7.18036716, 7.18036716]]
# -------------------------------------------------------------
# MR-HR (100, 100, 2) --> (500, 500, 2)
# data_path = 'example_data/wind_2007,2008_MR-HR-train.tfrecord'
# model_path = None
# r_mr_hr = [5]
# mu_sig = [[-0.65514054, -1.0888864], [5.75660848, 4.80301305]]


# SOLAR - LR-MR
# -------------------------------------------------------------
# LR-MR (20, 20, 2) --> (100, 100, 2)
# MR-HR (100, 100, 2) --> (500, 500, 2)
# '''
# data_type = 'solar'
# data_path = 'example_data/solar_LR-MR.tfrecord'
# model_path = 'models/solar_lr-mr/trained_gan/gan'
# r = [5]
# [[mean_],[]]
# mu_sig=[[344.3262, 113.7444], [370.8409, 111.1224]]
# '''

# SOLAR - MR-HR
# -------------------------------------------------------------
# '''
# data_type = 'solar'
# data_path = 'example_data/solar_MR-HR.tfrecord'
# model_path = 'models/solar_mr-hr/trained_gan/gan'
# r = [5]
# mu_sig = [[344.3262, 113.7444], [386.9283, 117.9627]]
# '''

if __name__ == '__main__':
    # WIND Total Scale: 50X
    data_type = 'wind'
    batch_size = 4
    epoch = 10
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
