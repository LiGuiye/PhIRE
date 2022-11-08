import os

import numpy as np
import torch
from tqdm import tqdm


def sliced_wasserstein_cuda(A, B, dir_repeats=4, dirs_per_repeat=128):
    """
    A, B: dreal, dfake(after normalize: -mean/std [0,1])

    Reference:
        https://github.com/tkarras/progressive_growing_of_gans
    """
    assert A.ndim == 2 and A.shape == B.shape                                   # (neighborhood, descriptor_component)
    device = torch.device("cuda")
    results = torch.empty(dir_repeats, device=torch.device("cpu"))
    A = torch.from_numpy(A).to(device) if not isinstance(A, torch.Tensor) else A.to(device)
    B = torch.from_numpy(B).to(device) if not isinstance(B, torch.Tensor) else B.to(device)
    for repeat in range(dir_repeats):
        dirs = torch.randn(A.shape[1], dirs_per_repeat, device=device, dtype=torch.float64)          # (descriptor_component, direction)
        dirs = torch.divide(dirs, torch.sqrt(torch.sum(torch.square(dirs), dim=0, keepdim=True)))  # normalize descriptor components for each direction
        projA = torch.matmul(A, dirs)                                           # (neighborhood, direction)
        projB = torch.matmul(B, dirs)
        projA = torch.sort(projA, dim=0)[0]                                     # sort neighborhood projections for each direction
        projB = torch.sort(projB, dim=0)[0]
        dists = torch.abs(projA - projB)                                        # pointwise wasserstein distances
        results[repeat] = torch.mean(dists)                                     # average over neighborhoods and directions
    return torch.mean(results)                                                  # average over repeats

def normalize_standard(image):
    """
    Standard Score Normalization

    (image - mean) / std

    return: data_new, mean, std
    """
    if isinstance(image, torch.Tensor):
        mean = torch.mean(image)
        std = torch.std(image)
        return (
            torch.divide(
                torch.add(image, -mean), torch.maximum(std, torch.tensor(1e-5))
            ),
            mean,
            std,
        )
    else:
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)
        mean = np.mean(image)
        std = np.std(image)
        return (image - mean) / max(std, 1e-5), mean, std


def calc_metrics(real_path, fake_path, mse=True, swd=True, model_type=None):
    real = np.load(real_path) # NHWC
    fake = np.load(fake_path) # NHWC
    n, h, w, channel = real.shape
    metrics_min = real.min()
    metrics_max = real.max()
    if mse:
        metrics_mse = np.mean(np.square(real-fake), axis=(1,2,3)) / np.square(np.mean(real, axis=(1,2,3)))
        save_name = 'error_mse_'+model_type+'.npy'
        save_path = os.path.join('/'.join(fake_path.split('/')[:2]), save_name)
        np.save(save_path, metrics_mse)
        print(save_path, 'saved!')
    if swd:
        metrics_swd = torch.empty((channel,n), dtype=torch.float64)
        for i in tqdm(range(n)):
            for c in range(channel):
                # normalize before calc SWD
                img_gt_n = normalize_standard(real[i,:,:,c])[0]
                img_fake_n = normalize_standard(fake[i,:,:,c])[0]
                metrics_swd[c][i] = sliced_wasserstein_cuda(img_gt_n,img_fake_n)
        
        save_name = 'error_swd_'+model_type+'.npy'
        save_path = os.path.join('/'.join(fake_path.split('/')[:2]), save_name)
        np.save(save_path,torch.mean(metrics_swd, 0).numpy())
        print(save_path, 'saved!')
    
    text_file = open(
        os.path.join('/'.join(fake_path.split('/')[:2]), "mean_median_metrics_mse_swd_"+str(model_type)+".txt"),
        "w")

    drange = metrics_max - metrics_min
    text_file.write("\n" + "Data Range: " + str(drange) + "\n")
    text_file.write(str(metrics_min) + ", " + str(metrics_max) + "\n")

    if mse:
        text_file.write("\n" + "MSE/(mean^2) --> mean" + "\n")
        text_file.write(str(np.mean(metrics_mse)) + "\n")
        text_file.write("\n" + "MSE/(mean^2) --> median" + "\n")
        text_file.write(str(np.median(metrics_mse)) + "\n")

    if swd:
        text_file.write("\n" + "SWD/(mean^2) --> mean" + "\n")
        text_file.write(str(torch.mean(metrics_swd).numpy()) + "\n")
        text_file.write("\n" + "SWD/(mean^2) --> median" + "\n")
        text_file.write(str(torch.median(metrics_swd).numpy()) + "\n")
    print("Validation metrics saved!")
    text_file.close()


data_type_list = ['Solar']
for data_type in data_type_list:
    if data_type == 'Wind':
        # calculate MSE
        data_out_path = 'data_out/wind-20221016-153330' # lr_mr_hr
        # 10X
        real_path = '/lustre/scratch/guiyli/Dataset_WIND/PhIRE/wind_2014_testAll_mr.npy'
        fake_path = data_out_path+'/wind_testAll_result_mr.npy'
        calc_metrics(real_path, fake_path, mse=True, swd=True, model_type='lr_mr')
        # data_out_path = 'data_out/wind-20221016-170305' # mr_hr (5X)
        # 50X or 5X
        real_path = '/lustre/scratch/guiyli/Dataset_WIND/PhIRE/wind_2014_testAll_real.npy'
        fake_path = data_out_path+'/wind_testAll_result_hr.npy'
        calc_metrics(real_path, fake_path, mse=True, swd=True, model_type='mr_hr')
    elif data_type == 'Solar':
        # calculate MSE
        # data_out_path = 'data_out/solar-20221020-115955' # lr_mr_hr
        data_out_path = 'data_out/solar-20221107-081227'
        # 5X
        real_path = '/lustre/scratch/guiyli/Dataset_NSRDB/PhIRE/solar_2014_testAll_mr.npy'
        fake_path = data_out_path+'/solar_testAll_result_mr.npy'
        calc_metrics(real_path, fake_path, mse=True, swd=True, model_type='lr_mr')
        # 25X
        real_path = '/lustre/scratch/guiyli/Dataset_NSRDB/PhIRE/solar_2014_testAll_real.npy'
        fake_path = data_out_path+'/solar_testAll_result_hr.npy'
        calc_metrics(real_path, fake_path, mse=True, swd=True, model_type='mr_hr')
