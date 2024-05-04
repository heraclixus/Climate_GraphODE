"""
obtain Fourier modes for visualization
"""
import torch
import seaborn as sns
import seaborn.objects as so 
import matplotlib.pyplot as plt
import torch.fft as fft
import numpy as np 
import pandas as pd
from torch_harmonics import RealSHT

"""
Helper functions 
"""

def obtain_freq_vars(x, x_pred, idx, type="fft"):
    if type == "fft":
        x_hat = torch.fft.fftshift(x, dim=(-2,-1))[:, idx, :, :]
        x_hat_pred = torch.fft.fftshift(x_pred, dim=(-2,-1))[:, idx, :, :]
    else: # sht
        x_hat = x[:, idx, :, :]
        x_hat_pred = x_pred[:, idx, :, :]
    spectrum = torch.abs(x_hat)
    spectrum_pred = torch.abs(x_hat_pred)
    amplitude_mean = spectrum[:, -1, :].mean(dim=0).cpu().numpy()
    amplitude_std = spectrum[:, -1, :].std(dim=0).cpu().numpy()
    amplitude_mean_pred = spectrum_pred[:, -1, :].mean(dim=0).cpu().numpy()
    amplitude_std_pred = spectrum_pred[:, -1, :].std(dim=0).cpu().numpy()
    N = spectrum.shape[-1]
    
    if type == "fft":
        wavenumbers = torch.fft.fftshift(torch.fft.fftfreq(N)).numpy()
        wavenumbers *= N
    else:
        wavenumbers = [i+1 for i in range(N)]
    
    wavenumbers = np.concatenate([wavenumbers, wavenumbers])
    preds_mean = np.concatenate([amplitude_mean, amplitude_mean_pred])
    preds_std = np.concatenate([amplitude_std, amplitude_std_pred])
    labels = np.array(["amplitude_target"] * N + ["amplitude_pred"] * N)
    data = pd.DataFrame({
        "wavenumber": wavenumbers,
        "amplitude": preds_mean,
        "pred_std": preds_std,
        "label": labels 
    })
    return data


"""
input: dataframe obtained from the helper above 
"""
def obtain_freq_ranges(data):
    N = len(data) // 2 
    wavenumbers = data.wavenumber[:N]
    upper, lower = data.amplitude + data.pred_std, data.amplitude - data.pred_std 
    upper_target, lower_target = upper[:N], lower[:N]
    upper_pred, lower_pred = upper[N:], lower[N:]
    return wavenumbers, upper_target, lower_target, upper_pred, lower_pred



"""
a specific plotting function when there's only one feature
"""
def one_step_plot_spectrum_single(pred_hat, pred_target, vars, model_name, predict_range=72, add_ribbon_bar=False, type="fft"):
    fig = plt.figure(figsize=(15,10))
    data = obtain_freq_vars(pred_hat, pred_target, 0, type=type)
    sns.lineplot(data=data, x="wavenumber", y="amplitude", hue="label")
    plt.title(f"spectrum plot for channel {vars[0]}")
    if add_ribbon_bar:
        wavenumbers, lower_target_amplitude, upper_target_amplitude, lower_pred_amplitude, upper_pred_amplitude = obtain_freq_ranges(data)
        plt.fill_between(wavenumbers, lower_pred_amplitude, upper_pred_amplitude)
        plt.fill_between(wavenumbers, lower_target_amplitude, upper_target_amplitude)
    plt.savefig(f"figs/spectrum/{model_name}/spectrum_plot_{model_name}_test_range={predict_range}_ribbon={add_ribbon_bar}_{type}.png")
    plt.close()

"""
Method 1: plot the frequency spectrum of a pred vs target batch
          generates one plot for each channel 
input: pred: Tensor (b, c, h, w)   
       target: Tensor (b, c, h, w)
"""

def one_step_plot_spectrum(pred, target, vars, model_name, batch_id = None, predict_range=72, add_ribbon_bar=False, type="fft"):
    print(f"visualizing data of shape {target.shape}")
    n_channels = pred.shape[1]
    
    if type == "fft":
        pred_hat = fft.fftn(pred, dim=(-2,-1))
        pred_target = fft.fftn(target, dim=(-2,-1))
    else: # modified to be sht
        nlat, nlon = target.shape[-2], target.shape[-1]
        lmax = mmax = nlat
        sht = RealSHT(nlat, nlon, lmax=lmax, mmax=mmax)
        pred_hat = sht(pred)  # (b,c,h,h)
        pred_target = sht(target) # (b,c,h,h)
        
    if len(vars) == 1:
        one_step_plot_spectrum_single(pred_hat, pred_target, vars, model_name, predict_range, add_ribbon_bar, type=type)
        return 
    n_rows = n_channels // 2
    fig, ax = plt.subplots(n_rows, 2)
    fig.set_figheight(10 * n_rows)
    fig.set_figwidth(10)
    for i in range(n_rows):
        idx1 = i * 2 # first plot index
        idx2 = i * 2 + 1  # second plot index, same row
        data1 = obtain_freq_vars(pred_hat, pred_target, idx1, type=type)
        ax1 = ax[i,0] if n_rows > 1 else ax[0]        
        sns.lineplot(data=data1, ax=ax1, x="wavenumber", y="amplitude", hue="label")
        ax1.set_title(f"spectrum plot for channel {vars[idx1]}")
        if add_ribbon_bar:
            wavenumbers, lower_target_amplitude, upper_target_amplitude, lower_pred_amplitude, upper_pred_amplitude = obtain_freq_ranges(data1)
            ax1.fill_between(wavenumbers, lower_pred_amplitude, upper_pred_amplitude)
            ax1.fill_between(wavenumbers, lower_target_amplitude, upper_target_amplitude)

        # second plot, same row 
        if i * 2 + 1 >= n_channels: 
            break 
        data2 = obtain_freq_vars(pred_hat, pred_target, idx2, type=type)
        ax2 = ax[i,1] if n_rows > 1 else ax[1]        
        sns.lineplot(data=data2, ax=ax2, x="wavenumber", y="amplitude", hue="label")
        ax2.set_title(f"spectrum plot for channel {vars[idx2]}")
        if add_ribbon_bar:
            wavenumbers, lower_target_amplitude, upper_target_amplitude, lower_pred_amplitude, upper_pred_amplitude = obtain_freq_ranges(data2)
            ax2.fill_between(wavenumbers, lower_pred_amplitude, upper_pred_amplitude)
            ax2.fill_between(wavenumbers, lower_target_amplitude, upper_target_amplitude)
    plt.savefig(f"figs/spectrum/{model_name}/spectrum_plot_{model_name}_{batch_id}_range={predict_range}_ribbon={add_ribbon_bar}_{type}.png")
    plt.close()



"""
TODO: implement a workaround for pytorch lightning training: add a visualization for the full batch test dataset 
for visualization on the entire test dataset 
- input: model, dataloader
- generates: fullbatch target + fullbatch prediction 
- output: one step plot spectrum of entire test dataset 
"""

"""
TODO: add a power spectrum plot with bins 
"""




# tests 

if __name__ == "__main__":
    pred = torch.randn((32, 3, 64, 128))
    target = torch.randn((32, 3, 64, 128))
    vars = ["test_vars1", "test_vars2"]
    one_step_plot_spectrum(pred, target, vars, "test", batch_id = 1, predict_range=72, add_ribbon_bar=False, type="sht")