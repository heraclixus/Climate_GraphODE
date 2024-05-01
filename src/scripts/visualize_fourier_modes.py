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


"""
Helper functions 
"""

def obtain_freq_vars(x, x_pred, idx):
    x_hat = torch.fft.fftshift(x, dim=(-2,-1))[:, idx, :, :]
    x_hat_pred = torch.fft.fftshift(x_pred, dim=(-2,-1))[:, idx, :, :]
    spectrum = torch.abs(x_hat)
    spectrum_pred = torch.abs(x_hat_pred)
    amplitude_mean = spectrum[:, -1, :].mean(dim=0).detach().cpu().numpy()
    amplitude_std = spectrum[:, -1, :].std(dim=0).detach().cpu().numpy()
    amplitude_mean_pred = spectrum_pred[:, -1, :].mean(dim=0).detach().cpu().numpy()
    amplitude_std_pred = spectrum_pred[:, -1, :].std(dim=0).detach().cpu().numpy()
    N = spectrum.shape[-1]
    wavenumbers = torch.fft.fftshift(torch.fft.fftfreq(N)).detach().cpu().numpy()
    wavenumbers *= N
    wavenumbers = np.concatenate([wavenumbers, wavenumbers])
    preds_mean = np.concatenate([amplitude_mean, amplitude_mean_pred])
    preds_std = np.concatenate([amplitude_std, amplitude_std_pred])
    labels = np.array(["amplitude_target"] * N + ["amplitude_pred"] * N)
    data = pd.DataFrame({
        "wavenumber": wavenumbers,
        "prediction": preds_mean,
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
    upper, lower = data.prediction + data.pred_std, data.prediction - data.pred_std 
    upper_target, lower_target = upper[:N], lower[:N]
    upper_pred, lower_pred = upper[N:], lower[N:]
    return wavenumbers, upper_target, lower_target, upper_pred, lower_pred



"""
Method 1: plot the frequency spectrum of a pred vs target batch
          generates one plot for each channel 
input: pred: Tensor (b, c, h, w)   
       target: Tensor (b, c, h, w)
"""

def one_step_plot_spectrum(pred, target, vars, model_name, batch_id = None, add_ribbon_bar=False):
    print(f"visualizing data of shape {target.shape}")
    n_channels = pred.shape[1] 
    pred_hat = fft.fftn(pred, dim=(-2,-1))
    pred_target = fft.fftn(target, dim=(-2,-1))
    
    n_rows = n_channels // 2  + 1 
    fig, ax = plt.subplots(n_rows, 2, figsize=(2*3, n_rows*3))
    fig.set_figheight(10 * n_rows)
    fig.set_figwidth(10)

    for i in range(n_rows):
        idx1 = i * 2 # first plot index
        idx2 = i * 2 + 1  # second plot index, same row
        data1 = obtain_freq_vars(pred_hat, pred_target, idx1)
        if n_rows == 1:
            sns.lineplot(data=data1, ax=ax[i ], x="wavenumber", y="prediction", hue="label")
            ax[0].set_title(f"spectrum plot for channel {vars[idx1]}")
        else:
            sns.lineplot(data=data1, ax=ax[i,0 ], x="wavenumber", y="prediction", hue="label")
            ax[i,0].set_title(f"spectrum plot for channel {vars[idx1]}")
        if add_ribbon_bar:
            wavenumbers, lower_target_amplitude, upper_target_amplitude, lower_pred_amplitude, upper_pred_amplitude = obtain_freq_ranges(data1)
            
            if n_rows == 1:
                ax[0].fill_between(wavenumbers, lower_pred_amplitude, upper_pred_amplitude)
                ax[0].fill_between(wavenumbers, lower_target_amplitude, upper_target_amplitude)
            else:
                ax[i,0].fill_between(wavenumbers, lower_pred_amplitude, upper_pred_amplitude)
                ax[i,0].fill_between(wavenumbers, lower_target_amplitude, upper_target_amplitude)
        # second plot, same row 
        if i * 2 + 1 >= n_channels: 
            break 
        data2 = obtain_freq_vars(pred_hat, pred_target, idx2)
        sns.lineplot(data=data2, ax=ax[i, 1], x="wavenumber", y="prediction", hue="label")
        ax[i,1].set_title(f"spectrum plot for channel {vars[idx2]}")
        if add_ribbon_bar:
            wavenumbers, lower_target_amplitude, upper_target_amplitude, lower_pred_amplitude, upper_pred_amplitude = obtain_freq_ranges(data2)
            if n_rows == 1:
                ax[1].fill_between(wavenumbers, lower_pred_amplitude, upper_pred_amplitude)
                ax[1].fill_between(wavenumbers, lower_target_amplitude, upper_target_amplitude)
            else:
                ax[i,1].fill_between(wavenumbers, lower_pred_amplitude, upper_pred_amplitude)
                ax[i,1].fill_between(wavenumbers, lower_target_amplitude, upper_target_amplitude)
    plt.savefig(f"/home/dingy6/Climate_GraphODE/src/plot/spectrum_plot_{model_name}_{batch_id}.png")


"""
TODO: add plotting functions for spherical harmoncis 
"""