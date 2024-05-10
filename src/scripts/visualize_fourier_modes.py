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

"""
update: use mean over minibatch, and provide the power spectrum (energy) plot
see: https://heraclixus.github.io/mode_vis/ 
"""
def obtain_spectrum_fft(x, x_pred, idx, n_bins=32):
    x_hat = torch.fft.fftn(x, dim=(-2,-1))[idx, :, :]
    x_hat_pred = torch.fft.fftn(x_pred, dim=(-2,-1))[idx, :, :]
    x_hat_centered = torch.fft.fftshift(x_hat, dim=(-2,-1))
    x_hat_centered_pred = torch.fft.fftshift(x_hat_pred, dim=(-2,-1))
    wavenumbers = torch.abs(x_hat-x_hat_centered).flatten()
    wavenumbers_pred = torch.abs(x_hat_pred - x_hat_centered_pred).flatten()
    amptliude = torch.abs(x_hat)
    amplitude_pred = torch.abs(x_hat_pred)
    energy, energy_pred = amptliude ** 2, amplitude_pred ** 2 
    energy, energy_pred = energy.flatten(), energy_pred.flatten()
    max_wavenumber = max([torch.max(wavenumbers), torch.max(wavenumbers_pred)])
    bins = torch.arange(0, max_wavenumber, torch.div(max_wavenumber, n_bins, rounding_mode="floor"))
    if torch.max(wavenumbers) > torch.max(wavenumbers_pred):
        indices = torch.bucketize(wavenumbers, bins)
    else:
        indices = torch.bucketize(wavenumbers_pred, bins)
    radial_spectrum_profile = torch.zeros(len(bins))
    radial_spectrum_profile_pred = torch.zeros(len(bins))
    for i in range(1, len(bins)):
        mask = indices == i
        if torch.any(mask):
            radial_spectrum_profile[i] = torch.sum(energy[mask])
            radial_spectrum_profile_pred[i] = torch.sum(energy_pred[mask])
    wavenumbers = np.concatenate([bins.numpy(), bins.numpy()])
    spectrum = np.concatenate([radial_spectrum_profile.numpy(), radial_spectrum_profile_pred.numpy()])
    labels = np.array(["energy_target"] * len(bins) + ["energy_pred"] * len(bins))
    data = pd.DataFrame({
        "wavenumber": wavenumbers,
        "spectrum energy": spectrum, 
        "label": labels 
    })

    data_diff = pd.DataFrame({
        "wavenumber": bins.numpy(),
        "Error energy spectrum": np.abs(radial_spectrum_profile.numpy() - radial_spectrum_profile_pred.numpy())
    })
    return data, data_diff


def obtain_spectrum_sht(x, x_pred, idx):
    nlat, nlon = x.shape[-2], x.shape[-1]
    lmax = mmax = nlat
    sht = RealSHT(nlat, nlon, lmax, mmax).to(x.device)
    x_hat, x_pred_hat = sht(x), sht(x_pred)
    amplitude, amplitude_pred = torch.abs(x_hat[idx,:,:]), torch.abs(x_pred_hat[idx,:,:]) 
    energy, energy_pred = torch.sum(amplitude ** 2, dim=-1), torch.sum(amplitude_pred ** 2, dim=-1)
    degrees = np.array([i for i in range(1, amplitude.shape[-1]+1)])
    degrees_ = np.concatenate([degrees, degrees])
    spectrum = np.concatenate([energy.numpy(), energy_pred.numpy()])
    labels = np.array(["energy_target"] * len(degrees) + ["energy_pred"] * len(degrees))
    data = pd.DataFrame({
        "degree": degrees_,
        "spectrum energy": spectrum,
        "label": labels
    })
    
    data_diff = pd.DataFrame({
        "degree": degrees, 
        "Error energy spectrum": np.abs(energy.numpy() - energy_pred.numpy())
    })

    return data, data_diff


def obtain_freq_vars(x, x_pred, idx, type="fft", n_bins=32):
    x = torch.mean(x, dim=0).to("cpu")
    x_pred = torch.mean(x_pred, dim=0).to("cpu")
    if type == "fft":
        return obtain_spectrum_fft(x, x_pred, idx, n_bins)
    else:
        return obtain_spectrum_sht(x, x_pred, idx)


"""
a specific plotting function when there's only one feature
"""
def one_step_plot_spectrum_single(x, x_pred, vars, model_name, predict_range=72, type="fft"):
    fig = plt.figure(figsize=(10,5))
    data, data_diff = obtain_freq_vars(x, x_pred, 0, type=type)
    xlabel = "wavenumber" if type == "fft" else "degree"
    sns.lineplot(data=data, x=xlabel, y="spectrum energy", hue="label")
    plt.title(f"spectrum energy plot for channel {vars[0]}")
    plt.savefig(f"figs/spectrum/{model_name}/spectrum_plot_{model_name}_test_range={predict_range}_{type}.png")
    plt.close()
    fig = plt.figure(figsize=(10,5))
    sns.lineplot(data=data_diff, x="wavenumber", y="Error energy spectrum")
    plt.title(f"Error in spectrum energy for {model_name}, predict_range = {predict_range}")
    plt.savefig(f"figs/spectrum/{model_name}/spectrum_error_{model_name}_test_range={predict_range}_{type}.png")

"""
Method 1: plot the frequency spectrum of a pred vs target batch
          generates one plot for each channel 
input: pred: Tensor (b, c, h, w)   
       target: Tensor (b, c, h, w)
"""

def one_step_plot_spectrum(pred, target, vars, model_name, batch_id = None, predict_range=72, type="fft"):
    print(f"visualizing data of shape {target.shape}")
    n_channels = pred.shape[1]        
    if len(vars) == 1:
        one_step_plot_spectrum_single(pred, target, vars, model_name, predict_range, type=type)
        return 
    n_rows = n_channels // 2
    fig, ax = plt.subplots(n_rows, 2)
    fig.set_figheight(10 * n_rows)
    fig.set_figwidth(10)
    xlabel = "wavenumber" if type == "fft" else "degree"
    for i in range(n_rows):
        idx1 = i * 2 # first plot index
        idx2 = i * 2 + 1  # second plot index, same row
        data1, _ = obtain_freq_vars(pred, target, idx1, type=type)
        ax1 = ax[i,0] if n_rows > 1 else ax[0]     
        sns.lineplot(data=data1, ax=ax1, x=xlabel, y="spectrum energy", hue="label")
        ax1.set_title(f"spectrum plot for channel {vars[idx1]}")
        # second plot, same row 
        if i * 2 + 1 >= n_channels: 
            break 
        data2, _ = obtain_freq_vars(pred, target, idx2, type=type)
        ax2 = ax[i,1] if n_rows > 1 else ax[1]        
        sns.lineplot(data=data2, ax=ax2, x=xlabel, y="spectrum energy", hue="label")
        ax2.set_title(f"spectrum plot for channel {vars[idx2]}")
    plt.tight_layout()
    plt.savefig(f"figs/spectrum/{model_name}/spectrum_plot_{model_name}_{batch_id}_range={predict_range}_{type}.png")
    plt.close()

    # error plot
    fig, ax = plt.subplots(n_rows, 2)
    fig.set_figheight(10 * n_rows)
    fig.set_figwidth(10)
    for i in range(n_rows):
        idx1 = i * 2 # first plot index
        idx2 = i * 2 + 1  # second plot index, same row
        _, data1 = obtain_freq_vars(pred, target, idx1, type=type)
        ax1 = ax[i,0] if n_rows > 1 else ax[0]     
        sns.lineplot(data=data1, ax=ax1, x=xlabel, y="Error energy spectrum")
        ax1.set_title(f"Error in spectrum energy for channel {vars[idx1]}")
        # second plot, same row 
        if i * 2 + 1 >= n_channels:
            break 
        _, data2 = obtain_freq_vars(pred, target, idx2, type=type)
        ax2 = ax[i,1] if n_rows > 1 else ax[1]        
        sns.lineplot(data=data2, ax=ax2, x=xlabel, y="Error energy spectrum")
        ax2.set_title(f"Error in spectrum energy for channel {vars[idx2]}")
    plt.tight_layout()
    plt.savefig(f"figs/spectrum/{model_name}/spectrum_error_{model_name}_{batch_id}_range={predict_range}_{type}.png")
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
    pred = torch.randn((32, 2, 64, 128))
    target = torch.randn((32, 2, 64, 128))
    vars = ["test_vars1", "test_vars2"]
    one_step_plot_spectrum(pred, target, vars, "test", batch_id = 1, predict_range=72, type="sht")
    one_step_plot_spectrum(pred, target, vars, "test", batch_id = 1, predict_range=72, type="fft")