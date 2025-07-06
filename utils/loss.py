import torch
import torch.nn as nn
import torch.nn.functional as F

def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and return magnitude and phase."""
    x_stft = torch.stft(x, n_fft=fft_size, hop_length=hop_size, win_length=win_length, window=window, return_complex=False)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]
    
    # Correctly handling the output of torch.stft
    mag = torch.sqrt(real**2 + imag**2 + 1e-9)
    
    return mag

class SpectralConvergengeLoss(nn.Module):
    """Spectral convergence loss module."""
    def __init__(self):
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of ground-truth signal (B, #frames, #freq_bins).
        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")

class LogSTFTMagnitudeLoss(nn.Module):
    """Log STFT magnitude loss module."""
    def __init__(self):
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of ground-truth signal (B, #frames, #freq_bins).
        """
        eps = 1e-7
        return F.l1_loss(torch.log(x_mag + eps), torch.log(y_mag + eps))

class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss module."""
    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240], window="hann_window", factor_sc=0.1, factor_mag=0.1):
        super(MultiResolutionSTFTLoss, self).__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.window = getattr(torch, window)
        
        self.sc_loss = SpectralConvergengeLoss()
        self.mag_loss = LogSTFTMagnitudeLoss()
        
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self, y_hat, y):
        """
        Args:
            y_hat (Tensor): Predicted waveform (B, T).
            y (Tensor): Ground-truth waveform (B, T).
        """
        sc_loss = 0.0
        mag_loss = 0.0
        
        # Ensure y and y_hat are on the same device
        device = y.device
        y_hat = y_hat.to(device)

        # Pad or trim to same length
        if y_hat.size(1) > y.size(1):
            y_hat = y_hat[:, :y.size(1)]
        else:
            y_hat = F.pad(y_hat, (0, y.size(1) - y_hat.size(1)))

        for (fft, hop, win) in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            win_tensor = self.window(win, periodic=True).to(device)
            y_mag = stft(y, fft, hop, win, win_tensor)
            y_hat_mag = stft(y_hat, fft, hop, win, win_tensor)
            
            sc_loss += self.sc_loss(y_hat_mag, y_mag)
            mag_loss += self.mag_loss(y_hat_mag, y_mag)
            
        sc_loss /= len(self.fft_sizes)
        mag_loss /= len(self.fft_sizes)
        
        total_loss = sc_loss * self.factor_sc + mag_loss * self.factor_mag
        
        return total_loss 