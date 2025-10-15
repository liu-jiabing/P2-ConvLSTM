import numpy as np
import torch
from skimage.metrics import structural_similarity as cal_ssim

# ==============================================================================
# 1. Individual Metric Functions (Adapted for Batches)
#    These functions now expect NumPy arrays with shape (B, T, C, H, W)
#    and compute the metric over the entire batch.
# ==============================================================================

def MAE(pred, true):
    """Computes Mean Absolute Error for a batch."""
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    """Computes Mean Squared Error for a batch."""
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    """Computes Root Mean Squared Error for a batch."""
    return np.sqrt(MSE(pred, true))

def PSNR(pred, true):
    """
    Computes Peak Signal-to-Noise Ratio for a batch.
    Assumes data is normalized to [0, 1].
    """
    mse = MSE(pred, true)
    if mse == 0:
        return float('inf')
    # Assuming the data range is 1.0 for normalized data
    return 20. * np.log10(1.0 / np.sqrt(mse))

# --- [NEW] Added SNR metric function ---
def SNR(pred, true):
    """
    Computes Signal-to-Noise Ratio for a batch, expressed in dB.
    Formula: 10 * log10( P_signal / P_noise )
    where P_signal is the power of the true signal (mean of squares)
    and P_noise is the power of the error signal.
    """
    signal_power = np.mean(true ** 2)
    noise = true - pred
    noise_power = np.mean(noise ** 2)

    # Handle the case of a perfect prediction (zero noise)
    if noise_power == 0:
        return float('inf')
    
    # Ensure the ratio is positive before taking the log
    snr_ratio = signal_power / noise_power
    if snr_ratio <= 0:
        return 0.0 # A non-positive ratio doesn't make sense; return a floor value.
        
    return 10 * np.log10(snr_ratio)

def SSIM(pred, true, data_range=1.0, default_win=7):
    """
    Computes Structural Similarity Index Measure for a batch.
    Averages SSIM over all images in the batch and sequence.
    """
    batch_size, seq_len, _, height, width = pred.shape
    
    ssim_scores = []
    for b in range(batch_size):
        for t in range(seq_len):
            pred_img, true_img = pred[b, t], true[b, t] # Shape (C, H, W)
            
            pred_img_hwc = np.transpose(pred_img, (1, 2, 0))
            true_img_hwc = np.transpose(true_img, (1, 2, 0))
            
            if pred_img_hwc.shape[2] == 1:
                pred_img_hwc, true_img_hwc = np.squeeze(pred_img_hwc, axis=2), np.squeeze(true_img_hwc, axis=2)
                channel_axis = None
            else:
                channel_axis = 2
            
            win_size = min(default_win, height, width)
            if win_size % 2 == 0: win_size -=1

            if win_size < 3:
                ssim_scores.append(1.0)
                continue

            ssim_val = cal_ssim(
                pred_img_hwc, 
                true_img_hwc, 
                win_size=win_size, 
                channel_axis=channel_axis, 
                data_range=data_range
            )
            ssim_scores.append(ssim_val)
            
    return np.mean(ssim_scores)


# ==============================================================================
# 2. Main Metrics Calculator Class
#    This class will be used in the training script to manage all evaluations.
# ==============================================================================

class MetricsCalculator:
    """A class to manage computation of multiple metrics over an epoch."""
    
    def __init__(self, metrics_list=['mae', 'mse', 'ssim', 'psnr']):
        """
        Args:
            metrics_list (list): A list of strings with the names of metrics to compute.
                                 Supported: 'mae', 'mse', 'rmse', 'ssim', 'psnr', 'snr'.
        """
        self.metrics_list = [m.lower() for m in metrics_list]
        self._results = {metric: [] for metric in self.metrics_list}
        
        # --- [MODIFIED] Register the new SNR function ---
        self.metric_functions = {
            'mae': MAE,
            'mse': MSE,
            'rmse': RMSE,
            'ssim': SSIM,
            'psnr': PSNR,
            'snr': SNR,
        }
        
        for metric in self.metrics_list:
            if metric not in self.metric_functions:
                raise ValueError(f"Metric '{metric}' is not supported.")

    def reset(self):
        """Resets the stored results for a new epoch."""
        self._results = {metric: [] for metric in self.metrics_list}

    @torch.no_grad()
    def update(self, pred_tensor, true_tensor):
        """
        Update the metric results with a new batch of predictions and ground truths.
        
        Args:
            pred_tensor (torch.Tensor): Predictions from the model (on GPU or CPU).
            true_tensor (torch.Tensor): Ground truth data (on GPU or CPU).
        """
        pred_np = pred_tensor.detach().cpu().numpy()
        true_np = true_tensor.detach().cpu().numpy()
        
        for metric in self.metrics_list:
            metric_func = self.metric_functions[metric]
            value = metric_func(pred_np, true_np)
            self._results[metric].append(value)
            
    def compute(self):
        """
        Compute the final averaged metrics over all updated batches.
        
        Returns:
            dict: A dictionary where keys are metric names and values are the final scores.
        """
        final_metrics = {
            metric: np.mean(values) for metric, values in self._results.items()
        }
        return final_metrics

# ==============================================================================
# 3. Test Entrypoint
# ==============================================================================

if __name__ == '__main__':
    print("--- Testing MetricsCalculator ---")
    
    # --- [MODIFIED] Add 'snr' to the list of metrics to test ---
    calculator = MetricsCalculator(metrics_list=['mae', 'mse', 'ssim', 'psnr', 'snr'])
    
    # Simulate a few training batches
    for i in range(3): # 3 batches
        print(f"\n--- Batch {i+1} ---")
        
        # Create fake data as PyTorch Tensors
        # Shape: (Batch, Time, Channels, Height, Width)
        pred_torch = torch.rand(4, 12, 1, 32, 64)
        true_torch = pred_torch * 0.8 + 0.1 # Make pred somewhat similar to true for more realistic metric values
        if i == 1:
            true_torch = pred_torch.clone() # Simulate a perfect prediction for one batch
            print("Injecting a perfect prediction for this batch to test inf values.")
        
        # Update the calculator with the batch data
        calculator.update(pred_torch, true_torch)
        print("Calculator updated.")

    # After the epoch (all batches), compute the final results
    final_scores = calculator.compute()
    
    print("\n--- Final Epoch Results ---")
    for name, score in final_scores.items():
        print(f"{name.upper()}: {score:.4f}")
        
    print("\n--- Testing Reset ---")
    calculator.reset()
    print(f"Results after reset: {calculator._results}")