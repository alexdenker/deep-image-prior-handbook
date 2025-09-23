import torch 
import numpy as np 
import os 
from PIL import Image 

def track_best_psnr_output(best_psnr_output):
    """
    Callback to track and save the output with the best PSNR.
    Usage: pass as a lambda with the required dict, e.g.
    best_output = {'value': 0, 'index': 0, 'output': None}
    cb = lambda i, x_pred, loss, mse_loss, psnr: track_best_psnr_output(best_output)(i, x_pred, loss, mse_loss, psnr)
    """
    def callback(i, x_pred, loss, mse_loss, psnr):
        if psnr > best_psnr_output['value']:
            best_psnr_output['value'] = psnr
            best_psnr_output['index'] = i
            best_psnr_output['reco'] = x_pred.detach().cpu().clone()
    return callback

def save_images(image_path, skip):

    def callback(i, x_pred, loss, mse_loss, psnr):
        if i % skip == 0:
            img = torch.clamp(x_pred.detach().cpu(), 0,1).numpy()[0,0] * 255
            img = img.astype(np.uint8)
            Image.fromarray(img).save(os.path.join(image_path, f"reco_{i:05d}.png"))
            pass 
    return callback  



def early_stopping(patience, delta, w, variance_list, best_psnr):
    """
    Early stopping based on the running variance of the prediction. 
    Variance calculated using a sliding window approach.
    
    patience: number of iterations to wait after last minimum before stopping
    delta: threshold factor for new minimum (e.g. 0.99 means new minimum must be at least 1% lower)
    w: window size for variance calculation
    variance_list: list to store variance values
    best_psnr: dict to store {'value': 0, 'index': 0, 'output': None}
    """

    g_min = float('inf')
    i_min = 0 # early stopping index 
    stopped = False
    x_buffer = []

    def callback(i, x_pred, loss, mse_loss, psnr):
        nonlocal x_buffer, g_min, i_min, stopped
        
        x = x_pred.detach().cpu().numpy().ravel()
        x_buffer.append(x)
        if len(x_buffer) > w:
            x_buffer.pop(0)
        
        running_var = np.mean(np.var(x_buffer, axis=0))
        variance_list.append(running_var)
        if len(x_buffer) == w and not stopped:
            g_i = running_var
            if g_i < delta * g_min:
                g_min = g_i
                i_min = i
                best_psnr['value'] = psnr
                best_psnr['index'] = i_min
                best_psnr['reco'] = x_pred.detach().cpu().clone()

            if i >= i_min + patience:
                stopped = True 
        
    return callback


def early_stopping_welfords(patience, delta, variance_list, best_psnr):
    """
    Early stopping based on the running variance of the prediction. 
    Variance calculated using Welfords online algorithm. 
    
    """

    g_min = float('inf')
    i_min = 0 # early stopping index 
    stopped = False
    n = 0
    mean = None
    M2 = None
    start_after = 100 # start calculating variance after these many iterations

    def callback(i, x_pred, loss, mse_loss, psnr):
        nonlocal n, mean, M2, g_min, i_min, stopped
        
        x = x_pred.detach().cpu().numpy().ravel()
        if mean is None:
            mean = np.zeros_like(x)
            M2 = np.zeros_like(x)
        
        # Welford update
        n += 1
        delta_x = x - mean
        mean += delta_x / n
        delta2 = x - mean
        M2 += delta_x * delta2
        
        # compute current variance (mean of all elements)
        running_var = np.mean(M2 / n) if n > 0 else 0.0
        variance_list.append(running_var)
        if i > start_after and not stopped:
            g_i = running_var
            if g_i < delta * g_min:
                g_min = g_i
                i_min = i
                best_psnr['value'] = psnr
                best_psnr['index'] = i_min
                best_psnr['reco'] = x_pred.detach().cpu().clone()
                print(f"New minimum variance {g_min:.6f} at iteration {i_min}, PSNR: {best_psnr['value']:.2f}")
            if i >= i_min + patience:
                stopped = True 
        
    return callback