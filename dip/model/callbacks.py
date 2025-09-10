

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

