import sys
import torch

# use custom lpips version with batch support
# https://github.com/david-klindt/PerceptualSimilarity/tree/batched
from Superposition_v1.psychophysics.metrics import LPIPS


def get_metric(device: torch.device, net: str = 'alex'):
    """
    This function returns a metric for the perceptual similarity between two images.

    Parameters:
    device (torch.device): The device on which to perform the computation.
    net (str, optional): The network to use for the computation. Defaults to 'alex' (best forward scores). 'vgg' (closer to "traditional" perceptual loss, when used for optimization).

    Returns:
    function: A function that takes two images and returns a metric of their perceptual similarity per layer.
    """
    
    loss_fn = LPIPS(net=net).to(device)
    def metric(a, b):
        img0 = torch.tensor(a, dtype=torch.float32, device=device)
        img1 = torch.tensor(b, dtype=torch.float32, device=device)
        with torch.no_grad():
            _, ret = loss_fn(img0, img1, normalize=True, retPerLayer=True)
            output = - torch.stack(ret, dim=-1).detach().cpu().numpy()
        return output
    return metric