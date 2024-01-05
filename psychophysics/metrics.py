import sys
import torch
import numpy as np

# use custom lpips version with batch support
# https://github.com/david-klindt/PerceptualSimilarity/tree/batched
from lpips import LPIPS


def get_lpips(net: str = 'alex') -> callable:
    """
    This function returns a metric for the perceptual similarity between two images.

    Parameters:
    device (torch.device): The device on which to perform the computation.
    net (str, optional): The network to use for the computation. Defaults to 'alex' (best forward scores). 'vgg' (closer to "traditional" perceptual loss, when used for optimization).

    Returns:
    function: A function that takes two images and returns a metric of their perceptual similarity per layer.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = LPIPS(net=net).to(device)
    def metric(a, b):
        img0 = torch.tensor(a, dtype=torch.float32, device=device)
        img1 = torch.tensor(b, dtype=torch.float32, device=device)
        with torch.no_grad():
            _, ret = loss_fn(img0, img1, normalize=True, retPerLayer=True)
            output = - torch.stack(ret, dim=-1).detach().cpu().numpy()
        return output
    return metric


def l2_metric(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    This function computes the pairwise L2 distance between two numpy arrays for each batch element.
    The batch is the first dimension in both 'a' and 'b'. For example, they could be image tensors of shape NCHW.

    Parameters:
    a (np.ndarray): The first input array. It should be a batch of data points.
    b (np.ndarray): The second input array. It should be a batch of data points.

    Returns:
    np.ndarray: The pairwise L2 distance between the batches of input arrays.
    """
    sum_over_axis = tuple(np.arange(2, len(a.shape) + 1))
    dist = np.sum((a[:, None] - b[None]) ** 2, axis=sum_over_axis)
    return -np.sqrt(dist)


def color_metric(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    This function computes the color metric between two numpy arrays, 
    where each array is an image batch of shape NCHW (channel first).

    Parameters:
    a (np.ndarray): The first input array.
    b (np.ndarray): The second input array.

    Returns:
    np.ndarray: The color metric between the input arrays.
    """
    color_a = np.mean(a, (2, 3))
    color_b = np.mean(b, (2, 3))
    return l2_metric(color_a, color_b)


def get_image_metric() -> callable:
    """
    This function computes all image metrics between two numpy arrays, 
    where each array is an image batch of shape NCHW (channel first).

    Parameters:
    a (np.ndarray): The first input array.
    b (np.ndarray): The second input array.

    Returns:
    np.ndarray: The color metric between the input arrays.
    """
    lpips_metric = get_lpips()
    def metric(a, b):
        output = np.concatenate([
            l2_metric(a, b)[:, :, None], 
            color_metric(a, b)[:, :, None],
            lpips_metric(a, b), 
        ], 2)
        return output
    return metric


def label_metric(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    This function checks if two numpy arrays are equal.

    Parameters:
    a (np.ndarray): The first input array.
    b (np.ndarray): The second input array.

    Returns:
    np.ndarray: A boolean array where True indicates the elements are equal and False indicates they are not.
    """
    return np.float32(a[:, None] == b[None])


def get_metric(metric_name):
    if metric_name == 'lpips':
        return get_lpips()
    elif metric_name == 'l2':
        return l2_metric
    elif metric_name == 'color':
        return color_metric
    elif metric_name == 'image':
        return get_image_metric()
    elif metric_name == 'label':
        return label_metric
    else:
        raise ValueError('Metric (%s) not implemented!' % metric_name)


if __name__ == "__main__":
    lpips_metric = get_metric('lpips')
    # Test images
    img_a = np.random.normal(0, 1, (2, 3, 224, 224))
    img_b = np.random.normal(0, 1, (3, 3, 224, 224))
    # Compute metric
    print('Test LPIPS')
    print("Output shape (should be 2 x 3 x 5): ", lpips_metric(img_a, img_b).shape)
    print(f"Perceptual similarity (should be symmetric and 0 on diagonal): {lpips_metric(img_a, img_a).mean(2)}")
    print(f"Same result? (i.e. test for deterministic): {get_metric('lpips')(img_a, img_a).mean(2)}")

    print("Test Color Metric")
    print("Output shape (should be 2 x 3): ", color_metric(img_a, img_b).shape)
    print(f"Perceptual similarity (should be symmetric and 0 on diagonal): {color_metric(img_a, img_a)}")

    print("Test L2 Metric")
    print("Output shape (should be 2 x 3): ", l2_metric(img_a, img_b).shape)
    print(f"Perceptual similarity (should be symmetric and 0 on diagonal): {l2_metric(img_a, img_a)}")
    
    print('Test Image Metrics (l2, color, lpips)')
    image_metric = get_metric('image')
    print("Output shape (should be 2 x 3 x 7): ", image_metric(img_a, img_b).shape)
    print(f"Perceptual similarity (should be symmetric and 0 on diagonal): {image_metric(img_a, img_a).mean(2)}")

    print("Test Label Metric")
    label_a = np.random.normal(0, 1, (2,))
    label_b = np.random.normal(0, 1, (3,))
    print("Output shape (should be 2 x 3): ", label_metric(label_a, label_b).shape)
    print(f"Perceptual similarity (should be symmetric and 0 on diagonal): {label_metric(label_a, label_a)}")
