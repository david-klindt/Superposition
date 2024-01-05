import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import MiniBatchKMeans


def normalize(data, tol=1e-6):
    data = data.copy()
    norms = np.sqrt((data ** 2).sum(axis=1))[:, None].clip(min=tol)
    return data / norms


def compute_kmeans(data_train: np.ndarray, data_test: np.ndarray, num_dic: int, seed: int = 42,
                   normalized: bool = True, align: bool = False, distances: bool = False) -> tuple:
    """
    This function computes k-means clustering on the input data and returns the kmeans object and distances.

    Parameters:
    data train (np.ndarray): The input data (samples x features) for clustering. 
    data test (np.ndarray): The input data (samples x features) for psychophysics. 
    num_dic (int): The number of clusters to form.
    seed (int, optional): The seed for the random number generator. Defaults to 42.
    normalized (bool, optional): Whether to normalize the data to unit norm, i.e., cluster cosine similarities.
    align (bool, optional): Whether to initialize the centroids as the neurons.
    distances (bool, optional): Whether to use kmeans distances instead of projections.

    Returns:
    tuple: A tuple containing the kmeans object, distances and projections.
    """
    if align and num_dic == data_train.shape[1]:
        print('try to align with neurons, init as eye')
        init = np.eye(num_dic)
        #n_init = 1
    else:
        init = 'k-means++'
        #n_init = 10
    if normalized:
        data_train = normalize(data_train)
        data_test = normalize(data_test)
    kmeans = MiniBatchKMeans(
        n_clusters=num_dic,
        random_state=seed,
        #n_init=n_init,
        verbose=False,
        init=init,
    ).fit(data_train)
    if distances:
        activations_train = - kmeans.transform(data_train)  # distances_train
        activations_test = - kmeans.transform(data_test)  # distances_test
    else:
        activations_train = data_train @ kmeans.cluster_centers_.T  # projections_train
        activations_test = data_test @ kmeans.cluster_centers_.T  # projections_test
    return kmeans, activations_train, activations_test


def fit_power_law(var_exp: np.ndarray, plot: bool = True) -> np.ndarray:
    """
    This function fits a power law to the variance explained by the principal components.
    Compare to: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6642054/

    Parameters:
    var_exp (np.ndarray): The variance explained by the principal components.
    plot (bool, optional): Whether to plot the variance explained and the fitted power law. Defaults to True.

    Returns:
    np.ndarray: The coefficients of the fitted power law.
    """
    d = 32 * 32 * 3
    alpha = -1 - 2 / d
    N = var_exp.shape[0]
    x = np.arange(1, N + 1)
    X = np.log(x)[:, None]
    var_exp = var_exp.copy()
    if np.any(var_exp < 1e-9):
        print(np.min(var_exp))
        ind_bad = var_exp < 1e-9
        ind_good = np.logical_not(ind_bad)
        var_exp[ind_bad] = np.min(var_exp[ind_good])
        print(np.min(var_exp))
    Y = np.log(var_exp)
    ind = np.arange(N)
    ind = np.logical_and(ind > 0.05 * N, ind < 0.5 * N) # they fit to 11-500
    reg = LinearRegression().fit(X[ind], Y[ind])
    Y_ = np.exp(reg.predict(X))
    if plot:
        plt.plot(x, var_exp, label='Data')
        plt.plot(x, Y_, label='Linear fit:\n' + r'$\alpha=%.4f$' % (
            reg.coef_) + '\n' + r'$\beta=%.4f$' % reg.intercept_)
        plt.plot(x, x ** alpha, label='Critical:\n' + r'$\alpha=%.4f$' % alpha)
        plt.semilogx()
        plt.semilogy()
        plt.legend()
        plt.title("Power Law Analysis (Stringer et al., 2019)")
    else:
        return reg.coef_
    

def get_svd(data: np.ndarray, filename: str, show: bool = False, compute_only: bool = False):
    """
    This function computes the singular value decomposition (SVD) of the input data.

    Parameters:
    data (np.ndarray): The input data for which the SVD is to be computed.
    filename (str): The name of the file where the plot will be saved.
    show (bool, optional): Whether to show the plot or not. Defaults to False.
    compute_only (bool, optional): Whether to only compute the SVD without plotting. Defaults to False.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray]: The singular values, the left singular vectors, and the right singular vectors.
    """
    data = data.copy()  # images x neurons
    data -= np.mean(data, axis=0, keepdims=True)
    data /= np.std(data, axis=0, keepdims=True).clip(min=1e-9)
    u, s, v = np.linalg.svd(data, full_matrices=False)
    if compute_only:
        return u, s, v
    var_exp = s ** 2
    var_exp /= np.sum(var_exp)
    plt.figure(figsize=(12, 4))
    for i in range(2):
        plt.subplot(1, 2, 1 + i)
        if i == 0:
            plt.plot(var_exp, '.-')
            plt.ylim(-.01, var_exp.max() * 1.1)
            plt.title('Spectrum\n0.95 Var Exp at n=%s' % (
                np.where(np.cumsum(var_exp) > .95)[0][0]
            ))
        else:
            fit_power_law(var_exp)
        plt.xlabel('SVD component')
        plt.ylabel('Variance Explained')
        plt.grid()
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(filename)
        plt.clf()
    return u, s, v