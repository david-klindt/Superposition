import numpy as np
from typing import Optional, List
import matplotlib.pyplot as plt

from Superposition_v1.psychophysics.metrics import get_metric


def get_psychophysics_data(
        activations: np.ndarray, 
        t: int = 10, 
        quantiles: Optional[List[float]] = None, 
        seed: int = 42):
    """
    This function is used to get the most activating images (their indices) as train and test for psychophysics.
    See https://arxiv.org/pdf/2307.05471.pdf Appendix A.2

    Parameters:
    activations (np.ndarray): The activations of the units.
    labels (np.ndarray, optional): The labels of the units. Defaults to None.
    t (int, optional): The number of top and bottom units to consider. Defaults to 10.
    quantiles (List[float], optional): The quantiles to consider. Defaults to None.
    seed (int): The seed for the random number generator.

    Returns:
    dict: A dictionary containing the indices of the most activating inputs.
    """
    outputs = dict()
    num_unit = activations.shape[1]
    np.random.seed(seed)
    if type(quantiles) == type(None):
        quantiles = [0]
    for q in quantiles:
        assert q < .5
        outputs['quantile_%s' % q] = {
            'query_top': np.zeros((num_unit, t, 1)).astype(int),
            'reference_top': np.zeros((num_unit, t, 9)).astype(int),
            'query_bottom': np.zeros((num_unit, t, 1)).astype(int),
            'reference_bottom': np.zeros((num_unit, t, 9)).astype(int),
        }
        
        for i in range(num_unit):
            y = activations[:, i].copy()
            # only use distribution center between quantiles, all else set to median
            ind = np.logical_or(
                y < np.quantile(y, q), 
                y > np.quantile(y, 1 - q)
            )
            # make sure there is at least twice as many left as we need
            assert y.shape[0] - np.sum(ind) > t * 10 * 2 * 2
            y[ind] = np.median(y)
            ind_sort_bottom = np.argsort(y)
            ind_sort_top = ind_sort_bottom[::-1]
            ind_query_top = np.random.choice(t, t, replace=False)
            ind_query_bottom = np.random.choice(t, t, replace=False)
            outputs['quantile_%s' % q]['query_top'][i, :, 0] = (
                ind_sort_top[:t][ind_query_top]).copy()
            outputs['quantile_%s' % q]['query_bottom'][i, :, 0] = (
                ind_sort_bottom[:t][ind_query_bottom]).copy()

            for j in range(t):
                ind_reference_top = np.random.choice(9, 9, replace=False)
                ind_reference_bottom = np.random.choice(9, 9, replace=False)

                for k in range(9):
                    outputs['quantile_%s' % q]['reference_top'][i, j, k] = (
                        ind_sort_top[(j + 1) * t:(j + 2) * t][ind_reference_top[k]]).copy()
                    outputs['quantile_%s' % q]['reference_bottom'][i, j, k] = (
                        ind_sort_bottom[(j + 1) * t:(j + 2) * t][ind_reference_bottom[k]]).copy()
                    
    return outputs


def run_psychophysics(data: dict, inputs: np.ndarray, labels: Optional[np.ndarray]):
    """
    This function runs the psychophysics experiment on the given data.

    Parameters:
    data (dict): A dictionary containing the indices of the most activating inputs.
    inputs (np.ndarray): The input data for the experiment.
    labels (Optional[np.ndarray]): The labels for the input data. 
        If not provided, only the 'lpips' and 'color' metrics will be used.

    Returns:
    dict: A dictionary containing the results of the psychophysics experiment. 
    The keys are the quantiles, and the values are dictionaries containing the 
    results for each metric ('lpips', 'color', and 'label' if labels are provided). 
    Each metric dictionary contains the 'logits', 'log_diffs', and 'accuracy' results.
    """
    pool_fun = np.mean  # min/max would not make sense with label
    quantiles = list(data.keys())
    num_unit = data[quantiles[0]]['query_top'].shape[0]
    t = data[quantiles[0]]['query_top'].shape[1]
    lpips_metric = get_metric()
    metrics = {'lpips': lpips_metric, 'color': color_metric}
    if labels:
        metrics['label'] = label_metric
    output = dict()
    for key_q in quantiles:
        output[key_q] = dict()
        for key_m in metrics:
            metric = metrics[key_m]
            print('Run Psychophysics:', key_q, key_m)
            if key_m == 'lpips':
                logits = np.zeros((num_unit, t, 2, 2, 5))
            else:
                logits = np.zeros((num_unit, t, 2, 2))
            for ind in range(num_unit):
                for batch in range(t):
                    if key_m == 'label':
                        reference_top = data[key_q]['label_reference_top'][ind, batch]
                        reference_bottom = data[key_q]['label_reference_bottom'][ind, batch]
                        query_top = data[key_q]['label_query_top'][ind, batch]
                        query_bottom = data[key_q]['label_query_bottom'][ind, batch]
                    else:
                        reference_top = inputs[data[key_q]['reference_top'][ind, batch]]
                        reference_bottom = inputs[data[key_q]['reference_bottom'][ind, batch]]
                        query_top = inputs[data[key_q]['query_top'][ind, batch]]
                        query_bottom = inputs[data[key_q]['query_bottom'][ind, batch]]

                    # logit for top
                    logits[ind, batch, 0, 0] = pool_fun(metric(
                        reference_top, query_top)) # evidence from top
                    logits[ind, batch, 0, 1] = pool_fun(metric(
                        reference_bottom, query_bottom)) # evidence from bottom
                    # logit bottom
                    logits[ind, batch, 1, 0] = pool_fun(metric(
                        reference_top, query_bottom)) # evidence from top
                    logits[ind, batch, 1, 1] = pool_fun(metric(
                        reference_bottom, query_top)) # evidence from bottom
                    
            output[key_q][key_m] = dict()
            output[key_q][key_m]['logits'] = logits.copy()
            log_diffs = logits.sum(3)
            if key_m == 'lpips':
                log_diffs = log_diffs.mean(3)
            log_diffs = log_diffs[:, :, 0] - log_diffs[:, :, 1]
            output[key_q][key_m]['log_diffs'] = log_diffs.mean(1)
            output[key_q][key_m]['accuracy'] = (log_diffs > 0).mean(1)
            if key_m == 'lpips':
                for l in range(5):
                    key_m_l = key_m + '_layer%s' % l
                    output[key_q][key_m_l] = dict()
                    log_diffs = logits[..., l].sum(3)
                    log_diffs = log_diffs[:, :, 0] - log_diffs[:, :, 1]
                    output[key_q][key_m_l]['log_diffs'] = log_diffs.mean(1)
                    output[key_q][key_m_l]['accuracy'] = (log_diffs > 0).mean(1)
    return output

    
def compare_psycho(results, key, filename, show=False):
    quantiles = list(results[key]['psychophysics'].keys())
    metrics = list(results[key]['psychophysics']['quantile_0.0'].keys())
    plt.figure(figsize=(6 * len(quantiles), 6 * len(metrics)))
    for i, m in enumerate(metrics):
        for j, q in enumerate(quantiles):
            all_accs, all_keys, all_scores = [], [], []
            all_accs.append(
                results[key]['psychophysics'][q][m]['accuracy']
            )
            all_scores.append(results[key]['monosemanticity'])
            all_keys.append(key)
            # kmeans
            for num_dic in results[key]['kmeans_num_dic']:
                key2 = 'kmeans' + '_%s' % num_dic
                all_accs.append(
                    results[key]['psychophysics' + '_' + key2][q][m]['accuracy']
                )
                all_scores.append(
                    results[key]['monosemanticity' + '_' + key2]
                )
                all_keys.append(key + '_' + key2)
            plt.subplot(len(metrics), len(quantiles), i * len(quantiles) + j + 1)
            plt.boxplot(all_accs, showmeans=False)
            plt.xticks(np.arange(len(all_accs)) + 1, all_keys, rotation=45)
            plt.grid()
            plt.title(key + '\nmetric=%s, quantile=%s' % (m , q))
            plt.ylabel("Psychophysics Accuracy")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(filename)
        plt.clf()


def compare_probs(results, key, filename):
    all_top_accs, all_negents, all_keys, all_scores = [], [], [], []
    probabilities = results[key]['probabilities']
    all_top_accs.append(probabilities.max(1))
    all_negents.append(
        np.sum(probabilities * np.log(probabilities + 1e-9), 1)
    )
    all_scores.append(results[key]['monosemanticity'])
    all_keys.append(key)
    # kmeans
    for num_dic in results[key]['kmeans_num_dic']:
        key2 = 'kmeans' + '_%s' % num_dic
        probabilities = results[key]['probabilities' + '_' + key2]
        all_top_accs.append(probabilities.max(1))
        all_negents.append(
            np.sum(probabilities * np.log(probabilities + 1e-9), 1)
        )
        all_scores.append(
            results[key]['monosemanticity' + '_' + key2]
        )
        all_keys.append(key + '_' + key2)

    plt.figure(figsize=(4 * len(all_keys), 8))
    for i in range(len(all_keys)):
        plt.subplot(2, len(all_keys), i + 1)
        plt.scatter(all_top_accs[i], all_scores[i])
        plt.ylabel('Monosemanticity Index')
        plt.xlabel("Top Class Probability")
        plt.grid()
        plt.title(all_keys[i])
        plt.text(all_top_accs[i].min(), all_scores[i].max(), 'correlation=%.4f' % (
            corr(all_top_accs[i], all_scores[i])[0]
        ))
        plt.subplot(2, len(all_keys), i + 1 + len(all_keys))
        plt.scatter(all_negents[i], all_scores[i])
        plt.ylabel('Monosemanticity Index')
        plt.xlabel("NegEntropy")
        plt.grid()
        plt.title(all_keys[i])
        plt.text(all_negents[i].min(), all_scores[i].max(), 'correlation=%.4f' % (
            corr(all_negents[i], all_scores[i])[0]
        ))
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()