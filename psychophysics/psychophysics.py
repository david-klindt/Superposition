import numpy as np
from typing import Optional, List

import os
import sys
from tqdm import tqdm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from psychophysics.metrics import get_metric


def get_center_ind(activations, quantile):
    """Get indices of all activations in center quantile, only use distribution center between quantiles."""
    return np.logical_and(
        activations >= np.quantile(activations, quantile), 
        activations <= np.quantile(activations, 1 - quantile))


def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def run_psychophysics(
        inputs: np.ndarray,
        activations: np.ndarray,
        labels: Optional[np.ndarray] = None):
    """
    Conducts a psychophysics experiment on all units.
    See https://arxiv.org/pdf/2307.05471.pdf Appendix A.2

    Parameters:
    inputs (np.ndarray): The input data for the experiment.
    activations (np.ndarray): The activations of the units.
    metric_name (str): The name of the metric to use.
    metric (callable, optional): The metric function to use. If None, a metric function is obtained using get_metric.

    Returns:
    dict: A dict containing the logits and score of the experiment.
    """
    result = {}
    num_unit = activations.shape[1]

    print('Computing all image metrics')
    metric = get_metric('image')
    for i in tqdm(range(num_unit)):
        output = run_single_unit_psychophysics(
            inputs=inputs, activations=activations[:, i], 
            metric_name='image', metric=metric)
        for key in output:
            if i == 0:
                result[key] = []
            result[key].append(output[key])
            if i == num_unit - 1:
                result[key] = np.stack(result[key], 0)

    if type(labels) != type(None):
        metric = get_metric('label')
        for label in range(labels.shape[1]):
            print('Computing metric for label', label)
            for i in tqdm(range(num_unit)):
                output = run_single_unit_psychophysics(
                    inputs=labels[:, label], activations=activations[:, i], 
                    metric_name='label', metric=metric)
                for key in output:
                    if i == 0:
                        result['label_%s_' % label + key] = []
                    result['label_%s_' % label + key].append(output[key])
                    if i == num_unit - 1:
                        result['label_%s_' % label + key] = np.stack(result['label_%s_' % label + key], 0)
    return result


def extract_logits(similarities, zscore, pool_fun):
    num_references = similarities.shape[0] // 2
    if zscore:
        similarities -= np.mean(similarities)
        std = np.std(similarities)
        if np.isnan(std) or std == 0.0:
            std = 1.0
        similarities /= std
    logits = np.zeros((2, 2))
    # compute logit for top
    logits[0, 0] = pool_fun(similarities[:num_references, 0])  # evidence_from_top
    logits[0, 1] = pool_fun(similarities[num_references:, 1])  # evidence_from_bottom
    # compute logit bottom
    logits[1, 0] = pool_fun(similarities[:num_references, 1])  # evidence_from_top
    logits[1, 1] = pool_fun(similarities[num_references:, 0])  # evidence_from_bottom
    return logits


def compute_score(logits):
    actual_logits = logits.sum(-1)  # summing evidences: Q x T x 2
    score = np.mean(softmax(actual_logits)[:, :, 0], axis=1)  # probablity of correct answer: Q
    return score


def run_single_unit_psychophysics(
        inputs: np.ndarray,
        activations: np.ndarray,
        metric_name: str,
        metric: Optional[callable] = None,
        quantiles: Optional[List[float]] = [0.0, 0.25], 
        num_trials: int = 20, 
        num_references: int = 9,
        pool_fun: callable = np.mean,  # min/max would not make sense with label
        zscore: bool = True,
        seed: int = 42):
    """
    Conducts a psychophysics experiment on a single unit.
    See https://arxiv.org/pdf/2307.05471.pdf Appendix A.2

    Parameters:
    inputs (np.ndarray): The input data for the experiment.
    activations (np.ndarray): The activations of the units.
    metric_name (str): The name of the metric to use.
    metric (callable, optional): The metric function to use. If None, a metric function is obtained using get_metric.
    quantiles (List[float], optional): The quantiles to consider. Defaults to [0.0, 0.25].
    num_trials (int, optional): The number of trials to conduct. Defaults to 20.
    num_references (int, optional): The number of reference units to consider. Defaults to 9.
    pool_function (callable, optional): The function to use for pooling. Defaults to np.mean.
    zscore (bool, optional): Whether to z-score (subtract mean and divide by std.dev.) similarity measures.
    seed (int, optional): The seed for the random number generator. Defaults to 42.

    Returns:
    dict: A tuple containing the logits and score of the experiment.
    """
    if len(activations.shape) != 1:
        raise ValueError("Activations must be a vector, but have shape %s." % list(activations.shape))
    if inputs.shape[0] != activations.shape[0]:
        raise ValueError("Input and activations must have the same first dimension.")
    if len(inputs.shape) == 4:  # images
        if not metric_name in ['l2', 'color', 'lpips', 'image']:
            raise ValueError("Invalid metric for image data.")
    elif len(inputs.shape) == 1:  # labels
        if metric_name != 'label':
            raise ValueError("Invalid metric for label data.")
    else:
        raise ValueError("Inputs of shape (%s) unclear (NCHW for images, NC for labels)." % list(inputs.shape))
    if metric is None:
        metric = get_metric(metric_name)
    if not all(q1 <= q2 for q1, q2 in zip(quantiles, quantiles[1:])):
        raise ValueError("Quantiles must be in ascending order.")
    if quantiles[0] < 0.0:
        raise ValueError("First quantile must be >= 0.0.")
    if quantiles[-1] >= 0.5:
        raise ValueError("Last quantile must be < 0.5.")
    # make sure there is at least twice as many left as we need for highest quantile.
    if np.sum(get_center_ind(activations, quantiles[-1])) <= num_trials * (num_references + 1) * 2 * 2:
        raise ValueError("Not enough data for the specified number of quantiles, trials and references.")
        
    np.random.seed(seed)
    output = dict()
    if metric_name in ['lpips', 'image']:
        output['logits_lpips'] = np.zeros((len(quantiles), num_trials, 2, 2))  # Q x T x 2 x 2
        for layer in range(5):
            output['logits_lpips_layer_%s' % layer] = np.zeros((len(quantiles), num_trials, 2, 2))  # Q x T x 2 x 2
        if metric_name == 'image':
            output['logits_l2'] = np.zeros((len(quantiles), num_trials, 2, 2))  # Q x T x 2 x 2
            output['logits_color'] = np.zeros((len(quantiles), num_trials, 2, 2))  # Q x T x 2 x 2
    else:
        output['logits'] = np.zeros((len(quantiles), num_trials, 2, 2))  # Q x T x 2 x 2

    for quantile_index, quantile in enumerate(quantiles):
        ind = get_center_ind(activations, quantile)
        y = activations[ind].copy()
        x = inputs[ind].copy()
        
        ind_sort_bottom = np.argsort(y)
        ind_sort_top = ind_sort_bottom[::-1]
        ind_query_top = np.random.choice(num_trials, num_trials, replace=False)
        ind_query_bottom = np.random.choice(num_trials, num_trials, replace=False)
        query_top = x[ind_sort_top[:num_trials][ind_query_top]]
        query_bottom = x[ind_sort_bottom[:num_trials][ind_query_bottom]]

        # permute only query, enough randomness, decreasing difficulty not a problem here
        ind_reference = num_trials + np.arange(num_references) * num_trials

        for trial_index in range(num_trials):
            reference_top = x[ind_sort_top[ind_reference + trial_index]]
            reference_bottom = x[ind_sort_bottom[ind_reference + trial_index]]
            
            # Run all together (twice as fast):
            reference = np.concatenate([reference_top, reference_bottom], 0)
            query = np.concatenate([query_top[trial_index:trial_index + 1], 
                                    query_bottom[trial_index:trial_index + 1]])
            similarities = metric(reference, query)
            if metric_name in ['lpips', 'image']:
                output['logits_lpips'][quantile_index, trial_index] = extract_logits(
                    similarities[:, :, 2:].mean(2), zscore, pool_fun)
                for layer in range(5):
                    output['logits_lpips_layer_%s' % layer][quantile_index, trial_index] = extract_logits(
                        similarities[:, :, 2 + layer], zscore, pool_fun)
                if metric_name == 'image':
                    output['logits_l2'][quantile_index, trial_index] = extract_logits(
                        similarities[:, :, 0], zscore, pool_fun)
                    output['logits_color'][quantile_index, trial_index] = extract_logits(
                        similarities[:, :, 1], zscore, pool_fun)
            else:
                output['logits'][quantile_index, trial_index] = extract_logits(
                    similarities, zscore, pool_fun)

    # compute score
    keys = list(output.keys())
    for key in keys:
        if key.startswith('logits'):
            output['score' + key[6:]] = compute_score(output[key])
        
    return output


if __name__ == "__main__":
    num_data = 10000
    inputs = np.random.normal(0, 1, (num_data, 3, 32, 32))
    activations = np.random.normal(0, 1, (num_data,))
    output = run_single_unit_psychophysics(
        inputs=inputs, activations=activations, metric_name='lpips', zscore=False)
    logits = output['logits']
    score = output['score']
    print('Shape of logits [should be (2, 20, 2, 2)]: ', logits.shape)
    print('Scores per quantile [should be close to 0.5]: ', score)
