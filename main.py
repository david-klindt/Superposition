import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import cm
import json
import argparse
import pickle 

from scipy.stats import mode
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
from scipy.stats import pearsonr as corr

import sys
import os
# https://github.com/huyvnphan/PyTorch_CIFAR10
sys.path.append('/home/cool_klindt/Superposition/PyTorch_CIFAR10/')
from cifar10_models.resnet import resnet50

#import lpips
# use custom version with batch support
# https://github.com/david-klindt/PerceptualSimilarity/tree/batched
sys.path.append('/home/cool_klindt/PerceptualSimilarity')
import lpips


global_seed = 20230831
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


def imshow(img):
    x = img - np.min(img)
    x = x / np.max(x)
    plt.imshow(np.transpose(x, (1, 2, 0)))
    plt.axis('off')


def get_data(batch_size=64, data_dir='/home/cool_klindt/data'):
    # https://github.com/huyvnphan/PyTorch_CIFAR10
    print('Getting data')
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2471, 0.2435, 0.2616]
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    return trainloader, testloader


def get_model():
    print('Loading pretrained model')
    return resnet50(pretrained=True).eval()


def get_svd(data, filename):
    data = data.copy()
    data -= np.mean(data)
    data /= np.std(data)
    u, s, v = np.linalg.svd(data, full_matrices=False)
    var_exp = s ** 2
    var_exp /= np.sum(var_exp)
    plt.figure(figsize=(6, 4))
    plt.plot(var_exp, '.-')
    plt.ylim(-.01, var_exp.max() * 1.1)
    plt.xlabel('SVD component')
    plt.ylabel('Variance Explained')
    #plt.semilogx()
    plt.title('Spectrum\n0.95 Var Exp at %s' % (
        np.where(np.cumsum(var_exp) > .95)[0][0]
    ))
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()
    return u, s, v


def plot_meis(activations, scores, inputs, filename, 
              num_show=10, show_all=False, plot=False):
    # show top5 and bottom5 scores
    # show by decreasing activations
    if show_all:    
        num_unit = activations.shape[1]
    else:
        num_unit = num_show
    if len(scores) > 0:
        ind_sort = np.argsort(scores)
        ind_units = np.concatenate([
            ind_sort[:num_unit // 2], ind_sort[-num_unit // 2 - 1:]
        ])
    elif scores == None:
        scores = activations.mean(0)
        ind_units = np.arange(num_unit)
    plt.figure(figsize=((num_show + 1) * 1.1, num_unit * 1.1))
    for i in range(num_unit):
        ind_unit = ind_units[i]
        y = activations[:, ind_unit].copy()
        plt.subplot(num_unit, num_show + 1, 1 + i * (num_show + 1))
        plt.axis('off')
        plt.text(0, 0, 'Unit: %s,\nScore: %.4f,\nMean: %.4f,\nStd: %.4f' % (
            ind_unit, scores[ind_unit], y.mean(), y.std()
        ))
        ind_sort = np.argsort(y)[::-1]
        for j in range(num_show):
            plt.subplot(num_unit, num_show + 1, j + 2 + i * (num_show + 1))
            imshow(inputs[ind_sort[j]])
            plt.title('%.4f' % y[ind_sort[j]])
    plt.tight_layout()
    if plot:
        plt.show()
    else:
        plt.savefig(filename)
        plt.clf()


loss_fn_alex = lpips.LPIPS(net='alex').to(device) # best forward scores
#loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization
def lpips_metric(a, b):
  img0 = torch.tensor(a, dtype=torch.float32, device=device)
  img1 = torch.tensor(b, dtype=torch.float32, device=device)
  with torch.no_grad():
      output = loss_fn_alex(img0, img1, normalize=True)
  return output.squeeze().detach().cpu().numpy()


def get_mei_data(activations, inputs, num_train=5, num_test=5):
    "Get most activating images as train and test for lpips"
    num_unit = activations.shape[1]
    data_train = np.zeros((num_unit, num_train, 3, 32, 32))
    data_test = np.zeros((num_unit, num_test, 3, 32, 32))
    for i in range(num_unit):
        y = activations[:, i].copy()
        ind_sort = np.argsort(y)[::-1]
        data_train[i] = inputs[ind_sort[:num_train]].copy()
        data_test[i] = inputs[ind_sort[num_train:num_train + num_test]].copy()
    return data_train, data_test


def get_label_data(activations, labels, num=20):
    "Get top labels for each unit"
    num_unit = activations.shape[1]
    classes = np.zeros((num_unit, num))
    for i in range(num_unit):
        y = activations[:, i].copy()
        ind_sort = np.argsort(y)[::-1]
        classes[i] = labels[ind_sort[:num]].copy()
    probabilities = np.zeros((num_unit, 10))
    for i in range(10):
        ind = classes == i
        probabilities[:, i] = ind.sum(1)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    return probabilities


def get_monosemanticity(data_train):
    # Compute all pairwise distances and return average, lower is better
    num_unit = data_train.shape[0]
    #num_train = data_train.shape[1]
    scores = np.zeros(num_unit)
    for i in range(num_unit):
        similarities = lpips_metric(data_train[i], data_train[i])
        scores[i] = np.mean(similarities)
    return scores


def compute_kmeans(data, num_dic, seed=global_seed):
    if num_dic == data.shape[1]:
        print('try to align with neurons, init as eye')
        init = np.eye(num_dic)
        n_init = 1
    else:
        init = 'k-means++'
        n_init = 50 #"auto",
    kmeans = MiniBatchKMeans(
        n_clusters=num_dic,
        random_state=seed,
        n_init=n_init,
        verbose=False,
        init=init,
        #batch_size=6,
        #max_iter=10,
    ).fit(data)
    distances = kmeans.transform(data)
    return kmeans, distances

    
def compare_mono(results, key, filename, all_units=True):
    all_scores, all_keys = [], []
    if all_units:
        all_scores.append(-results[key]['monosemanticity'])
        all_keys.append(key)
    else: # split by mono scores
        activations_normed = results[key]['activations_normed'].copy()
        num_unit = activations_normed.shape[1]
        monosemanticity = results[key]['monosemanticity'].copy()
        ind_sort = np.argsort(monosemanticity)
        ind_top = ind_sort[:num_unit // 2]
        ind_bottom = ind_sort[num_unit // 2:]
        all_scores.append(-monosemanticity[ind_top])
        all_keys.append(key + '_top')
        all_scores.append(-monosemanticity[ind_bottom])
        all_keys.append(key + '_bottom')

    # kmeans
    for num_dic in results[key]['kmeans_num_dic']:
        if all_units:
            key2 = 'kmeans' + '_%s' % num_dic
            all_scores.append(
                -results[key]['monosemanticity' + '_' + key2]
            )
            all_keys.append(key + '_' + key2)
        else:
            for subset in ['top', 'bottom']:
                key2 = 'kmeans' + '_%s_%s' % (num_dic, subset)
                all_scores.append(
                    -results[key]['monosemanticity' + '_' + key2]
                )
                all_keys.append(key + '_' + key2)

    plt.figure(figsize=(6, 6))
    plt.boxplot(all_scores, showmeans=False)
    plt.xticks(
        np.arange(len(all_scores)) + 1,
        all_keys, rotation=90
    )
    plt.grid()
    plt.title(key)
    plt.ylabel("Monosemanticity Index")
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()


def psychophyscis_lpips(data_train, data_test, num_pair=10, seed=global_seed):
    num_unit = data_train.shape[0]
    assert not num_unit % 2
    num_train = data_train.shape[1]
    num_test = data_test.shape[1]
    target = np.zeros(num_test * 2)
    target[num_test:] = 1
    accuracies = np.zeros((num_unit, num_pair))
    for i in range(num_pair):
        np.random.seed(seed + i)
        ind_units = np.random.choice(num_unit, num_unit, replace=False)
        for j in range(num_unit // 2):
            ind_a = ind_units[j * 2]
            ind_b = ind_units[j * 2 + 1]
            data_train_pair = np.concatenate([data_train[ind_a], data_train[ind_b]], 0)
            data_test_pair = np.concatenate([data_test[ind_a], data_test[ind_b]], 0)
            similarities = lpips_metric(data_train_pair, data_test_pair)
            logits = np.stack([
                similarities[:num_train].min(0), similarities[num_train:].min(0)
            ], 1)
            accuracy = np.mean(logits.argmin(1) == target)
            accuracies[ind_a, i] = accuracies[ind_b, i] = accuracy
    return accuracies.mean(1)

    
def compare_psycho(results, key, filename):
    all_accs, all_keys, all_scores = [], [], []
    all_accs.append(
        results[key]['psychophysics_acc']
    )
    all_scores.append(-results[key]['monosemanticity'])
    all_keys.append(key)
    # kmeans
    for num_dic in results[key]['kmeans_num_dic']:
        key2 = 'kmeans' + '_%s' % num_dic
        all_accs.append(
            results[key]['psychophysics_acc' + '_' + key2]
        )
        all_scores.append(
            -results[key]['monosemanticity' + '_' + key2]
        )
        all_keys.append(key + '_' + key2)

    plt.figure(figsize=(6 * len(all_keys), 6))
    plt.subplot(1, len(all_keys) + 1, 1)
    plt.boxplot(all_accs, showmeans=False)
    plt.xticks(np.arange(len(all_accs)) + 1, all_keys, rotation=90)
    plt.grid()
    plt.title(key)
    plt.ylabel("Psychophysics Accuracy")
    for i in range(len(all_keys)):
        plt.subplot(1, len(all_keys) + 1, 2 + i)
        plt.scatter(all_accs[i], all_scores[i])
        plt.ylabel('Monosemanticity Index')
        plt.xlabel("Psychophysics Accuracy")
        plt.grid()
        plt.title(all_keys[i])
        plt.text(all_accs[i].min(), all_scores[i].max(), 'correlation=%.4f' % (
            corr(all_accs[i], all_scores[i])[0]
        ))
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()


def compare_probs(results, key, filename):
    all_top_accs, all_negents, all_keys, all_scores = [], [], [], []
    probabilities = results[key]['probabilities']
    all_top_accs.append(probabilities.max(1))
    all_negents.append(
        np.sum(probabilities * np.log(probabilities + 1e-9), 1)
    )
    all_scores.append(-results[key]['monosemanticity'])
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
            -results[key]['monosemanticity' + '_' + key2]
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


def main(args):
    #"""
    trainloader, testloader = get_data()
    model = get_model().to(device)

    # Pick layer
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    model.layer1.register_forward_hook(get_activation('layer1'))
    model.layer2.register_forward_hook(get_activation('layer2'))
    model.layer3.register_forward_hook(get_activation('layer3'))
    model.layer4.register_forward_hook(get_activation('layer4'))
    _ = model(next(iter(trainloader))[0].to(device))
    results = {'keys': list(activation.keys())} 
    for key in results['keys']:
        results[key] = {
            'output_shape': activation[key].shape[1:],
            'activations': [],
        }
        print('Testing', key, 'output_shape', results[key]['output_shape'])

    print('Getting activations')
    results['inputs'] = []
    results['labels'] = []
    with torch.no_grad():
        for batch in trainloader:
            results['inputs'].append(batch[0].detach().cpu().numpy().copy())
            results['labels'].append(batch[1].detach().cpu().numpy().copy())
            _ = model(batch[0].to(device))
            for key in activation:
                # take center pixel
                ind_x = ind_y = results[key]['output_shape'][-1] // 2
                results[key]['activations'].append(
                    activation[key][:, :, ind_x, ind_y].detach().cpu().numpy().copy()
                )
    results['inputs'] = np.concatenate(results['inputs'], 0)
    results['labels'] = np.concatenate(results['labels'], 0)
    for key in activation:
        results[key]['activations'] = np.concatenate(results[key]['activations'], 0)
        # normalize for k-Means
        length = np.sqrt((results[key]['activations'] ** 2).sum(
            axis=1))[:, None].clip(min=1e-6)
        results[key]['activations_normed'] = results[key]['activations'] / length


    with open(os.path.join(args.log_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    #with open(os.path.join(args.log_dir, 'results.pkl'), 'rb') as f:
    #    results = pickle.load(f)

    # SVD
    for key in results['keys']:
        u, s, v = get_svd(
            results[key]['activations'],
            os.path.join(args.log_dir, key + '_svd.png')
        )
        # needed later?
        #results[key]['activations_u'] = u.copy()
        #results[key]['activations_s'] = s.copy()
        #results[key]['activations_v'] = v.copy()


    # Compute K Means
    for key in results['keys']:
        num_unit = results[key]['activations'].shape[1]
        # for first layer check different kmeans_num_dic
        if key == 'layer1':
            results[key]['kmeans_num_dic'] = [num_unit // 2, num_unit, num_unit * 2]
        else:
            results[key]['kmeans_num_dic'] = [num_unit]

        # to limit later
        #for key in results['keys']:
        #    results[key]['kmeans_num_dic'] = [results[key]['kmeans_num_dic'][1]]


        for num_dic in results[key]['kmeans_num_dic']:
            print('Computing K-Means for', key, 'num_dic', num_dic)
            kmeans, distances = compute_kmeans(
                results[key]['activations_normed'], num_dic
            )
            key2 = 'kmeans' + '_%s' % num_dic
            results[key][key2] = kmeans
            results[key][key2 + 'dist'] = distances.copy()

    with open(os.path.join(args.log_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    #with open(os.path.join(args.log_dir, 'results.pkl'), 'rb') as f:
    #    results = pickle.load(f)
    
    # Plot MEIs
    for key in results['keys']:
        print('Plotting MEIs for', key)
        plot_meis(
            results[key]['activations'], 
            results[key]['activations'].mean(0),
            results['inputs'],
            os.path.join(args.log_dir, key + '_meis' + '.png')
        )
        # kmeans
        for num_dic in results[key]['kmeans_num_dic']:
            key2 = 'kmeans' + '_%s' % num_dic
            plot_meis(
                -results[key][key2 + 'dist'],
                -results[key][key2 + 'dist'].mean(0),
                results['inputs'],
                os.path.join(args.log_dir, key + '_' + key2 + '_meis' + '.png')
            )

    with open(os.path.join(args.log_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    #with open(os.path.join(args.log_dir, 'results.pkl'), 'rb') as f:
    #    results = pickle.load(f)

    # Get Psychophysics data
    for key in results['keys']:
        print('Get Psychophysics data for', key)
        data_train, data_test = get_mei_data(
            results[key]['activations'], results['inputs']
        )
        results[key]['data_train'] = data_train.copy()
        results[key]['data_test'] = data_test.copy()
        # kmeans
        for num_dic in results[key]['kmeans_num_dic']:
            key2 = 'kmeans' + '_%s' % num_dic
            data_train, data_test = get_mei_data(
                -results[key][key2 + 'dist'], results['inputs']
            )
            results[key]['data_train' + '_' + key2] = data_train.copy()
            results[key]['data_test' + '_' + key2] = data_test.copy()
                
    with open(os.path.join(args.log_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    #with open(os.path.join(args.log_dir, 'results.pkl'), 'rb') as f:
    #    results = pickle.load(f)

    # Get prevalence of most common class or class entrop as semantics index
    for key in results['keys']:
        print('Get Class data for', key)
        probabilities = get_label_data(
            results[key]['activations'], results['labels']
        )
        results[key]['probabilities'] = probabilities.copy()
        # kmeans
        for num_dic in results[key]['kmeans_num_dic']:
            key2 = 'kmeans' + '_%s' % num_dic
            probabilities = get_label_data(
                -results[key][key2 + 'dist'], results['labels']
            )
            results[key]['probabilities' + '_' + key2] = probabilities.copy()
                
    with open(os.path.join(args.log_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    #with open(os.path.join(args.log_dir, 'results.pkl'), 'rb') as f:
    #    results = pickle.load(f)

    # Get Monosemanticity scores
    for key in results['keys']:
        print('Get Monosemanticity scores for', key)
        scores = get_monosemanticity(results[key]['data_train'])
        results[key]['monosemanticity'] = scores.copy()
        plot_meis(
            results[key]['activations'], 
            scores,
            results['inputs'],
            os.path.join(args.log_dir, key + '_monoscored_meis' + '.png')
        )
        # kmeans
        for num_dic in results[key]['kmeans_num_dic']:
            key2 = 'kmeans' + '_%s' % num_dic
            scores = get_monosemanticity(results[key]['data_train' + '_' + key2])
            results[key]['monosemanticity' + '_' + key2] = scores.copy()
            plot_meis(
                -results[key][key2 + 'dist'],
                scores,
                results['inputs'],
                os.path.join(
                    args.log_dir, 
                    key + '_monoscored_meis' + '_' + key2 + '.png'
                )
            )
        compare_mono(
            results, key, 
            os.path.join(args.log_dir, key + '_monosemanticity' + '.png')
        )
        compare_probs(
            results, key,
            os.path.join(args.log_dir, key + '_probs' + '.png')
        )

    with open(os.path.join(args.log_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    #with open(os.path.join(args.log_dir, 'results.pkl'), 'rb') as f:
    #    results = pickle.load(f)


    ### Split Mono vs Polysemantic neurons and cluster again
    # Compute K Means
    for key in results['keys']:
        activations_normed = results[key]['activations_normed'].copy()
        num_unit = activations_normed.shape[1]
        monosemanticity = results[key]['monosemanticity'].copy()
        ind_sort = np.argsort(monosemanticity)
        ind_top = ind_sort[:num_unit // 2]
        ind_bottom = ind_sort[num_unit // 2:]
        for num_dic in results[key]['kmeans_num_dic']:
            for subset in ['top', 'bottom']:
                print('Computing K-Means for', key, 'num_dic', num_dic, 'subset', subset)
                if subset == 'top':
                    data = activations_normed[:, ind_top]
                else:
                    data = activations_normed[:, ind_bottom]
                # kmeans
                kmeans, distances = compute_kmeans(data, num_dic // 2)
                # todo: measure sparsity in top vs bottom (top more sparse!??)
                key2 = 'kmeans' + '_%s_%s' % (num_dic, subset)
                results[key][key2] = kmeans
                results[key][key2 + 'dist'] = distances.copy()
                # get psycho data
                data_train, data_test = get_mei_data(
                    -results[key][key2 + 'dist'], results['inputs']
                )
                results[key]['data_train' + '_' + key2] = data_train.copy()
                results[key]['data_test' + '_' + key2] = data_test.copy()
                # mono scores
                scores = get_monosemanticity(results[key]['data_train' + '_' + key2])
                results[key]['monosemanticity' + '_' + key2] = scores.copy()
        compare_mono(
            results, key, 
            os.path.join(args.log_dir, key + '_monosemanticity_split' + '.png'),
            all_units=False
        )

    with open(os.path.join(args.log_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    #with open(os.path.join(args.log_dir, 'results.pkl'), 'rb') as f:
    #    results = pickle.load(f)

    # Do Psychophysics
    for key in results['keys']:
        print('Do Psychophysics for', key)
        results[key]['psychophysics_acc'] = psychophyscis_lpips(
            results[key]['data_train'], results[key]['data_test']
        )
        # kmeans
        for num_dic in results[key]['kmeans_num_dic']:
            key2 = 'kmeans' + '_%s' % num_dic
            results[key]['psychophysics_acc' + '_' + key2] = psychophyscis_lpips(
                results[key]['data_train' + '_' + key2], 
                results[key]['data_test' + '_' + key2]
            )
        compare_psycho(
            results, key, 
            os.path.join(args.log_dir, key + '_psychophysics' + '.png')
        )

    with open(os.path.join(args.log_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    #with open(os.path.join(args.log_dir, 'results.pkl'), 'rb') as f:
    #    results = pickle.load(f)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Superposition')
    parser.add_argument(
        '--log_dir', type=str, help='specify path'
    )
    args = parser.parse_args()
    main(args)




