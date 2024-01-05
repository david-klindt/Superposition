import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from time import time
import pickle
import argparse

from PyTorch_CIFAR10.cifar10_models.resnet import resnet50

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import compute_kmeans
from psychophysics.psychophysics import run_psychophysics


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


def get_model(untrained):
    if untrained:
        print('Loading untrained model')
        return resnet50(pretrained=False).eval()
    else:
        print('Loading pretrained model')
        return resnet50(pretrained=True).eval()
    

def check_nans(data):
    if np.any(np.isnan(data)):
        print(np.isnan(data).mean(), 'nan in data')
        raise ValueError
    if np.any(np.isinf(data)):
        print(np.isinf(data).mean(), 'inf in data')
        raise ValueError


def get_activations(model, trainloader, testloader, device):
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    # Pick layer
    model.layer1.register_forward_hook(get_activation('layer1'))
    #model.layer2.register_forward_hook(get_activation('layer2'))
    #model.layer3.register_forward_hook(get_activation('layer3'))
    #model.layer4.register_forward_hook(get_activation('layer4'))
    _ = model(next(iter(trainloader))[0].to(device))
    results = {'keys': list(activation.keys())} 
    for key in results['keys']:
        results[key] = {
            'output_shape': activation[key].shape[1:],
            'activations_train': [],
            'activations_test': [],
        }
        print('Testing', key, 'output_shape', results[key]['output_shape'])

    print('Getting activations')
    results['inputs_train'], results['inputs_test'] = [], []
    results['labels_train'], results['labels_test'] = [], []
    with torch.no_grad():
        for batch in trainloader:
            results['inputs_train'].append(batch[0].detach().cpu().numpy().copy())
            results['labels_train'].append(batch[1].detach().cpu().numpy().copy())
            _ = model(batch[0].to(device))
            for key in activation:
                # take center pixel
                #ind_x = ind_y = results[key]['output_shape'][-1] // 2
                #results[key]['activations'].append(
                #    activation[key][:, :, ind_x, ind_y].detach().cpu().numpy().copy()
                #)
                # take spatial average
                # https://arxiv.org/pdf/2307.05471.pdf
                # "a unit is defined as one feature map of a convolutional layer, where the activation across
                # the feature map is aggregated by calculating the mean, just like for natural stimuli"
                results[key]['activations_train'].append(
                    np.mean(activation[key].detach().cpu().numpy().copy(), (2, 3))
                )
        for batch in testloader:
            results['inputs_test'].append(batch[0].detach().cpu().numpy().copy())
            results['labels_test'].append(batch[1].detach().cpu().numpy().copy())
            _ = model(batch[0].to(device))
            for key in activation:
                results[key]['activations_test'].append(
                    np.mean(activation[key].detach().cpu().numpy().copy(), (2, 3))
                )
    results['inputs_train'] = np.concatenate(results['inputs_train'], 0)
    results['inputs_test'] = np.concatenate(results['inputs_test'], 0)
    results['labels_train'] = np.concatenate(results['labels_train'], 0)
    results['labels_test'] = np.concatenate(results['labels_test'], 0)
    for key in activation:
        results[key]['activations_train'] = np.concatenate(results[key]['activations_train'], 0)
        check_nans(results[key]['activations_train'])
        results[key]['activations_test'] = np.concatenate(results[key]['activations_test'], 0)
        check_nans(results[key]['activations_test'])
        print(key, 'train', results[key]['activations_train'].shape, 
              'test', results[key]['activations_test'].shape)
    return results
    
    
def test_get_model():
    model_untrained = get_model(True)
    model_trained = get_model(False)
    assert isinstance(model_untrained, torch.nn.Module), "Untrained model is not a PyTorch Module"
    assert isinstance(model_trained, torch.nn.Module), "Trained model is not a PyTorch Module"


def main(args):
    save_file = 'results_untrained=%s_normalized=%s_psychophysics_trained=%s_distances=%s.pkl' % (
        args.untrained, args.normalized, args.psychophysics_trained, args.distances
    )
    save_file = os.path.join(args.log_dir, save_file)
    print("Saving to ", save_file)
    t0 = time()

    """
    # Get Data
    trainloader, testloader = get_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    model = get_model(args.untrained).to(device)
    results = get_activations(model, trainloader, testloader, device)
    with open(save_file, 'wb') as f:
        pickle.dump(results, f)
    #with open(save_file, 'rb') as f:
    #    results = pickle.load(f)
    print('Total run time', time() - t0)

    # Compute K Means
    for key in results['keys']:
        num_unit = results[key]['activations_train'].shape[1]
        results[key]['kmeans_num_dic'] = [num_unit]
        # for first layer check different kmeans_num_dic
        if False:#key == 'layer1':
            results[key]['kmeans_num_dic'] = list(2 ** np.arange(1, 14))
            
        for num_dic in results[key]['kmeans_num_dic']:
            print('Computing K-Means for', key, 'num_dic', num_dic)
            kmeans, activations_train, activations_test = compute_kmeans(
                data_train=results[key]['activations_train'],
                data_test=results[key]['activations_test'],
                num_dic=num_dic,
                normalized=args.normalized,
                align=False,
            )
            key2 = 'kmeans' + '_%s' % num_dic
            results[key][key2] = kmeans
            results[key]['activations_train_' + key2] = activations_train.copy()
            results[key]['activations_test_' + key2] = activations_test.copy()

    with open(save_file, 'wb') as f:
        pickle.dump(results, f)
    """
    with open(save_file, 'rb') as f:
        results = pickle.load(f)
    print('Total run time', time() - t0)

    # Run Psychophysics
    for key in results['keys']:
        print('Run Psychophysics for', key)
        key_train = 'train' if args.psychophysics_trained else 'test'
        results[key]['psychophysics'] = run_psychophysics(
            activations=results[key]['activations_%s' % key_train],
            inputs=results['inputs_%s' % key_train],
            labels=results['labels_%s' % key_train],
        )
        # kmeans
        for num_dic in results[key]['kmeans_num_dic']:
            key2 = 'kmeans' + '_%s' % num_dic
            print('Do Psychophysics for', key, key2)
            results[key]['psychophysics_' + key2] = run_psychophysics(
                activations=results[key]['activations_%s_' % key_train + key2], 
                inputs=results['inputs_%s' % key_train],
                labels=results['labels_%s' % key_train],
            )

    with open(save_file, 'wb') as f:
        pickle.dump(results, f)
    print('Total run time', time() - t0)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Superposition')
    parser.add_argument(
        '--log_dir', type=str, help='specify path'
    )
    parser.add_argument(
        '--untrained', action='store_true',
        default=False, help='analyze untrained model'
    )
    parser.add_argument(
        '--normalized', action='store_true',
        default=False, help='kmeans in cosine similarity'
    )
    parser.add_argument(
        '--psychophysics_trained', action='store_true',
        default=False, help='run psychophysics on training data'
    )
    parser.add_argument(
        '--distances', action='store_true',
        default=False, help='use kmeans distances instead of projections'
    )
    args = parser.parse_args()
    main(args)
