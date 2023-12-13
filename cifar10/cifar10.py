import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from time import time
import os
import pickle
import argparse

from PyTorch_CIFAR10.cifar10_models.resnet import resnet50


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
    

def get_activations(model, trainloader, device):
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    # Pick layer
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
                #ind_x = ind_y = results[key]['output_shape'][-1] // 2
                #results[key]['activations'].append(
                #    activation[key][:, :, ind_x, ind_y].detach().cpu().numpy().copy()
                #)
                # take spatial average
                # https://arxiv.org/pdf/2307.05471.pdf
                # "a unit is defined as one feature map of a convolutional layer, where the activation across
                # the feature map is aggregated by calculating the mean, just like for natural stimuli"
                results[key]['activations'].append(
                    np.mean(activation[key].detach().cpu().numpy().copy(), (2, 3))
                )
    results['inputs'] = np.concatenate(results['inputs'], 0)
    results['labels'] = np.concatenate(results['labels'], 0)
    for key in activation:
        results[key]['activations'] = np.concatenate(results[key]['activations'], 0)
        if np.any(np.isnan(results[key]['activations'])):
            print(np.isnan(results[key]['activations']).mean(), 'nan in data')
            raise ValueError
        if np.any(np.isinf(results[key]['activations'])):
            print(np.isinf(results[key]['activations']).mean(), 'inf in data')
            raise ValueError
    return results
    
    
def test_get_model():
    model_untrained = get_model(True)
    model_trained = get_model(False)
    assert isinstance(model_untrained, torch.nn.Module), "Untrained model is not a PyTorch Module"
    assert isinstance(model_trained, torch.nn.Module), "Trained model is not a PyTorch Module"


def main(args):
    t0 = time()
    trainloader, _ = get_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    model = get_model(args.untrained).to(device)
    results = get_activations(model, trainloader)
    with open(os.path.join(args.log_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    #with open(os.path.join(args.log_dir, 'results.pkl'), 'rb') as f:
    #    results = pickle.load(f)
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
    args = parser.parse_args()
    main(args)
