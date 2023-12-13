import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import pickle
from scipy.stats import pearsonr as corr
from tqdm import tqdm

from torchvision.models import AlexNet_Weights
from torchvision.io import read_image

import torch
import sys
sys.path.append('/home/cool_klindt/PerceptualSimilarity')
import lpips  # https://github.com/richzhang/PerceptualSimilarity/pull/124

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

imagenet_dir = '/home/datasets/ImageNet/ILSVRC/Data/CLS-LOC/train/'

df = pd.read_csv('human_responses/responses_main.csv')
print('unfiltered', len(df))
df = df[df['catch_trial'] == False]
df = df[df['is_demo'] == False]
print('filtered', len(df))

filename = 'image_data/imagenet_exemplars_file_map.json'

with open(filename) as json_file:
    mapping = json.load(json_file)

# Initialize the Weight Transforms for LPIPS
weights = AlexNet_Weights.DEFAULT  # for lpips
preprocess = weights.transforms()

def load_image(filename):
    img = read_image(filename)
    if img.shape[0] == 1:
        img = img.repeat((3, 1, 1))
    return preprocess(img)
    
# lpips
loss_fn_alex = lpips.LPIPS(net='alex').to(device)

def perceptual_similarity(a, b):
    _, ret = loss_fn_alex(a, b, normalize=False, retPerLayer=True)
    return -torch.stack(ret, dim=-1).detach().cpu().numpy()

MODELS = [
    'googlenet',
    'densenet_201',
    'wide_resnet50',
    #'googlenet_hard85',  # empty
    'clip-vit_b32',
    'in1k-vit_b32',
    'convnext_b',
    'resnet50',
    'resnet50-l2',
    'resnet50_hard85',
    'resnet50_hard95',
    'clip-resnet50',
    'clip-resnet50_hard85',
    'clip-resnet50_hard95'
]

for model in MODELS:
    df_model = df[df['model'] == model]
    print(model, len(df_model))
    #df_model = df_model[df_model['task_id'].str.contains('natural')]
    #print('natural', len(df_model))
    df_model = df_model.reset_index()
    print('Accuracy:', df_model['correct'].mean())

    # compute distances and similarities across all (also across lpips layers)
    correct_trial = []
    layers = []
    confidences = []
    logits_all = []
    task_ids = []
    for i in tqdm(range(len(df_model))):
        try:
            trial_dic = df_model[i:i + 1].to_dict()
            model = trial_dic['model'][i]
            layer = '%s/channel_%s' % (
                trial_dic['layer'][i], int(trial_dic['channel'][i])
            )
            batch = 'batch_%i' % trial_dic['batch'][i]
            task_id = trial_dic['task_id'][i]
            img_top, img_bottom = [], []
            for j in range(10):
                filename = os.path.join(
                    imagenet_dir, 
                    mapping[model][layer][batch]['max_%s.png' % j]
                )
                img = load_image(filename)
                img_top.append(img)
                filename = os.path.join(
                    imagenet_dir, 
                    mapping[model][layer][batch]['min_%s.png' % j]
                )
                img = load_image(filename)
                img_bottom.append(img)

            img_top = torch.stack(img_top, 0).to(device)
            img_bottom = torch.stack(img_bottom, 0).to(device)
            img_all = torch.cat([img_top, img_bottom], 0)
            logits = perceptual_similarity(img_all, img_all) # 20 x 20 x 5

            correct_trial.append(trial_dic['correct'][i])
            layers.append(layer)
            confidences.append(trial_dic['confidence'][i])
            task_ids.append(task_id)
            logits_all.append(logits.copy())

            if not i % 10:
                results = {
                    'correct_trial': np.array(correct_trial),
                    'layers': layers,
                    'confidences': np.array(confidences),
                    'logits_all': np.array(logits_all),
                    'task_ids': task_ids,
                }
                with open('results_%s.pickle' % model, 'wb') as handle:
                    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                #with open('results_%s.pickle' % model, 'rb') as handle:
                #    results = pickle.load(handle)

            
        except Exception as e:
            print(e)
