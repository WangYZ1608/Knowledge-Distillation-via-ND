"""
Visualized the embedding feature of the pre-train model on the training set.
"""
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict

import Models
from Dataset import CIFAR

import numpy as np
import argparse
import json

def emb_fea(model, data, args):
    # model to evaluate mode
    model.eval()
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    EMB = {}

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.cuda(), labels.cuda()

            # compute output
            emb_fea, logits = model(images, embed=True)

            for emb, i in zip(emb_fea, labels):
                i = i.item()
                assert len(emb) == args.emb_size
                if str(i) in EMB:
                    for j in range(len(emb)):
                        EMB[str(i)][j].append(round(emb[j].item(), 4))
                else:
                    EMB[str(i)] = [[] for _ in range(len(emb))]
                    for j in range(len(emb)):
                        EMB[str(i)][j].append(round(emb[j].item(), 4))

    
    for key, value in EMB.items():
        for i in range(args.emb_size):
            EMB[key][i] = round(np.array(EMB[key][i]).mean(), 4)
    
    return EMB


if __name__ == "__main__":
    model_names = sorted(name for name in Models.__dict__ 
                         if name.islower() and not name.startswith("__") 
                         and callable(Models.__dict__[name]))

    parser = argparse.ArgumentParser(description='Visualized the embedding feature of the model on the train set.')
    parser.add_argument("--model_name", type=str, default="resnet56_cifar", choices=model_names, help="model architecture")
    parser.add_argument("--model_weights", type=str, default="", help="model weights path")
    parser.add_argument("--emb_size", type=int, default=64, help="emb fea size")
    parser.add_argument("--dataset", type=str, default='cifar10')
    parser.add_argument("--batch_size", type=int, default=64, help="total batch size")
    parser.add_argument('--workers', default=16, type=int, help='number of data loading workers')
    args = parser.parse_args()
    
    # dataset
    if args.dataset in ['cifar10', 'cifar100']:
        train_set, test_set, num_class = CIFAR(name=args.dataset)
    else:
        print("No Dataset!!!")
    
    model = Models.__dict__[args.model_name](num_class=num_class)
        
    if args.model_weights:
        print('Visualized the embedding feature of the {} model on the train set'.format(args.model_name))
        
        model_ckpt = torch.load(args.model_weights)['model_state_dict']
        new_state_dict = OrderedDict()
        for k, v in model_ckpt.items():
            name = k[7:]   # remove 'module.'
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

        for param in model.parameters():
            param.requires_grad = False

    else:
        print('No load Pre-trained weights!')
    
    model = model.cuda()

    emb = emb_fea(model=model, data=train_set, args=args)
    emb_json = json.dumps(emb, indent=4)
    with open("./run/{}_embedding_fea/{}.json".format(args.dataset, args.model_name), 'w', encoding='utf-8') as f:
        f.write(emb_json)
    f.close()