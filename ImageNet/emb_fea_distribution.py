"""
Visualized the embedding feature of the model on the training set.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
from torch.utils.data import DataLoader

import Models
from Dataset import ImageNet

import numpy as np
import argparse
import json
import pdb

def emb_fea(model, data, args):
    # model to evaluate mode
    model.eval()
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    EMB = {}

    # pdb.set_trace()
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

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training with Distributed Data Parallel.')
    parser.add_argument("--model_name", type=str, default="resnet50", choices=model_names, help="model architecture")
    parser.add_argument("--model_weights", type=str, default="", help="model weights path")
    parser.add_argument("--emb_size", type=int, default=2048, help="emb fea size")
    parser.add_argument("--dataset", type=str, default='ImageNet')
    parser.add_argument("--batch_size", type=int, default=256, help="total batch size")
    parser.add_argument('--workers', default=32, type=int, help='number of data loading workers')
    args = parser.parse_args()
    
    # dataset
    train_data, test_data, num_class = ImageNet()
    model = Models.__dict__[args.model_name]()

    if args.model_weights:
        print('Using torchvision pretrained model for teacher {}'.format(args.model_name))
        model_ckpt = torch.load(args.model_weights)
        model.load_state_dict(model_ckpt)

        for param in model.parameters():
            param.requires_grad = False
    
    model = model.cuda()

    emb = emb_fea(model=model, data=train_data, args=args)
    emb_json = json.dumps(emb, indent=4)
    with open("./ckpt/teacher/resnet152/center_emb_train.json", 'w', encoding='utf-8') as f:
        f.write(emb_json)
    f.close()