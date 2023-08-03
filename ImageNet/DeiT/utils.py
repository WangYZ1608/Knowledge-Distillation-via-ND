import os, shutil
import pdb
import torch, math
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def Save_Checkpoint(state, last, last_path, best, best_path, is_best):
    if os.path.exists(last):
        shutil.rmtree(last)
    last_path.mkdir(parents=True, exist_ok=True)
    torch.save(state, os.path.join(last_path, 'ckpt.pth'))

    if is_best:
        if os.path.exists(best):
            shutil.rmtree(best)
        best_path.mkdir(parents=True, exist_ok=True)
        torch.save(state, os.path.join(best_path, 'ckpt.pth'))


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def param_groups_weight_decay(model: nn.Module, weight_decay=1e-5, no_weight_decay_list=()):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # res.append(correct_k.mul_(100.0 / batch_size))
            res.append(correct_k)
        return res


class DirectNormLoss(nn.Module):
    """
    S_EMB: dict, {class}: emb, student embedding feature;
    T_EMB: dict, {class}: emb, teacher embedding center feature, avg of all samples;
    labels: list, [0, 1, ...], ground truth;
    nd_loss_factor: the weights of nd loss;
    """
    def __init__(self, num_class=100, nd_loss_factor=1.0):
        super(DirectNormLoss, self).__init__()
        self.num_class = num_class
        self.nd_loss_factor = nd_loss_factor
    
    def project_center(self, s_emb, t_emb, T_EMB, labels):
        assert s_emb.size() == t_emb.size()
        assert s_emb.shape[0] == len(labels)
        loss = 0.0
        for s, t, i in zip(s_emb, t_emb, labels):
            i = i.item()
            center = torch.tensor(T_EMB[str(i)]).cuda()
            e_c = center / center.norm(p=2)
            max_norm = max(s.norm(p=2), t.norm(p=2))
            loss += 1 - torch.dot(s, e_c) / max_norm
        return loss
    
    def project_center_v2(self, s_emb, t_emb, centers):
        assert s_emb.size() == t_emb.size() == centers.size()
        loss = 0.0
        for s, t, c in zip(s_emb, t_emb, centers):
            e_c = c / c.norm(p=2)
            max_norm = max(s.norm(p=2), t.norm(p=2))
            loss += 1 - torch.dot(s, e_c) / max_norm
        return loss
    
    def forward(self, s_emb, t_emb, T_EMB, labels):
        # nd_loss = self.project_center(s_emb=s_emb, t_emb=t_emb, T_EMB=T_EMB, labels=labels) * self.nd_loss_factor
        nd_loss = self.project_center_v2(s_emb=s_emb, t_emb=t_emb, centers=T_EMB) * self.nd_loss_factor
        
        return nd_loss / s_emb.shape[0]


class KDLoss(nn.Module):
    '''
	Distilling the Knowledge in a Neural Network
	https://arxiv.org/pdf/1503.02531.pdf
	'''
    def __init__(self, kd_loss_factor=1.0, T=1.0):
        super(KDLoss, self).__init__()
        self.T = T
        self.kd_loss_factor = kd_loss_factor

    def forward(self, s_out, t_out):
        # soft
        # kd_loss = F.kl_div(F.log_softmax(s_out / self.T, dim=1), 
        #                    F.softmax(t_out / self.T, dim=1), 
        #                    reduction='batchmean',
        #                    ) * self.T * self.T
        
        # hard
        kd_loss = F.cross_entropy(s_out, t_out.argmax(dim=1))
        return kd_loss * self.kd_loss_factor


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count