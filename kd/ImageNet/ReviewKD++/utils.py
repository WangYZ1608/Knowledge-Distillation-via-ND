import os, shutil
import pdb
import torch
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


class DirectNormLoss(nn.Module):

    def __init__(self, num_class=1000, nd_loss_factor=1.0):
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
     
    def forward(self, s_emb, t_emb, T_EMB, labels):
        nd_loss = self.project_center(s_emb=s_emb, t_emb=t_emb, T_EMB=T_EMB, labels=labels) * self.nd_loss_factor
        
        return nd_loss / len(labels)