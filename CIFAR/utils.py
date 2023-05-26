import os, shutil
import torch
import torch.nn.functional as F
import torch.nn as nn

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
     
    def forward(self, s_emb, t_emb, T_EMB, labels):
        nd_loss = self.project_center(s_emb=s_emb, t_emb=t_emb, T_EMB=T_EMB, labels=labels) * self.nd_loss_factor
        
        return nd_loss / len(labels)


class KDLoss(nn.Module):
    '''
	Distilling the Knowledge in a Neural Network
	https://arxiv.org/pdf/1503.02531.pdf
	'''
    def __init__(self, kl_loss_factor=1.0, T=4.0):
        super(KDLoss, self).__init__()
        self.T = T
        self.kl_loss_factor = kl_loss_factor

    def forward(self, s_out, t_out):
        kd_loss = F.kl_div(F.log_softmax(s_out / self.T, dim=1), 
                           F.softmax(t_out / self.T, dim=1), 
                           reduction='batchmean',
                           ) * self.T * self.T
        return kd_loss * self.kl_loss_factor


class DKDLoss(nn.Module):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, alpha=1.0, beta=1.0, T=4.0):
        super(DKDLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.T = T
    
    def dkd_loss(self, s_logits, t_logits, labels):
        gt_mask = self.get_gt_mask(s_logits, labels)
        other_mask = self.get_other_mask(s_logits, labels)
        s_pred = F.softmax(s_logits / self.T, dim=1)
        t_pred = F.softmax(t_logits / self.T, dim=1)
        s_pred = self.cat_mask(s_pred, gt_mask, other_mask)
        t_pred = self.cat_mask(t_pred, gt_mask, other_mask)
        s_log_pred = torch.log(s_pred)
        tckd_loss = (
            F.kl_div(s_log_pred, t_pred, size_average=False)
            * (self.T**2)
            / labels.shape[0]
        )
        pred_teacher_part2 = F.softmax(
            t_logits / self.T - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            s_logits / self.T - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
            * (self.T**2)
            / labels.shape[0]
        )
        return self.alpha * tckd_loss + self.beta * nckd_loss
    
    def get_gt_mask(self, logits, labels):
        labels = labels.reshape(-1)
        mask = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1).bool()
        return mask
    
    def get_other_mask(self, logits, labels):
        labels = labels.reshape(-1)
        mask = torch.ones_like(logits).scatter_(1, labels.unsqueeze(1), 0).bool()
        return mask
    
    def cat_mask(self, t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        rt = torch.cat([t1, t2], dim=1)
        return rt
    
    def forward(self, s_logits, t_logits, labels):
        loss = self.dkd_loss(s_logits, t_logits, labels)

        return loss
 

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