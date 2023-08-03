import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.parallel
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import Models, timm
from Models.embtrans import EmbTrans
from Dataset import ImageNet
from Dataset.data import Mixup
from utils import colorstr, Save_Checkpoint, adjust_learning_rate, param_groups_weight_decay, DirectNormLoss, KDLoss

# from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import numpy as np
from pathlib import Path
import os
import time
import json
import random
import argparse
import warnings
from torch.utils.tensorboard import SummaryWriter
import pdb


def init_distributed(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
        
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    args.gpu = int(os.environ["LOCAL_RANK"])
    print("Use GPU: {} for training".format(args.gpu))

    args.distributed = True
    # torch.cuda.set_device(args.gpu)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=world_size, rank=rank)
    return rank, world_size, args.gpu


def main(args, teacher, T_EMB):
    rank, world_size, args.gpu = init_distributed(args)

    cudnn.benchmark = True

    if dist.get_rank() == 0:
        # weights
        save_dir = Path(args.save_dir)
        weights = save_dir / 'weights'
        weights.mkdir(parents=True, exist_ok=True)
        last = weights / 'last'
        best = weights / 'best'

        # acc,loss
        acc_loss = save_dir / 'acc_loss'
        acc_loss.mkdir(parents=True, exist_ok=True)
        train_acc_top1_savepath = acc_loss / 'train_acc_top1.npy'
        train_acc_top5_savepath = acc_loss / 'train_acc_top5.npy'
        train_loss_savepath = acc_loss / 'train_loss.npy'
        val_acc_top1_savepath = acc_loss / 'val_acc_top1.npy'
        val_acc_top5_savepath = acc_loss / 'val_acc_top5.npy'
        val_loss_savepath = acc_loss / 'val_loss.npy'

        # tensorboard
        logdir = save_dir / 'logs'
        logdir.mkdir(parents=True, exist_ok=True)
        summary_writer = SummaryWriter(logdir, flush_secs=120)

    # dataset
    train_dataset, val_dataset, num_class = ImageNet(args=args)
    
    if args.lr is None:
        args.lr = args.blr * args.batch_size / 256
        args.batch_size = int(args.batch_size / world_size)

    if dist.get_rank() == 0:
        print(colorstr('green', "base lr: {}, absolute lr: {}".format(args.blr, args.lr)))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        # val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False)

    # loss
    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    mixup_fn = None
    mixup_active = args.mixup > 0. or args.cutmix > 0.
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix, 
            label_smoothing=args.smoothing, 
            num_classes=num_class
        )
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy().to(device)
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy().to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    kd_loss = KDLoss(kd_loss_factor=args.kd_loss_factor).to(device)
    nd_loss = DirectNormLoss(num_class=num_class, nd_loss_factor=args.nd_loss_factor).to(device)

    # model
    model = Models.__dict__[args.model_name](num_classes=num_class, drop_path_rate=args.drop_path)
    if args.model_name in ['deit_base_patch16']:
        model = EmbTrans(student=model, model_name=args.model_name)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # if dist.get_rank() == 0:
    #     print("Model = %s" % str(model_without_ddp))
    #     print('number of params (M): %.2f' % (n_parameters / 1.e6))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        teacher.to(device)
    
    params = param_groups_weight_decay(model=model_without_ddp, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(params=params,
                                  lr=args.lr,
                                  betas=(0.9, 0.95),
                                  )
    
    if args.resume:
        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        elif torch.cuda.is_available():
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
           
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_acc = torch.tensor(checkpoint['best_acc'])
        if args.gpu is not None:
            # best_acc may be from a checkpoint from a different GPU
            best_acc = best_acc.to(args.gpu)

        train_acc_top1 = checkpoint['train_acc_top1']
        train_acc_top5 = checkpoint['train_acc_top5']
        train_loss = checkpoint['train_loss']
        test_acc_top1 = checkpoint['test_acc_top1']
        test_acc_top5 = checkpoint['test_acc_top5']
        test_loss = checkpoint['test_loss']
        if dist.get_rank() == 0:
            print(colorstr('green', 'Resuming training from {} epoch'.format(start_epoch)))
    else:
        start_epoch = 0
        best_acc = 0
        train_acc_top1 = []
        train_acc_top5 = []
        train_loss = []
        test_acc_top1 = []
        test_acc_top5 = []
        test_loss = []
    

    if args.evaluate:
        val_epoch_loss, val_acc1, val_acc5 = validate(model=model,
                                                      val_loader=val_loader,
                                                      criterion=criterion,
                                                      args=args)
        if dist.get_rank() == 0:
            print("Test Loss: {}, Test Acc Top1: {}, Test Acc Top5: {}".format(val_epoch_loss, val_acc1, val_acc5))
        return
    
    for epoch in range(start_epoch, args.epochs):
        if dist.get_rank() == 0:
            print("Epoch {}/{}".format(epoch + 1, args.epochs))
        if args.distributed:
            train_sampler.set_epoch(epoch)

        norm_dir_loss, distilled_loss, cls_loss, train_epoch_loss, train_acc1, train_acc5 = train(model=model,
                                                                                                  teacher=teacher,
                                                                                                  T_EMB=T_EMB,
                                                                                                  train_loader=train_loader,
                                                                                                  optimizer=optimizer,
                                                                                                  criterion=criterion,
                                                                                                  kd_loss=kd_loss,
                                                                                                  nd_loss=nd_loss,
                                                                                                  mixup_fn=mixup_fn,
                                                                                                  args=args,
                                                                                                  epoch=epoch)

        val_epoch_loss, val_acc1, val_acc5 = validate(model=model,
                                                      val_loader=val_loader,
                                                      args=args)
        
        s = "Train Loss: {:.3f}, Train Acc Top1: {:.3f}, Train Acc Top5: {:.3f}, Test Loss: {:.3f}, Test Acc Top1: {:.3f}, Test Acc Top5: {:.3f}, lr: {:.5f}".format(
            train_epoch_loss, train_acc1, train_acc5, val_epoch_loss, val_acc1, val_acc5, optimizer.param_groups[0]['lr'])
        if dist.get_rank() == 0:
            print(colorstr('green', s))

            # save acc,loss
            train_loss.append(train_epoch_loss)
            train_acc_top1.append(train_acc1)
            train_acc_top5.append(train_acc5)
            test_loss.append(val_epoch_loss)
            test_acc_top1.append(val_acc1)
            test_acc_top5.append(val_acc5)

            # save model
            is_best = val_acc1 > best_acc
            best_acc = max(best_acc, val_acc1)
            state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'train_acc_top1': train_acc_top1,
                'train_acc_top5': train_acc_top5,
                'train_loss': train_loss,
                'test_acc_top1': test_acc_top1,
                'test_acc_top5': test_acc_top5,
                'test_loss': test_loss,
            }

            last_path = last / 'epoch_{}_loss_{:.4f}_acc_{:.3f}'.format(
                epoch + 1, val_epoch_loss, val_acc1)
            best_path = best / 'epoch_{}_acc_{:.4f}'.format(
                epoch + 1, best_acc)
            Save_Checkpoint(state, last, last_path, best, best_path, is_best)

            if epoch == 1:
                images, labels = next(iter(train_loader))
                img_grid = torchvision.utils.make_grid(images)
                summary_writer.add_image('ImageNet Image', img_grid)

            summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            summary_writer.add_scalar('train_loss', train_epoch_loss, epoch)
            summary_writer.add_scalar('train_acc_top1', train_acc1, epoch)
            summary_writer.add_scalar('train_acc_top5', train_acc5, epoch)
            summary_writer.add_scalar('val_loss', val_epoch_loss, epoch)
            summary_writer.add_scalar('val_acc_top1', val_acc1, epoch)
            summary_writer.add_scalar('val_acc_top5', val_acc5, epoch)

            summary_writer.add_scalar('norm_dir_loss', norm_dir_loss, epoch)
            summary_writer.add_scalar('distilled_loss', distilled_loss, epoch)
            summary_writer.add_scalar('cls_loss', cls_loss, epoch)
    
    if dist.get_rank() == 0:
        summary_writer.close()
        if not os.path.exists(train_acc_top1_savepath) or not os.path.exists(train_loss_savepath):
            np.save(train_acc_top1_savepath, train_acc_top1)
            np.save(train_acc_top5_savepath, train_acc_top5)
            np.save(train_loss_savepath, train_loss)
            np.save(val_acc_top1_savepath, test_acc_top1)
            np.save(val_acc_top5_savepath, test_acc_top5)
            np.save(val_loss_savepath, test_loss)


def train(model, teacher, T_EMB, train_loader, optimizer, criterion, kd_loss, nd_loss, mixup_fn, args, epoch):
    train_loss = AverageMeter()
    train_acc1 = AverageMeter()
    train_acc5 = AverageMeter()

    Cls_loss = AverageMeter()
    Distilled_loss = AverageMeter()
    Norm_Dir_loss = AverageMeter()

    # Model on train mode
    model.train()
    teacher.eval()
    step_per_epoch = len(train_loader)
    for step, (images, labels) in enumerate(train_loader):
        torch.cuda.synchronize()
        start = time.time()

        adjust_learning_rate(optimizer, step / step_per_epoch + epoch, args)

        if args.gpu is not None and torch.cuda.is_available():
            images = images.cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)
        
        if mixup_fn is not None:
            # images, labels = mixup_fn(images, labels)
            images, labels, centers = mixup_fn(images, labels, T_EMB)
        
        # compute output
        s_emb, s_logits, s_dist = model(images, embed=True)

        with torch.no_grad():
            t_emb, t_dist = teacher(images)

        # cls loss
        cls_loss = criterion(s_logits, labels)
        # Kd loss
        dis_loss = kd_loss(s_dist, t_dist) * min(1.0, epoch/args.warm_up)
        # nd loss
        norm_dir_loss = nd_loss(s_emb=s_emb, t_emb=t_emb, T_EMB=centers, labels=None)

        loss = cls_loss + dis_loss + norm_dir_loss

        train_loss.update(loss.item(), images.size(0))
        Cls_loss.update(cls_loss.item(), images.size(0))
        Distilled_loss.update(dis_loss.item(), images.size(0))
        Norm_Dir_loss.update(norm_dir_loss.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        s1 = '\r{} [{}/{}]'.format(t, step+1, step_per_epoch)
        s2 = ' - {:.2f}ms/step - nd_loss: {:.3f} - dis_loss: {:.3f} - cls_loss: {:.3f} - train_loss: {:.3f} - train_acc_top1: {:.3f} - train_acc_top5: {:.3f}'.format(
             1000 * (time.time()-start), norm_dir_loss.item(), dis_loss.item(), cls_loss.item(), train_loss.val, train_acc1.val, train_acc5.val)
        if dist.get_rank() == 0:
            print(s1+s2, end='', flush=True)
    
    if dist.get_rank() == 0:
        print()
    return Norm_Dir_loss.avg, Distilled_loss.avg, Cls_loss.avg, train_loss.avg, train_acc1.avg, train_acc5.avg


def validate(model, val_loader, args):
    val_loss = AverageMeter()
    val_acc1 = AverageMeter()
    val_acc5 = AverageMeter()
    
    # model to evaluate mode
    criterion = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        for step, (images, labels) in enumerate(val_loader):
            if args.gpu is not None and torch.cuda.is_available():
                 images = images.cuda(args.gpu, non_blocking=True)
                 labels = labels.cuda(args.gpu, non_blocking=True)

            # compute output
            logits = model(images, embed=False)
            loss = criterion(logits, labels)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))

            # Average loss and accuracy across processes
            if args.distributed:
                loss = reduce_tensor(loss, args)
                acc1 = reduce_tensor(acc1, args)
                acc5 = reduce_tensor(acc5, args)

            val_loss.update(loss.item(), images.size(0))
            val_acc1.update(acc1[0].item(), images.size(0))
            val_acc5.update(acc5[0].item(), images.size(0))
    
    return val_loss.avg, val_acc1.avg, val_acc5.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def reduce_tensor(tensor, args):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


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
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


if __name__ == '__main__':
    model_names = sorted(name for name in Models.__dict__ 
                         if name.islower() and not name.startswith("__") 
                         and callable(Models.__dict__[name]))
    
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training with Distributed Data Parallel.')
    # model parameters
    parser.add_argument("--model_name", type=str, default="vit_base_patch16", choices=model_names, help="model architecture")
    parser.add_argument("--model_weights", type=str, default="", help="model weights path")
    parser.add_argument("--drop_path", type=float, default=0.1, help='Drop path rate')

    # teacher parameters
    parser.add_argument("--teacher", type=str, default="regnety_160", help="teacher architecture")
    parser.add_argument("--teacher_weights", type=str, default="", help="teacher weights path")
    parser.add_argument("--kd_loss_factor", type=float, default=1.0, help="kd loss weight factor")
    parser.add_argument("--t", type=float, default=1.0, help="temperature")
    parser.add_argument("--nd_loss_factor", type=float, default=1.0, help="nd loss weight factor")
    parser.add_argument("--warm_up", type=float, default=5.0, help='loss weight warm up epochs')

    # optimizer parameters
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=256, help="total batch size")
    parser.add_argument('--workers', default=64, type=int, help='number of data loading workers')
    parser.add_argument("--weight_decay", type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N', help='epochs to warmup LR')
    
    # augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8, help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0, help='cutmix alpha, cutmix enabled if > 0.')
    
    parser.add_argument('--evaluate', action='store_true', help='evaluate model on validation set')

    parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')

    parser.add_argument("--resume", type=str, help="ckpt's path to resume most recent training")
    parser.add_argument("--save_dir", type=str, default="./run", help="save path, eg, acc_loss, weights, tensorboard, and so on")
    args = parser.parse_args()


    if args.model_weights:
        train_data, test_data, num_class = ImageNet(args=args)
        model = Models.__dict__[args.model_name](num_classes=num_class, drop_path_rate=args.drop_path)
        print('Using pretrained model for {}'.format(args.model_name))
        model_ckpt = torch.load(args.model_weights)
        model.load_state_dict(model_ckpt)

        for param in model.parameters():
            param.requires_grad = False
    
    if args.teacher_weights:
        print('Using pretrained model for teacher {}'.format(args.teacher))
        teacher = timm.create_model("regnety_160", num_classes=1000)
        teacher_ckpt = torch.load(args.teacher_weights)
        teacher.load_state_dict(teacher_ckpt["model"])

        for param in teacher.parameters():
            param.requires_grad = False
    
    # regnety_160   ./ckpt/regnet160/center_emb_train.json

    with open("./ckpt/regnet160/center_emb_train.json", 'r') as f:
        T_EMB = json.load(f)
    f.close()

    print(colorstr('green', 'Use ' + args.teacher + ' Training ' + args.model_name + ' ...'))    
    main(args=args, teacher=teacher, T_EMB=T_EMB)