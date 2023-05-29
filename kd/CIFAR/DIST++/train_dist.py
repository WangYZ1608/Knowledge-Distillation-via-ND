import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,2'
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import models
from models.embtrans_cifar import EmbTrans
from Dataset import CIFAR
from utils import colorstr, Save_Checkpoint, AverageMeter, DirectNormLoss, DIST
from torchsummaryX import summary

import numpy as np
from pathlib import Path
import os
import time
import json
import random
import logging
import argparse
import warnings
from torch.utils.tensorboard import SummaryWriter
import pdb

def train(model, teacher, T_EMB, train_dataloader, optimizer, criterion, kd_loss, nd_loss, args, epoch):
    train_loss = AverageMeter()
    train_error = AverageMeter()

    Cls_loss = AverageMeter()
    Div_loss = AverageMeter()
    Norm_Dir_loss = AverageMeter()

    # Model on train mode
    model.train()
    teacher.eval()
    step_per_epoch = len(train_dataloader)
    
    for step, (images, labels) in enumerate(train_dataloader):
        start = time.time()
        images, labels = images.cuda(), labels.cuda()

        # compute output
        s_emb, s_logits = model(images, embed=True)

        with torch.no_grad():
            t_emb, t_logits = teacher(images, embed=True)
        
        # cls loss
        cls_loss = criterion(s_logits, labels) * args.cls_loss_factor
        # Kd loss
        div_loss = kd_loss(s_logits, t_logits) * args.kd_loss_factor * min(1.0, epoch/args.warm_up) 
        # ND loss
        norm_dir_loss = nd_loss(s_emb=s_emb, t_emb=t_emb, T_EMB=T_EMB, labels=labels)

        loss = cls_loss + div_loss + norm_dir_loss
        # measure accuracy and record loss
        batch_size = images.size(0)
        _, pred = s_logits.data.cpu().topk(1, dim=1)
        train_error.update(torch.ne(pred.squeeze(), labels.cpu()).float().sum().item() / batch_size, batch_size)
        train_loss.update(loss.item(), batch_size)

        Cls_loss.update(cls_loss.item(), batch_size)
        Div_loss.update(div_loss.item(), batch_size)
        Norm_Dir_loss.update(norm_dir_loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        s1 = '\r{} [{}/{}]'.format(t, step+1, step_per_epoch)
        s2 = ' - {:.2f}ms/step - nd_loss: {:.3f} - div_loss: {:.3f} - cls_loss: {:.3f} - train_loss: {:.3f} - train_acc: {:.3f}'.format(
             1000 * (time.time() - start), norm_dir_loss.item(), div_loss.item(), cls_loss.item(), train_loss.val, 1-train_error.val)

        print(s1+s2, end='', flush=True)

    print()
    return Norm_Dir_loss.avg, Div_loss.avg, Cls_loss.avg, train_loss.avg, train_error.avg


def test(model, test_dataloader, criterion):
    test_loss = AverageMeter()
    test_error = AverageMeter()

    # Model on eval mode
    model.eval()

    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.cuda(), labels.cuda()

            # compute logits
            logits = model(images, embed=False)

            loss = criterion(logits, labels)

            # measure accuracy and record loss
            batch_size = images.size(0)
            _, pred = logits.data.cpu().topk(1, dim=1)
            test_error.update(torch.ne(pred.squeeze(), labels.cpu()).float().sum().item() / batch_size, batch_size)
            test_loss.update(loss.item(), batch_size)

    return test_loss.avg, test_error.avg


def epoch_loop(model, teacher, train_set, test_set, args):
    # data loaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    
    # model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.DataParallel(model, device_ids=args.gpus)
    model.to(device)
    teacher = nn.DataParallel(teacher, device_ids=args.gpus)
    teacher.to(device)

    # loss
    criterion = nn.CrossEntropyLoss().to(device)
    kd_loss = DIST(beta=1, gamma=1, tau=args.t).to(device)
    nd_loss = DirectNormLoss(num_class=100, nd_loss_factor=args.nd_loss_factor).to(device)
    # optimizer
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    
    # 权重
    save_dir = Path(args.save_dir)
    weights = save_dir / 'weights'
    weights.mkdir(parents=True, exist_ok=True)
    last = weights / 'last'
    best = weights / 'best'

    # acc,loss
    acc_loss = save_dir / 'acc_loss'
    acc_loss.mkdir(parents=True, exist_ok=True)

    train_acc_savepath = acc_loss / 'train_acc.npy'
    train_loss_savepath = acc_loss / 'train_loss.npy'
    val_acc_savepath = acc_loss / 'val_acc.npy'
    val_loss_savepath = acc_loss / 'val_loss.npy'

    # tensorboard
    logdir = save_dir / 'logs'
    logdir.mkdir(parents=True, exist_ok=True)
    summary_writer = SummaryWriter(logdir, flush_secs=120)
    
    # resume
    if args.resume:
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_error = checkpoint['best_error']
        train_acc = checkpoint['train_acc']
        train_loss = checkpoint['train_loss']
        test_acc = checkpoint['test_acc']
        test_loss = checkpoint['test_loss']
        logger.info(colorstr('green', 'Resuming training from {} epoch'.format(start_epoch)))
    else:
        start_epoch = 0
        best_error = 0
        train_acc = []
        train_loss = []
        test_acc = []
        test_loss = []

    # Train model
    best_error = 1
    for epoch in range(start_epoch, args.epochs):
        if epoch in [150, 180, 210]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        print("Epoch {}/{}".format(epoch + 1, args.epochs))
        norm_dir_loss, div_loss, cls_loss, train_epoch_loss, train_error = train(model=model,
                                                                                 teacher=teacher,
                                                                                 T_EMB=T_EMB,
                                                                                 train_dataloader=train_loader,
                                                                                 optimizer=optimizer,
                                                                                 criterion=criterion,
                                                                                 kd_loss=kd_loss,
                                                                                 nd_loss=nd_loss,
                                                                                 args=args,
                                                                                 epoch=epoch)
        test_epoch_loss, test_error = test(model=model, 
                                           test_dataloader=test_loader,
                                           criterion=criterion)
        
        s = "Train Loss: {:.3f}, Train Acc: {:.3f}, Test Loss: {:.3f}, Test Acc: {:.3f}, lr: {:.5f}".format(
            train_epoch_loss, 1-train_error, test_epoch_loss, 1-test_error, optimizer.param_groups[0]['lr'])
        logger.info(colorstr('green', s))

        # save acc,loss
        train_loss.append(train_epoch_loss)
        train_acc.append(1-train_error)
        test_loss.append(test_epoch_loss)
        test_acc.append(1-test_error)

        # save model
        is_best = test_error < best_error
        best_error = min(best_error, test_error)
        state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_error': best_error,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'test_acc': test_acc,
                'test_loss': test_loss,
            }
        
        last_path = last / 'epoch_{}_loss_{:.3f}_acc_{:.3f}'.format(
            epoch + 1, test_epoch_loss, 1-test_error)
        best_path = best / 'epoch_{}_acc_{:.3f}'.format(
                epoch + 1, 1-best_error)

        Save_Checkpoint(state, last, last_path, best, best_path, is_best)

        # tensorboard
        if epoch == 1:
            images, labels = next(iter(train_loader))
            img_grid = torchvision.utils.make_grid(images)
            summary_writer.add_image('Cifar Image', img_grid)
        summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        summary_writer.add_scalar('train_loss', train_epoch_loss, epoch)
        summary_writer.add_scalar('train_error', train_error, epoch)
        summary_writer.add_scalar('val_loss', test_epoch_loss, epoch)
        summary_writer.add_scalar('val_error', test_error, epoch)

        summary_writer.add_scalar('nd_loss', norm_dir_loss, epoch)
        summary_writer.add_scalar('kd_loss', div_loss, epoch)
        summary_writer.add_scalar('cls_loss', cls_loss, epoch)

    summary_writer.close()
    if not os.path.exists(train_acc_savepath) or not os.path.exists(train_loss_savepath):
        np.save(train_acc_savepath, train_acc)
        np.save(train_loss_savepath, train_loss)
        np.save(val_acc_savepath, test_acc)
        np.save(val_loss_savepath, test_loss)


if __name__ == "__main__":
    model_names = sorted(name for name in models.__dict__ 
                         if name.islower() and not name.startswith("__") 
                         and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
    parser.add_argument("--model_name", type=str, default="resnet20_cifar", choices=model_names, help="model architecture")
    parser.add_argument("--dataset", type=str, default='cifar100')
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch_size", type=int, default=128, help="batch size per gpu")
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument("--teacher", type=str, default="resnet50_cifar", help="teacher architecture")
    parser.add_argument("--teacher_weights", type=str, default="", help="teacher weights path")
    parser.add_argument("--cls_loss_factor", type=float, default=1.0, help="cls loss weight factor")
    parser.add_argument("--kd_loss_factor", type=float, default=1.0, help="KL loss weight factor")
    parser.add_argument("--t", type=float, default=4.0, help="temperature")
    parser.add_argument("--nd_loss_factor", type=float, default=1.0, help="ND loss weight factor")
    parser.add_argument("--warm_up", type=float, default=20.0, help='loss weight warm up epochs')

    parser.add_argument("--gpus", type=list, default=[0, 1])
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
    parser.add_argument("--resume", type=str, help="best ckpt's path to resume most recent training")
    parser.add_argument("--save_dir", type=str, default="./run", help="save path, eg, acc_loss, weights, tensorboard, and so on")
    args = parser.parse_args()

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

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [line:%(lineno)d] %(message)s',
                        datefmt='%d %b %Y %H:%M:%S')
    logger = logging.getLogger(__name__)

    args.batch_size = args.batch_size * len(args.gpus)

    logger.info(colorstr('green', "Distribute train, gpus:{}, total batch size:{}, epoch:{}".format(args.gpus, args.batch_size, args.epochs)))
    
    train_set, test_set, num_class = CIFAR(name=args.dataset)
    model = models.__dict__[args.model_name](num_class=num_class)
    if args.model_name in ['wrn40_1_cifar', 'mobilenetv2', 'shufflev1_cifar', 'shufflev2_cifar']:
        model = EmbTrans(student=model, model_name=args.model_name)
    teacher = models.__dict__[args.teacher](num_class=num_class)

    # resnet56/resnet32x4, no need emb size
    if args.teacher_weights:
        print('Load Teacher Weights')
        teacher_ckpt = torch.load(args.teacher_weights)['model']
        teacher.load_state_dict(teacher_ckpt)

        for param in teacher.parameters():
            param.requires_grad = False

    # res56    ./ckpt/teacher/resnet56/center_emb_train.json
    # res32x4  ./ckpt/teacher/resnet32x4/center_emb_train.json
    # wrn40_2  ./ckpt/teacher/wrn_40_2/center_emb_train.json
    # res50    ./ckpt/teacher/resnet50/center_emb_train.json
    # class-mean
    with open("./ckpt/teacher/resnet56/center_emb_train.json", 'r') as f:
        T_EMB = json.load(f)
    f.close()
    
    logger.info(colorstr('green', 'Use ' + args.teacher + ' Training ' + args.model_name + ' ...'))
    # Train the model
    epoch_loop(model=model, teacher=teacher, train_set=train_set, test_set=test_set, args=args)