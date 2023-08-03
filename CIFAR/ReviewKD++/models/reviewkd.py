import math
import pdb
import torch.nn.functional as F
from torch import nn
import torch

from .resnet_cifar import resnet20_cifar, resnet8x4_cifar
from .wrn_cifar import wrn40_1_cifar
from .mobilenetv2_cifar import mobilenetv2
from .shufflenetv1 import shufflev1_cifar
from .shufflenetv2 import shufflev2_cifar

class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                    nn.Conv2d(mid_channel*2, 2, kernel_size=1),
                    nn.Sigmoid(),
                )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None, out_shape=None):
        n,_,h,w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, (shape,shape), mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = (x * z[:,0].view(n,1,h,w) + y * z[:,1].view(n,1,h,w))
        # output 
        if x.shape[-1] != out_shape:
            x = F.interpolate(x, (out_shape, out_shape), mode="nearest")
        y = self.conv2(x)
        return y, x


class ReviewKD(nn.Module):
    def __init__(
        self, student, in_channels, out_channels, shapes, out_shapes, model_name
    ):  
        super(ReviewKD, self).__init__()
        self.student = student
        self.shapes = shapes
        self.out_shapes = shapes if out_shapes is None else out_shapes

        abfs = nn.ModuleList()

        mid_channel = min(512, in_channels[-1])
        for idx, in_channel in enumerate(in_channels):
            abfs.append(ABF(in_channel, mid_channel, out_channels[idx], idx < len(in_channels)-1))
        self.abfs = abfs[::-1]
        # self.to('cuda')

        if 'wrn40_1_cifar' in model_name:
            s_emb = 64
            t_emb = 128
            self.fc1 = nn.Sequential(
                nn.BatchNorm1d(s_emb),
                nn.Dropout(0.5),
                nn.Linear(s_emb, t_emb)
                )
        
        elif 'mobilenetv2' in model_name:
            s_emb = 1280
            t_emb = 2048
            self.fc1 = nn.Sequential(
                nn.BatchNorm1d(s_emb),
                nn.Dropout(0.5),
                nn.Linear(s_emb, t_emb)
                )
            
        elif 'shufflev1' in model_name:
            s_emb = 960
            t_emb = 256
            self.fc1 = nn.Sequential(
                nn.BatchNorm1d(s_emb),
                nn.Dropout(0.5),
                nn.Linear(s_emb, t_emb)
                )
        
        elif 'shufflev2' in model_name:
            s_emb = 1024
            t_emb = 256
            self.fc1 = nn.Sequential(
                nn.BatchNorm1d(s_emb),
                nn.Dropout(0.5),
                nn.Linear(s_emb, t_emb)
                )
        
        else:
            self.fc1 = None
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        student_features, emb_fea, logit = self.student(x,is_feat=True)
        if self.fc1 is not None:
            emb_fea = self.fc1(emb_fea)
        # logit = student_features[1]
        # x = student_features[0][::-1]
        x = student_features[::-1]
        results = []
        out_features, res_features = self.abfs[0](x[0], out_shape=self.out_shapes[0])
        results.append(out_features)
        for features, abf, shape, out_shape in zip(x[1:], self.abfs[1:], self.shapes[1:], self.out_shapes[1:]):
            out_features, res_features = abf(features, res_features, shape, out_shape)
            results.insert(0, out_features)

        return results, emb_fea, logit


def build_review_kd(student_name, num_class, teacher_name):
    out_shapes = None
    if 'resnet20_cifar' == student_name and 'resnet56_cifar' == teacher_name:
        student = resnet20_cifar(num_class=num_class)
        in_channels = [16,32,64,64]
        out_channels = [16,32,64,64]
        shapes = [1,8,16,32,32]
    
    elif 'resnet8x4_cifar' == student_name and 'resnet32x4_cifar' == teacher_name:
        student = resnet8x4_cifar(num_class=num_class)
        in_channels = [64,128,256,256]
        out_channels = [64,128,256,256]
        shapes = [1,8,16,32,32]
    
    elif 'wrn40_1_cifar' == student_name and 'wrn40_2_cifar' == teacher_name:
        student = wrn40_1_cifar(num_class=num_class)
        in_channels = [16,32,64,64]
        out_channels = [32,64,128,128]
        shapes = [1,8,16,32]
    
    elif 'mobilenetv2' == student_name and 'resnet50_cifar' == teacher_name:
        student = mobilenetv2(num_class=num_class)
        in_channels = [12,16,48,160,1280]
        out_channels = [256,512,1024,2048,2048]
        shapes = [1,2,4,8,16]
    
    elif 'shufflev1_cifar' == student_name and 'resnet32x4_cifar' == teacher_name:
        student = shufflev1_cifar(num_class=num_class)
        in_channels = [240,480,960,960]
        out_channels = [64,128,256,256]
        out_shapes = [1,8,16,32]
        shapes = [1,4,8,16]
    
    elif 'shufflev2_cifar' == student_name and 'resnet32x4_cifar' == teacher_name:
        student = shufflev2_cifar(num_class=num_class)
        in_channels = [116,232,464,1024]
        out_channels = [64,128,256,256]
        out_shapes = [1,8,16,32]
        shapes = [1,4,8,16]
    
    else:
        assert False

    backbone = ReviewKD(
        student=student,
        in_channels=in_channels,
        out_channels=out_channels,
        shapes = shapes,
        out_shapes = out_shapes,
        model_name=student_name
    )
    return backbone


def hcl(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n,c,h,w = fs.shape
        loss = F.mse_loss(fs, ft, reduction='mean')
        cnt = 1.0
        tot = 1.0
        for l in [4,2,1]:
            if l >=h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l,l))
            tmpft = F.adaptive_avg_pool2d(ft, (l,l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all
