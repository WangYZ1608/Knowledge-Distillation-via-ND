import torch
from torch import nn
import torch.nn.functional as F

from .resnet  import resnet18
from .mobilenet import mobilenetv1

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

    def forward(self, x, y=None, shape=None):
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
        y = self.conv2(x)
        return y, x


class ReviewKD(nn.Module):
    def __init__(
        self, student, in_channels, out_channels, mid_channel, student_name, teacher_name):
        super(ReviewKD, self).__init__()
        self.shapes = [1,7,14,28,56]
        self.student = student

        abfs = nn.ModuleList()

        for idx, in_channel in enumerate(in_channels):
            abfs.append(ABF(in_channel, mid_channel, out_channels[idx], idx < len(in_channels)-1))

        self.abfs = abfs[::-1]

        if 'resnet18' == student_name:
            if 'resnet34' == teacher_name:
                self.fc1 = None
            elif teacher_name in ['resnet50', 'resnet101', 'resnet152']:
                s_emb = 512
                t_emb = 2048
                self.fc1 = nn.Sequential(
                    nn.BatchNorm1d(s_emb),
                    nn.Dropout(0.5),
                    nn.Linear(s_emb, t_emb)
                    )
        
        elif 'mobilenetv1' == student_name:
            s_emb = 1024
            t_emb = 2048
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
        # student_features = self.student(x,is_feat=True)
        # logit = student_features[1]
        # x = student_features[0][::-1]
        student_features, emb_fea, logit = self.student(x,is_feat=True)
        if self.fc1 is not None:
            emb_fea = self.fc1(emb_fea)
        x = student_features[::-1]
        results = []
        out_features, res_features = self.abfs[0](x[0])
        results.append(out_features)
        for features, abf, shape in zip(x[1:], self.abfs[1:], self.shapes[1:]):
            out_features, res_features = abf(features, res_features, shape)
            results.insert(0, out_features)

        return results, emb_fea, logit


def build_review_kd(student_name, teacher_name):
    if 'resnet18' == student_name:
        in_channels = [64,128,256,512,512]
        mid_channel = 512
        student = resnet18()
        if 'resnet34' == teacher_name:
            out_channels = [64,128,256,512,512]
        elif teacher_name in ['resnet50', 'resnet101', 'resnet152']:
            out_channels = [256,512,1024,2048,2048]

    elif student_name == 'mobilenetv1':
        in_channels = [128,256,512,1024,1024]
        out_channels = [256,512,1024,2048,2048]
        mid_channel = 256
        student = mobilenetv1()
        
    else:
        print(student, 'is not defined.')
        assert False
    model = ReviewKD(student, in_channels, out_channels, mid_channel, student_name, teacher_name)
    return model


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
