import torch, math
import torch.nn as nn

__all__ = ['wrn40_2_cifar', 'wrn40_1_cifar',]

wrn = {
    "wrn40_1": [16, 16, 32, 64],
    "wrn40_2": [16, 32, 64, 128], 
    }

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            (not self.equalInOut)
            and nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            or None
        )

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride
        )

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride):
        layers = []
        for i in range(nb_layers):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):

    def __init__(self, depth, num_filters, num_class):
        super(WideResNet, self).__init__()

        assert (depth - 4) % 6 == 0, "depth should be 6n+4"
        n = (depth - 4) // 6
        block = BasicBlock
        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, num_filters[0], num_filters[1], block, 1)
        self.block2 = NetworkBlock(n, num_filters[1], num_filters[2], block, 2)
        self.block3 = NetworkBlock(n, num_filters[2], num_filters[3], block, 2)
        
        self.bn = nn.BatchNorm2d(num_filters[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(num_filters[3], num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        

    def forward(self, x, embed=True):
        x = self.conv1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.relu(self.bn(x))

        x = self.avgpool(x)
        emb_fea = torch.flatten(x, 1)
        logits = self.fc(emb_fea)

        if embed:
            return emb_fea, logits
        else:
            return logits


def wrn40_1_cifar(model_name='wrn40_1', **kwargs):
    model = WideResNet(depth=40, num_filters=wrn[model_name], **kwargs)
    return model


def wrn40_2_cifar(model_name='wrn40_2', **kwargs):
    model = WideResNet(depth=40, num_filters=wrn[model_name], **kwargs)
    return model