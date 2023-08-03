import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import trunc_normal_


class VisionTransformerDistilled(VisionTransformer):
    """ Vision Transformer w/ Distillation Token and Head
    Distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """
    def __init__(self, **kwargs):
        weight_init = kwargs.pop('weight_init', '')
        super().__init__(**kwargs)

        self.dis_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes)

        self.deit_init_weights(weight_init)
        # del super.forward_features()
        # del self.forward_head()
        # del self.forward()
    
    def deit_init_weights(self, mode=''):
        trunc_normal_(self.dis_token, std=0.02)
        super().init_weights(mode=mode)
    
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = torch.cat((
            self.cls_token.expand(x.shape[0], -1, -1),
            self.dis_token.expand(x.shape[0], -1, -1), x
        ), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x
    
    def forward_head(self, x, embed: bool = False):
        emb = (x[:, 0] + x[:, 1]) / 2
        logits, dist = self.head(x[:, 0]), self.head_dist(x[:, 1])
        if embed:
            return emb, logits, dist
        else:
            return (logits + dist) / 2
    
    def forward(self, x, embed):
        x = self.forward_features(x)
        if embed:
            emb, logits, dist = self.forward_head(x, embed=True)
            return emb, logits, dist
        else:
            logits = self.forward_head(x, embed=False)
            return logits


def deit_base_patch16(**kwarg):
    model = VisionTransformerDistilled(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, **kwarg
    )
    return model