import torch.nn as nn
from torchsummaryX import summary

class EmbTrans(nn.Module):

    def __init__(self, student, model_name):
        super(EmbTrans, self).__init__()

        self.student = student

        if model_name in ['deit_base_patch16']:
            # student: deit-b, teacher: regnety-16GF
            s_emb = 768
            t_emb = 3024
            self.fc1 = nn.Sequential(
                nn.BatchNorm1d(s_emb),
                nn.Dropout(0.5),
                nn.Linear(s_emb, t_emb)
            )
        
        else:
            self.fc1 = None

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.normal_(m.bias, mean=0.0, std=0.01)
        
    def forward(self, x, embed=True):
        if embed:
            emb, logits, dist = self.student(x, embed=True)
            if self.fc1 is not None:
                emb = self.fc1(emb)
            return emb, logits, dist
        else:
            logits = self.student(x, embed=False)
            return logits
