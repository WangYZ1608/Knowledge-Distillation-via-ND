import torch.nn as nn

class EmbTrans(nn.Module):

    def __init__(self, student, model_name):
        super(EmbTrans, self).__init__()

        self.student = student

        if 'vits' in model_name:
            s_emb = 512
            t_emb = 384
            self.fc1 = nn.Sequential(
                nn.BatchNorm1d(s_emb),
                nn.Dropout(0.5),
                nn.Linear(s_emb, t_emb)
                )
            
        elif 'vitb' in model_name:
            s_emb = 512
            t_emb = 768
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
        emb_fea, logits = self.student(x, embed=True)
        if embed:
            if self.fc1 is not None:
                emb_fea = self.fc1(emb_fea)
            return emb_fea, logits
        else:
            return logits
