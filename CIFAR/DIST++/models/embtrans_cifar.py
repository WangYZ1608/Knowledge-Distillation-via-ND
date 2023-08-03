import torch.nn as nn

class EmbTrans(nn.Module):

    def __init__(self, student, model_name):
        super(EmbTrans, self).__init__()

        self.student = student

        if 'wrn' in model_name:
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
        
    def forward(self, x, embed=True):
        emb_fea, logits = self.student(x, embed=True)
        if embed:
            if self.fc1 is not None:
                emb_fea = self.fc1(emb_fea)
            return emb_fea, logits
        else:
            return logits
