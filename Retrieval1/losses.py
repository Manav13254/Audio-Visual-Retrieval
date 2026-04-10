import torch.nn as nn
import torch.nn.functional as F

class CosineTripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, sim_pos, sim_neg):
        loss = F.relu(sim_neg - sim_pos + self.margin)
        return loss.mean()