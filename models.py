import torch
from torch import nn


class ClassicLinkPredNet (nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.fc0 = nn.Linear(embedding_dim * 3, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, h_head, h_relation, h_tail):
        out = torch.cat([h_head, h_relation, h_tail], dim=1)
        out = torch.relu(self.fc0(out))
        out = torch.relu(self.fc1(out))
        #out = torch.sigmoid(self.fc2(out))
        out = torch.sigmoid(self.fc_out(out))
        return out


class VectorReconstructionNet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.fc0 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, embedding_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, h_head, h_relation):
        out = torch.cat([h_head, h_relation], dim=1)
        out = torch.sigmoid(self.fc0(out))
        out = torch.sigmoid(self.fc1(out))
        out = self.fc_out(out)
        return out