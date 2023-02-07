import torch
from torch import nn
from torch.nn import Parameter


class ClassicLinkPredNet (nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.fc0 = nn.Linear(embedding_dim * 3, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
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


# todo debug
class DistMultNet (nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.fc_head = nn.Linear(embedding_dim, hidden_dim)
        self.fc_rel = nn.Linear(embedding_dim, hidden_dim)
        self.fc_tail = nn.Linear(embedding_dim, hidden_dim)

        self.fc1_head = nn.Linear(hidden_dim, hidden_dim)
        self.fc1_rel = nn.Linear(hidden_dim, hidden_dim)
        self.fc1_tail = nn.Linear(hidden_dim, hidden_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, h_head, h_relation, h_tail):

        head = torch.sigmoid(self.fc_head(h_head))
        rel = torch.sigmoid(self.fc_rel(h_relation))
        tail = torch.sigmoid(self.fc_tail(h_tail))

        head = self.fc1_head(h_head)
        rel = self.fc1_rel(h_relation)
        tail = self.fc1_tail(h_tail)

        return torch.sum(head * rel * tail, dim=1)
