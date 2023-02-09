import torch
from torch import nn
from torch.nn import Parameter


# todo implement batch normalization as an option
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

    def forward(self, h_head, h_relation, h_tail, head_idx=None, rel_idx=None, tail_idx=None):
        out = torch.cat([h_head, h_relation, h_tail], dim=1)
        out = torch.relu(self.fc0(out))
        out = torch.relu(self.fc1(out))
        out = torch.sigmoid(self.fc_out(out))
        return torch.flatten(out)


# todo implement batch normalization as an option
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


# todo implement batch normalization as an option
class DistMultNet (nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        print('inint DistMultNet')
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

    def forward(self, h_head, h_relation, h_tail, head_idx=None, rel_idx=None, tail_idx=None):

        h_head = torch.sigmoid(self.fc_head(h_head))
        h_relation = torch.sigmoid(self.fc_rel(h_relation))
        h_tail = torch.sigmoid(self.fc_tail(h_tail))

        h_head = self.fc1_head(h_head)
        h_relation = self.fc1_rel(h_relation)
        h_tail = self.fc1_tail(h_tail)

        out = torch.sigmoid(torch.sum(h_head * h_relation * h_tail, dim=1))
        out = torch.flatten(out)

        return out


# todo implement batch normalization as an option
class ComplExNet (nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_entities=0, num_relations=0):
        super().__init__()
        self.fc_head = nn.Linear(embedding_dim, hidden_dim)
        self.fc_rel = nn.Linear(embedding_dim, hidden_dim)
        self.fc_tail = nn.Linear(embedding_dim, hidden_dim)

        self.Ei = torch.nn.Embedding(num_entities, hidden_dim)
        self.Ri = torch.nn.Embedding(num_relations, hidden_dim)

        self.input_dropout = torch.nn.Dropout(0.1)
        self.bn0 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, h_head, h_relation, h_tail, head_idx=None, rel_idx=None, tail_idx=None):

        h_head = torch.sigmoid(self.fc_head(h_head))
        h_relation = torch.sigmoid(self.fc_rel(h_relation))
        h_tail = torch.sigmoid(self.fc_tail(h_tail))

        head_i = self.Ei(head_idx)
        rel_i = self.Ri(rel_idx)
        tail_i = self.Ei(tail_idx)

        h_head = self.bn0(h_head)
        h_head = self.input_dropout(h_head)
        h_relation = self.bn0(h_relation)
        h_relation = self.input_dropout(h_relation)
        h_tail = self.bn0(h_tail)
        h_tail = self.input_dropout(h_tail)

        head_i = self.bn1(head_i)
        head_i = self.input_dropout(head_i)
        rel_i = self.bn1(rel_i)
        rel_i = self.input_dropout(rel_i)
        tail_i = self.bn1(tail_i)
        tail_i = self.input_dropout(tail_i)

        real_real_real = (h_head * h_relation * h_tail).sum(dim=1)
        real_imag_imag = (h_head * rel_i * tail_i).sum(dim=1)
        imag_real_imag = (head_i * h_relation * tail_i).sum(dim=1)
        imag_imag_real = (head_i * rel_i * h_tail).sum(dim=1)

        pred = real_real_real + real_imag_imag + imag_real_imag - imag_imag_real
        pred = torch.sigmoid(pred)
        return pred
