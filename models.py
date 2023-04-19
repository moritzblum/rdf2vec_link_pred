import torch
from torch import nn
from torch.nn import Parameter


class ClassicLinkPredNet (nn.Module):
    def __init__(self, embedding_dim, hidden_dim, batch_norm=True, dropout=0.0):
        super().__init__()
        self.batch_norm = batch_norm
        self.fc0 = nn.Linear(embedding_dim * 3, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.bn0 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)

        self.dropout = torch.nn.Dropout(dropout)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, h_head, h_relation, h_tail, head_idx=None, rel_idx=None, tail_idx=None):
        out = torch.cat([h_head, h_relation, h_tail], dim=1)
        out = self.dropout(out)
        out = torch.sigmoid(self.fc0(out))
        if self.batch_norm:
            out = self.bn0(out)
        out = self.dropout(out)
        out = torch.sigmoid(self.fc1(out))
        if self.batch_norm:
            out = self.bn1(out)
        out = torch.sigmoid(self.fc_out(out))
        return torch.flatten(out)


class VectorReconstructionNet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, batch_norm=True, dropout=0.0):
        super().__init__()
        self.batch_norm = batch_norm
        self.fc0 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, embedding_dim)
        self.bn0 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)

        self.dropout = torch.nn.Dropout(dropout)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, h_head, h_relation):
        out = torch.cat([h_head, h_relation], dim=1)
        out = self.dropout(out)
        out = torch.sigmoid(self.fc0(out))
        if self.batch_norm:
            out = self.bn0(out)
        out = torch.sigmoid(self.fc1(out))
        out = self.dropout(out)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.fc_out(out)
        return out


class DistMult (nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super().__init__()
        self.head = torch.nn.Embedding(num_entities, embedding_dim)
        self.rel = torch.nn.Embedding(num_relations, embedding_dim)
        self.tail = torch.nn.Embedding(num_entities, embedding_dim)

    def forward(self, head_idx, rel_idx, tail_idx):

        h_head = self.head(head_idx)
        h_relation = self.rel(rel_idx)
        h_tail = self.head(tail_idx)

        out = torch.sigmoid(torch.sum(h_head * h_relation * h_tail, dim=1))
        out = torch.flatten(out)

        return out

    def l3_regularization(self):
        return (self.head.weight.norm(p=3) ** 3 + self.rel.weight.norm(p=3) ** 3)


class DistMultNet (nn.Module):
    def __init__(self, embedding_dim, hidden_dim, batch_norm=True, layers=1):
        super().__init__()
        self.batch_norm = batch_norm
        self.layers = layers
        self.fc_head = nn.Linear(embedding_dim, hidden_dim)
        self.fc_rel = nn.Linear(embedding_dim, hidden_dim)
        self.fc_tail = nn.Linear(embedding_dim, hidden_dim)

        self.bn0 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)

        self.fc1_head = nn.Linear(embedding_dim, hidden_dim)
        self.fc1_rel = nn.Linear(embedding_dim, hidden_dim)
        self.fc1_tail = nn.Linear(embedding_dim, hidden_dim)

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

        if self.batch_norm:
            h_head = self.bn0(h_head)
            h_relation = self.bn1(h_relation)
            h_tail = self.bn2(h_tail)

        if self.layers >= 2:
            h_head = torch.sigmoid(self.fc1_head(h_head))
            h_relation = torch.sigmoid(self.fc1_rel(h_relation))
            h_tail = torch.sigmoid(self.fc1_tail(h_tail))

        out = torch.sigmoid(torch.sum(h_head * h_relation * h_tail, dim=1))
        out = torch.flatten(out)

        return out


class ComplExNet (nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_entities=0, num_relations=0, batch_norm=True):
        super().__init__()
        self.batch_norm = batch_norm
        self.fc_head = nn.Linear(embedding_dim, hidden_dim)
        self.fc_rel = nn.Linear(embedding_dim, hidden_dim)
        self.fc_tail = nn.Linear(embedding_dim, hidden_dim)

        self.Ei = torch.nn.Embedding(num_entities, hidden_dim)
        self.Ri = torch.nn.Embedding(num_relations, hidden_dim)

        self.input_dropout = torch.nn.Dropout(0.1)
        self.bn0 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn4 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn5 = torch.nn.BatchNorm1d(hidden_dim)

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

        if self.batch_norm:
            h_head = self.bn0(h_head)
            h_relation = self.bn1(h_relation)
            h_tail = self.bn2(h_tail)
            head_i = self.bn3(head_i)
            rel_i = self.bn4(rel_i)
            tail_i = self.bn5(tail_i)

        h_head = self.input_dropout(h_head)
        h_relation = self.input_dropout(h_relation)
        h_tail = self.input_dropout(h_tail)

        head_i = self.input_dropout(head_i)
        rel_i = self.input_dropout(rel_i)
        tail_i = self.input_dropout(tail_i)

        real_real_real = (h_head * h_relation * h_tail).sum(dim=1)
        real_imag_imag = (h_head * rel_i * tail_i).sum(dim=1)
        imag_real_imag = (head_i * h_relation * tail_i).sum(dim=1)
        imag_imag_real = (head_i * rel_i * h_tail).sum(dim=1)

        pred = real_real_real + real_imag_imag + imag_real_imag - imag_imag_real
        pred = torch.sigmoid(pred)
        return pred
