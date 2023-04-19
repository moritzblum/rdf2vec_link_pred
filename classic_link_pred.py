import time

import torch
import numpy as np
from torch.functional import F
from torch_geometric.data import Data
import os

from tqdm import tqdm

from main import compute_rank
from models import DistMult

DEVICE = torch.device('cuda:0')
EMBEDDING_DIM = 200


"""
Already integrated into main.py
"""


def negative_sampling(edge_index, num_nodes, eta=1):
    # Sample edges by corrupting either the subject or the object of each edge.
    mask_1 = torch.rand(edge_index.size(0) * eta) < 0.5
    mask_2 = ~mask_1

    mask_1 = mask_1.to(DEVICE)
    mask_2 = mask_2.to(DEVICE)

    neg_edge_index = edge_index.clone().repeat(eta, 1)
    neg_edge_index[mask_1, 0] = torch.randint(num_nodes, (1, mask_1.sum()), device=DEVICE)
    neg_edge_index[mask_2, 1] = torch.randint(num_nodes, (1, mask_2.sum()), device=DEVICE)

    return neg_edge_index


def read_lp_data(path, entities, relations, data_sample):
    edge_index = []
    edge_type = []

    with open(f'{path}/{data_sample}.txt') as triples_in:
        for line in triples_in:
            head, relation, tail = line[:-1].split('\t')
            edge_index.append([entities.index(head), entities.index(tail)])
            edge_type.append(relations.index(relation))

    return Data(edge_index=torch.tensor(edge_index).t(),
                edge_type=torch.tensor(edge_type))


def train_standard_lp(eta=5, regularization=False):
    model.train()
    start = time.time()

    train_edge_index_t = data_train.edge_index.t().to(DEVICE)
    train_edge_type = data_train.edge_type.to(DEVICE)

    edge_index_batches = torch.split(train_edge_index_t, 1000)
    edge_type_batches = torch.split(train_edge_type, 1000)

    indices = np.arange(len(edge_index_batches))
    np.random.shuffle(indices)

    loss_total = 0
    for i in indices:
        edge_idxs, relation_idx = edge_index_batches[i], edge_type_batches[i]
        optimizer.zero_grad()

        edge_idxs_neg = negative_sampling(edge_idxs, len(entities), eta=eta)

        out_pos = model.forward(edge_idxs[:, 0], relation_idx, edge_idxs[:, 1])
        out_neg = model.forward(edge_idxs_neg[:, 0], relation_idx.repeat(eta), edge_idxs_neg[:, 1])

        out = torch.cat([out_pos, out_neg], dim=0)
        gt = torch.cat([torch.ones(len(relation_idx)), torch.zeros(len(relation_idx) * eta)], dim=0).to(DEVICE)

        loss = loss_function(out, gt)
        if regularization:
               loss += 0.000001 * model.l3_regularization()
        loss_total += loss
        loss.backward()
        optimizer.step()
    end = time.time()
    print('esapsed time:', end - start)
    print('loss:', loss_total / len(edge_index_batches))


@torch.no_grad()
def compute_mrr_triple_scoring(model, eval_edge_index, eval_edge_type,
                               fast=False):
    model.eval()
    ranks = []
    num_samples = eval_edge_type.numel() if not fast else 5000
    for triple_index in tqdm(range(num_samples)):
        (src, dst), rel = eval_edge_index[:, triple_index], eval_edge_type[triple_index]

        # Try all nodes as tails, but delete true triplets:
        tail_mask = torch.ones(len(entities), dtype=torch.bool)
        for (heads, tails), types in [
            (data_train.edge_index, data_train.edge_type),
            (data_val.edge_index, data_val.edge_type),
            (data_test.edge_index, data_test.edge_type),
        ]:
            tail_mask[tails[(heads == src) & (types == rel)]] = False

        tail = torch.arange(len(entities))[tail_mask]
        tail = torch.cat([torch.tensor([dst]), tail])
        head = torch.full_like(tail, fill_value=src)
        eval_edge_typ_tensor = torch.full_like(tail, fill_value=rel).to(DEVICE)

        out = model.forward(head.to(DEVICE), eval_edge_typ_tensor.to(DEVICE), tail.to(DEVICE))

        rank = compute_rank(out)
        ranks.append(rank)

        # Try all nodes as heads, but delete true triplets:
        head_mask = torch.ones(len(entities), dtype=torch.bool)
        for (heads, tails), types in [
            (data_train.edge_index, data_train.edge_type),
            (data_val.edge_index, data_val.edge_type),
            (data_test.edge_index, data_test.edge_type),
        ]:
            head_mask[heads[(tails == dst) & (types == rel)]] = False

        head = torch.arange(len(entities))[head_mask]
        head = torch.cat([torch.tensor([src]), head])
        tail = torch.full_like(head, fill_value=dst)
        eval_edge_typ_tensor = torch.full_like(head, fill_value=rel).to(DEVICE)

        out = model.forward(head.to(DEVICE), eval_edge_typ_tensor.to(DEVICE), tail.to(DEVICE))

        rank = compute_rank(out)
        ranks.append(rank)

    num_ranks = len(ranks)
    ranks = torch.tensor(ranks, dtype=torch.float)
    return (1. / ranks).mean(), \
           ranks.mean(), \
           ranks[ranks <= 10].size(0) / num_ranks, \
           ranks[ranks <= 5].size(0) / num_ranks, \
           ranks[ranks <= 3].size(0) / num_ranks, \
           ranks[ranks <= 1].size(0) / num_ranks


if __name__ == '__main__':

    dataset = 'fb15k'

    entities = set()
    relations = set()
    for graph in ['train', 'valid', 'test']:
        with open(f'./data/{dataset}/{graph}.txt') as triples_in:
            for line in triples_in:
                head, relation, tail = line[:-1].split('\t')
                entities.add(head)
                entities.add(tail)
                relations.add(relation)
    entities = list(entities)
    relations = list(relations)

    if not os.path.isfile(f'./data/{dataset}/train_onehot.pt'):
        data_train = read_lp_data(path=f'./data/{dataset}/', entities=entities, relations=relations,
                                  data_sample='train')
        torch.save(data_train, f'./data/{dataset}/train_onehot.pt')
    if not os.path.isfile(f'./data/{dataset}/val_onehot.pt'):
        data_val = read_lp_data(path=f'./data/{dataset}/', entities=entities, relations=relations,
                                data_sample='valid')
        torch.save(data_val, f'./data/{dataset}/val_onehot.pt')
    if not os.path.isfile(f'./data/{dataset}/test_onehot.pt'):
        data_test = read_lp_data(path=f'./data/{dataset}/', entities=entities, relations=relations,
                                 data_sample='test')
        torch.save(data_test, f'./data/{dataset}/test_onehot.pt')

    data_train = torch.load(f'./data/{dataset}/train_onehot.pt')
    data_val = torch.load(f'./data/{dataset}/val_onehot.pt')
    data_test = torch.load(f'./data/{dataset}/test_onehot.pt')

    model = DistMult(len(entities), len(relations), 200)
    model.to(DEVICE)

    loss_function = torch.nn.BCELoss()  # torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005) #, weight_decay=1e-5)
    model.train()

    for epoch in range(0, 1000):
        train_standard_lp()
        if epoch % 50 == 0:
            mrr, mr, hits10, hits5, hits3, hits1 = compute_mrr_triple_scoring(model,
                                                                              data_train.edge_index,
                                                                              data_train.edge_type,
                                                                              fast=True)
            print('val mrr:', mrr, 'mr:', mr, 'hits@10:', hits10, 'hits@5:', hits5, 'hits@3:', hits3, 'hits@1:', hits1)
