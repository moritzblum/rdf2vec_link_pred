import argparse
import os
import os.path as osp
import json
from tqdm import tqdm
import numpy as np
import time
import torch
from torch import nn
from gensim.models import Word2Vec
from torch_geometric.data import Data
from torch.nn import CosineEmbeddingLoss, MSELoss

from models import VectorReconstructionNet, ClassicLinkPredNet

BATCH_SIZE = 1000
EMBEDDING_DIM = 200
HIDDEN_DIM = 200
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def read_lp_data(path, entities, relations, data_sample, entity_embedding=False, relation_embeddings=False):
    edge_index = []
    edge_type = []

    with open(f'{path}/{data_sample}.txt') as triples_in:
        for line in triples_in:
            head, relation, tail = line[:-1].split('\t')
            edge_index.append([entities.index(head), entities.index(tail)])
            edge_type.append(relations.index(relation))

    if entity_embedding:
        x_entity = []
        for e in entities:
            if 'http://wd/' + e in wv_model.wv:
                x_entity.append(wv_model.wv['http://wd/' + e])
            else:
                x_entity.append(np.zeros(EMBEDDING_DIM))
        x_entity = torch.tensor(x_entity)
    else:
        x_entity = None

    if not relation_embeddings:
        # derive features according to Paulheim et al.
        relation_dict = {'http://wd/' + r: [] for r in relations}
        x_relation = []
        for (head, tail), relation in zip(edge_index, edge_type):
            relation_dict['http://wd/' + relations[relation]].append(x_entity[head] - x_entity[tail])
        for r in relations:
            if len(relation_dict['http://wd/' + r]) > 0:
                x_relation.append(torch.mean(torch.stack(relation_dict['http://wd/' + r]), dim=0))
            else:
                x_relation.append(torch.zeros(EMBEDDING_DIM))
        x_relation = torch.stack(x_relation)
    else:
        x_relation = []
        for r in relations:
            x_relation.append(wv_model.wv['http://wd/' + r])
        x_relation = torch.tensor(x_relation)


    return (Data(x_entity=x_entity,
                 x_relation=x_relation,
                 edge_index=torch.tensor(edge_index).t(),
                 edge_type=torch.tensor(edge_type)))


def train_vector_reconstruction(model, optimizer, entity_input='head'):
    model.train()

    start = time.time()

    edge_index = data_train.edge_index.t().to(DEVICE)
    edge_type = data_train.edge_type.to(DEVICE)

    edge_index_batches = torch.split(edge_index, BATCH_SIZE)
    edge_type_batches = torch.split(edge_type, BATCH_SIZE)

    loss_total = 0
    for edge_idxs, relation_idxs in zip(edge_index_batches, edge_type_batches):
        optimizer.zero_grad()
        relation_embeddings = data_train.x_relation[relation_idxs].float().to(DEVICE)

        head_embeddings = data_train.x_entity[edge_idxs[:, 0]].float().to(DEVICE)
        tail_embeddings = data_train.x_entity[edge_idxs[:, 1]].float().to(DEVICE)

        if entity_input == 'head':
            y_pred = model.forward(head_embeddings, relation_embeddings)
            loss = loss_function(y_pred, tail_embeddings)
            loss_total += loss
            loss.backward()
            optimizer.step()
        else:
            y_pred = model.forward(tail_embeddings, relation_embeddings)
            loss = loss_function(y_pred, head_embeddings)
            loss_total += loss
            loss.backward()
            optimizer.step()

    end = time.time()
    print('esapsed time:', end - start)
    print('loss:', loss_total / len(edge_index_batches))
    return (loss_total)


def negative_sampling(edge_index, num_nodes):
    # Sample edges by corrupting either the subject or the object of each edge.
    mask_1 = torch.rand(edge_index.size(0)) < 0.5
    mask_2 = ~mask_1

    mask_1 = mask_1.to(DEVICE)
    mask_2 = mask_2.to(DEVICE)

    neg_edge_index = edge_index.clone()
    neg_edge_index[mask_1, 0] = torch.randint(num_nodes, (1, mask_1.sum()), device=DEVICE)
    neg_edge_index[mask_2, 1] = torch.randint(num_nodes, (1, mask_2.sum()), device=DEVICE)

    return neg_edge_index


def train_triple_scoring(model, optimizer):
    start = time.time()

    edge_index = data_train.edge_index.t().to(DEVICE)
    edge_type = data_train.edge_type.to(DEVICE)

    edge_index_batches = torch.split(edge_index, 128)
    edge_type_batches = torch.split(edge_type, 128)

    loss_total = 0
    for edge_idxs, relation_idx in zip(edge_index_batches, edge_type_batches):
        optimizer.zero_grad()

        relation_embeddings = data_train.x_relation[relation_idx].float().to(DEVICE)

        head_embeddings_pos = data_train.x_entity[edge_idxs[:, 0]].float().to(DEVICE)
        tail_embeddings_pos = data_train.x_entity[edge_idxs[:, 1]].float().to(DEVICE)

        edge_idxs_neg = negative_sampling(edge_idxs, len(entities)).to(DEVICE)

        head_embeddings_neg = data_train.x_entity[edge_idxs_neg[:, 0]].float().to(DEVICE)
        tail_embeddings_neg = data_train.x_entity[edge_idxs_neg[:, 1]].float().to(DEVICE)

        y_pred_pos = model.forward(head_embeddings_pos, relation_embeddings, tail_embeddings_pos)
        y_pred_neg = model.forward(head_embeddings_neg, relation_embeddings, tail_embeddings_neg)

        out = torch.cat([y_pred_pos, y_pred_neg])
        gt = torch.cat([torch.ones_like(y_pred_pos), torch.zeros_like(y_pred_neg)])

        cross_entropy_loss = loss_function(out, gt)
        loss_total += cross_entropy_loss

        cross_entropy_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    end = time.time()
    print('esapsed time:', end - start)
    print('loss:', loss_total / len(edge_index_batches))




@torch.no_grad()
def compute_mrr_triple_scoring(model, num_entities, eval_edge_index, eval_edge_type, data_train, data_val, data_test, fast=False):

    ranks = []
    num_samples = eval_edge_type.numel() if not fast else 5000
    for i in tqdm(range(num_samples)):
        (src, dst), rel = eval_edge_index[:, i], eval_edge_type[i]

        # Try all nodes as tails, but delete true triplets:
        tail_mask = torch.ones(num_entities, dtype=torch.bool)
        for (heads, tails), types in [
            (data_train.edge_index.to(DEVICE), data_train.edge_type.to(DEVICE)),
            (data_val.edge_index.to(DEVICE), data_val.edge_type.to(DEVICE)),
            (data_test.edge_index.to(DEVICE), data_test.edge_type.to(DEVICE)),
        ]:
            tail_mask[tails[(heads == src) & (types == rel)]] = False

        tail = torch.arange(num_entities)[tail_mask]
        tail = torch.cat([torch.tensor([dst]), tail])
        head = torch.full_like(tail, fill_value=src)
        eval_edge_type = torch.full_like(tail, fill_value=rel).to(DEVICE)

        h = data_train.x_entity[head].to(DEVICE)
        r = data_train.x_relation[eval_edge_type].to(DEVICE)
        t = data_train.x_entity[tail].to(DEVICE)
        out = model.forward(h.float(), r.float(), t.float())


        rank = compute_rank(out)
        ranks.append(rank)

        # Try all nodes as heads, but delete true triplets:
        head_mask = torch.ones(num_entities, dtype=torch.bool)
        for (heads, tails), types in [
            (data_train.edge_index.to(DEVICE), data_train.edge_type.to(DEVICE)),
            (data_val.edge_index.to(DEVICE), data_val.edge_type.to(DEVICE)),
            (data_test.edge_index.to(DEVICE), data_test.edge_type.to(DEVICE)),
        ]:
            head_mask[heads[(tails == dst) & (types == rel)]] = False

        head = torch.arange(num_entities)[head_mask]
        head = torch.cat([torch.tensor([src]), head])
        tail = torch.full_like(head, fill_value=dst)
        eval_edge_type = torch.full_like(head, fill_value=rel).to(DEVICE)

        h = data_train.x_entity[head].to(DEVICE)
        r = data_train.x_relation[eval_edge_type].to(DEVICE)
        t = data_train.x_entity[tail].to(DEVICE)
        out = model.forward(h.float(), r.float(), t.float())

        rank = compute_rank(out)
        ranks.append(rank)

    num_ranks = len(ranks)
    ranks = torch.tensor(ranks, dtype=torch.float)
    return (1. / ranks).mean(), ranks.mean(), ranks[ranks <= 10].size(0) / num_ranks



@torch.no_grad()
def compute_rank(ranks):
    # print(ranks)
    # fair ranking prediction as the average
    # of optimistic and pessimistic ranking
    true = ranks[0]
    optimistic = (ranks > true).sum() + 1
    pessimistic = (ranks >= true).sum()
    return (optimistic + pessimistic).float() * 0.5


@torch.no_grad()
def compute_mrr_vector_reconstruction(model_tail_pred, model_head_pred, entity_features, relation_features, edge_index, edge_type,
                fast=False):
    eval_loss_function = loss_function = MSELoss(reduction='none')  # CosineEmbeddingLoss(reduction='none')

    num_samples = edge_type.numel() if not fast else 5000
    ranks = []
    for i in tqdm(range(num_samples)):
        # for i in tqdm(range(5000)):
        (src, dst), rel = edge_index[:, i], edge_type[i]

        out_model = model_tail_pred(torch.reshape(entity_features[src], (1, EMBEDDING_DIM)).float().to(DEVICE),
                                    torch.reshape(relation_features[rel], (1, EMBEDDING_DIM)).float().to(DEVICE))

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
        tail_features = entity_features[tail].to(DEVICE)

        out = out_model.repeat(len(tail), 1)

        out = eval_loss_function(out, tail_features)  # , -torch.ones(len(tail)).to(DEVICE))
        out = - torch.mean(out, dim=1)  # added for MSE loss
        # print(out)
        # print(out)
        rank = compute_rank(out)
        ranks.append(rank)

        out_model = model_head_pred(torch.reshape(entity_features[dst], (1, EMBEDDING_DIM)).float().to(DEVICE),
                                    torch.reshape(relation_features[rel], (1, EMBEDDING_DIM)).float().to(DEVICE))

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
        head_features = entity_features[head].to(DEVICE)

        out = out_model.repeat(len(head), 1)

        out = eval_loss_function(out, head_features)  # , -torch.ones(len(head)).to(DEVICE))
        out = - torch.mean(out, dim=1)  # added for MSE loss

        rank = compute_rank(out)
        ranks.append(rank)

    num_ranks = len(ranks)
    ranks = torch.tensor(ranks, dtype=torch.float)
    return (1. / ranks).mean(), ranks.mean(), ranks[ranks <= 10].size(0) / num_ranks



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='rdf2vec link prediction')
    # specify dataset
    parser.add_argument('--dataset', type=str, default='fb15k', help='fb15k or fb15-237')
    parser.add_argument('--architecture', type=str, default='ClassicLinkPredNet', help="ClassicLinkPredNet or VectorReconstructionNet")
    parser.add_argument('--relationfeatures', type=str, default='standard', help="standard (use the ones trained by RDF2Vec) or derived (derive them form the entity features automatically)")

    args = parser.parse_args()
    dataset = args.dataset
    architecture = args.architecture
    relation_embeddings = True if args.relationfeatures == 'standard' else False


    # read in entities and relations for indexing
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

    # load RDF2Vec models for features
    wv_model = Word2Vec.load(f'./data/{dataset}_rdf2vec/model')

    data_train = read_lp_data(path='./data/fb15k/', entities=entities, relations=relations, data_sample='train',
                              entity_embedding=True, relation_embeddings=relation_embeddings)
    data_val = read_lp_data(path='./data/fb15k/', entities=entities, relations=relations, data_sample='valid',
                            entity_embedding=True, relation_embeddings=relation_embeddings)
    data_test = read_lp_data(path='./data/fb15k/', entities=entities, relations=relations, data_sample='test',
                             entity_embedding=True, relation_embeddings=relation_embeddings)

    # ClassicLinkPredNet
    if architecture == 'ClassicLinkPredNet':
        model = ClassicLinkPredNet(EMBEDDING_DIM, HIDDEN_DIM)
        model.to(DEVICE)
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # , weight_decay=1e-4
        model.train()

        for epoch in range(0, 5000):
            train_triple_scoring(model, optimizer)
            if epoch % 50 == 0:
                mrr, mr, hits10 = compute_mrr_triple_scoring(model, len(entities), data_val.edge_index, data_val.edge_type, data_train,
                                           data_val, data_test, fast=True)
                print('mrr:', mrr, 'mr:', mr, 'hits@10:', hits10)

        torch.save(model.state_dict(), 'model_ClassicLinkPredNet.pth')
        mrr, mr, hits10 = compute_mrr_triple_scoring(model, len(entities), data_val.edge_index, data_val.edge_type, data_train, data_val, data_test, fast=False)
        print('val mrr:', mrr, 'mr:', mr, 'hits@10:', hits10)
        mrr, mr, hits10 = compute_mrr_triple_scoring(model, len(entities), data_test.edge_index, data_test.edge_type, data_train, data_val, data_test, fast=False)
        print('test mrr:', mrr, 'mr:', mr, 'hits@10:', hits10)

    # VectorReconstructionNet
    elif architecture == 'VectorReconstructionNet':
        torch.backends.cudnn.benchmark = True
        model_tail_pred = VectorReconstructionNet(EMBEDDING_DIM, HIDDEN_DIM)
        model_head_pred = VectorReconstructionNet(EMBEDDING_DIM, HIDDEN_DIM)

        model_tail_pred.to(DEVICE)
        model_head_pred.to(DEVICE)
        optimizer_tail_pred = torch.optim.Adam(model_tail_pred.parameters(), lr=.001)
        optimizer_head_pred = torch.optim.Adam(model_head_pred.parameters(), lr=.001)

        loss_function = MSELoss()  # CosineEmbeddingLoss()

        for epoch in range(0, 5000):
            train_vector_reconstruction(model_tail_pred, optimizer_tail_pred, 'head')
            train_vector_reconstruction(model_head_pred, optimizer_head_pred, 'tail')
            if epoch % 50 == 0:
                mrr, mr, hits10 = compute_mrr_vector_reconstruction(model_tail_pred, model_head_pred, data_train.x_entity, data_train.x_relation,
                                              data_val.edge_index, data_val.edge_type, fast=True)
                print('mrr:', mrr, 'mr:', mr, 'hits@10:', hits10)

        mrr, mr, hits10 = compute_mrr_vector_reconstruction(model_tail_pred, model_head_pred, data_train.x_entity, data_train.x_relation,
                                      data_val.edge_index, data_val.edge_type, fast=False)
        print('val mrr:', mrr, 'mr:', mr, 'hits@10:', hits10)
        mrr, mr, hits10 = compute_mrr_vector_reconstruction(model_tail_pred, model_head_pred, data_train.x_entity, data_train.x_relation,
                                      data_test.edge_index, data_test.edge_type, fast=False)
        print('test mrr:', mrr, 'mr:', mr, 'hits@10:', hits10)
        torch.save(model_tail_pred.state_dict(), 'model_VectorReconstructionNet_tail_pred.pth')
        torch.save(model_head_pred.state_dict(), 'model_VectorReconstructionNet_head_pred.pth')


