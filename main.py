import argparse
import os
import os.path as osp
import json

from datetime import datetime
from ray import air, tune
from ray.tune import Tuner, ExperimentAnalysis, CLIReporter
from ray.tune.schedulers import ASHAScheduler
from tqdm import tqdm
import numpy as np
import time
import torch
from torch import nn
from gensim.models import Word2Vec
from torch_geometric.data import Data
from torch.nn import CosineEmbeddingLoss, MSELoss, BCELoss

from models import VectorReconstructionNet, ClassicLinkPredNet, DistMultNet, ComplExNet

EMBEDDING_DIM = 200
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def read_lp_data(path, entities, relations, data_sample, wv_model, relation_embeddings=False):
    edge_index = []
    edge_type = []

    with open(f'{path}/{data_sample}.txt') as triples_in:
        for line in triples_in:
            head, relation, tail = line[:-1].split('\t')
            edge_index.append([entities.index(head), entities.index(tail)])
            edge_type.append(relations.index(relation))

    x_entity = []
    for e in entities:
        if 'http://wd/' + e in wv_model.wv:
            x_entity.append(wv_model.wv['http://wd/' + e])
        else:
            x_entity.append(np.zeros(EMBEDDING_DIM))
    x_entity = torch.tensor(x_entity)

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

    return Data(x_entity=x_entity,
                x_relation=x_relation,
                edge_index=torch.tensor(edge_index).t(),
                edge_type=torch.tensor(edge_type))


def train_vector_reconstruction(model, optimizer, entity_input='head'):
    model.train()
    start = time.time()

    edge_index_batches = torch.split(train_edge_index_t, BATCH_SIZE)
    edge_type_batches = torch.split(train_edge_type, BATCH_SIZE)

    indices = np.arange(len(edge_index_batches))
    np.random.shuffle(indices)

    loss_total = 0
    for i in indices:
        edge_idxs, relation_idxs = edge_index_batches[i], edge_type_batches[i]
        optimizer.zero_grad()
        relation_embeddings = x_relation[relation_idxs]

        head_embeddings = x_entity[edge_idxs[:, 0]]
        tail_embeddings = x_entity[edge_idxs[:, 1]]

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


def train_triple_scoring(model, optimizer, model_type='ClassicLinkPredNet'):
    start = time.time()

    edge_index_batches = torch.split(train_edge_index_t, BATCH_SIZE)
    edge_type_batches = torch.split(train_edge_type, BATCH_SIZE)

    indices = np.arange(len(edge_index_batches))
    np.random.shuffle(indices)

    loss_total = 0
    for i in indices:
        edge_idxs, relation_idx = edge_index_batches[i], edge_type_batches[i]
        optimizer.zero_grad()

        relation_embeddings = x_relation[relation_idx]

        head_embeddings_pos = x_entity[edge_idxs[:, 0]]
        tail_embeddings_pos = x_entity[edge_idxs[:, 1]]

        edge_idxs_neg = negative_sampling(edge_idxs, num_entities)

        head_embeddings_neg = x_entity[edge_idxs_neg[:, 0]]
        tail_embeddings_neg = x_entity[edge_idxs_neg[:, 1]]

        if model_type in ['ClassicLinkPredNet', 'DistMultNet']:
            out = model.forward(torch.cat([head_embeddings_pos, head_embeddings_neg], dim=0),
                                torch.cat([relation_embeddings, relation_embeddings], dim=0),
                                torch.cat([tail_embeddings_pos, tail_embeddings_neg], dim=0))
        if model_type == 'ComplExNet':
            out = model.forward(torch.cat([head_embeddings_pos, head_embeddings_neg], dim=0),
                                torch.cat([relation_embeddings, relation_embeddings], dim=0),
                                torch.cat([tail_embeddings_pos, tail_embeddings_neg], dim=0),
                                torch.cat([edge_idxs[:, 0], edge_idxs_neg[:, 0]], dim=0),
                                torch.cat([relation_idx, relation_idx], dim=0),
                                torch.cat([edge_idxs[:, 1], edge_idxs_neg[:, 1]], dim=0))

        # classification target: first for true, then for corrupted triples
        gt = torch.cat([torch.ones(len(relation_idx)), torch.zeros(len(relation_idx))], dim=0).to(DEVICE)

        loss = loss_function(out, gt)
        loss_total += loss
        loss.backward()
        optimizer.step()
    end = time.time()
    print('esapsed time:', end - start)
    print('loss:', loss_total / len(edge_index_batches))


@torch.no_grad()
def compute_mrr_triple_scoring(model, eval_edge_index, eval_edge_type,
                               fast=False, entities_idx=None, model_type='ClassicLinkPredNet'):

    if entities_idx:
        # inverse relevant_entities_idx
        entities_idx = torch.tensor(entities_idx)
        mask = torch.ones(num_entities, dtype=torch.bool)
        mask[entities_idx] = False
        not_relevant_entities_idx = torch.range(0, num_entities - 1, dtype=torch.int)[~mask]

    ranks = []
    num_samples = eval_edge_type.numel() if not fast else 5000
    for triple_index in tqdm(range(num_samples)):
        (src, dst), rel = eval_edge_index[:, triple_index], eval_edge_type[triple_index]

        # Try all nodes as tails, but delete true triplets:
        tail_mask = torch.ones(num_entities, dtype=torch.bool)
        for (heads, tails), types in [
            (train_edge_index, train_edge_type),
            (val_edge_index, val_edge_type),
            (test_edge_index, test_edge_type),
        ]:
            tail_mask[tails[(heads == src) & (types == rel)]] = False

        if entities_idx:
            tail_mask[not_relevant_entities_idx] = False

        tail = torch.arange(num_entities)[tail_mask]
        tail = torch.cat([torch.tensor([dst]), tail])
        head = torch.full_like(tail, fill_value=src)
        eval_edge_typ_tensor = torch.full_like(tail, fill_value=rel).to(DEVICE)

        h = x_entity[head]
        r = x_relation[eval_edge_typ_tensor]
        t = x_entity[tail]
        if model_type in ['ClassicLinkPredNet', 'DistMultNet']:
            out = model.forward(h, r, t)
        if model_type == 'ComplExNet':
            out = model.forward(h, r, t, head.to(DEVICE), eval_edge_typ_tensor.to(DEVICE), tail.to(DEVICE))


        rank = compute_rank(out)
        ranks.append(rank)

        # Try all nodes as heads, but delete true triplets:
        head_mask = torch.ones(num_entities, dtype=torch.bool)
        for (heads, tails), types in [
            (train_edge_index, train_edge_type),
            (val_edge_index, val_edge_type),
            (test_edge_index, test_edge_type),
        ]:
            head_mask[heads[(tails == dst) & (types == rel)]] = False

        if entities_idx:
            head_mask[not_relevant_entities_idx] = False

        head = torch.arange(num_entities)[head_mask]
        head = torch.cat([torch.tensor([src]), head])
        tail = torch.full_like(head, fill_value=dst)
        eval_edge_typ_tensor = torch.full_like(head, fill_value=rel).to(DEVICE)

        h = x_entity[head]
        r = x_relation[eval_edge_typ_tensor]
        t = x_entity[tail]

        if model_type in ['ClassicLinkPredNet', 'DistMultNet']:
            out = model.forward(h, r, t)
        if model_type == 'ComplExNet':
            out = model.forward(h, r, t, head.to(DEVICE), eval_edge_typ_tensor.to(DEVICE), tail.to(DEVICE))

        rank = compute_rank(out)
        ranks.append(rank)

    num_ranks = len(ranks)
    ranks = torch.tensor(ranks, dtype=torch.float)
    return (1. / ranks).mean(), ranks.mean(), ranks[ranks <= 10].size(0) / num_ranks, \
           ranks[ranks <= 5].size(0) / num_ranks, \
           ranks[ranks <= 3].size(0) / num_ranks, \
           ranks[ranks <= 1].size(0) / num_ranks


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
def compute_mrr_vector_reconstruction(model_tail_pred, model_head_pred, entity_features, relation_features, edge_index,
                                      edge_type, fast=False, entities_idx=None):
    """

    :param model_tail_pred:
    :param model_head_pred:
    :param entity_features:
    :param relation_features:
    :param edge_index:
    :param edge_type:
    :param entities_idx: entities not in the considered graph - only required in the inductive setting
    :param fast:
    :param inductive:
    :return:
    """
    if entities_idx:
        # inverse relevant_entities_idx
        entities_idx = torch.tensor(entities_idx)
        mask = torch.ones(num_entities, dtype=torch.bool)
        mask[entities_idx] = False
        not_relevant_entities_idx = torch.range(0, num_entities - 1, dtype=torch.int)[~mask]

    eval_loss_function = MSELoss(reduction='none')  # CosineEmbeddingLoss(reduction='none')

    num_samples = edge_type.numel() if not fast else 5000
    ranks = []
    for i in tqdm(range(num_samples)):
        (src, dst), rel = edge_index[:, i], edge_type[i]

        out_model = model_tail_pred(torch.reshape(entity_features[src], (1, EMBEDDING_DIM)).float().to(DEVICE),
                                    torch.reshape(relation_features[rel], (1, EMBEDDING_DIM)).float().to(DEVICE))

        # Try all nodes as tails, but delete true triplets:
        tail_mask = torch.ones(num_entities, dtype=torch.bool)
        for (heads, tails), types in [
            (data_train.edge_index, data_train.edge_type),
            (data_val.edge_index, data_val.edge_type),
            (data_test.edge_index, data_test.edge_type),
        ]:
            tail_mask[tails[(heads == src) & (types == rel)]] = False

        if entities_idx:
            tail_mask[not_relevant_entities_idx] = False

        tail = torch.arange(num_entities)[tail_mask]
        tail = torch.cat([torch.tensor([dst]), tail])

        tail_features = entity_features[tail].to(DEVICE)

        out = out_model.repeat(len(tail), 1)

        out = eval_loss_function(out, tail_features)  # , -torch.ones(len(tail)).to(DEVICE))
        out = - torch.mean(out, dim=1)  # added for MSE loss
        rank = compute_rank(out)
        ranks.append(rank)

        out_model = model_head_pred(torch.reshape(entity_features[dst], (1, EMBEDDING_DIM)).float().to(DEVICE),
                                    torch.reshape(relation_features[rel], (1, EMBEDDING_DIM)).float().to(DEVICE))

        # Try all nodes as heads, but delete true triplets:
        head_mask = torch.ones(num_entities, dtype=torch.bool)
        for (heads, tails), types in [
            (data_train.edge_index, data_train.edge_type),
            (data_val.edge_index, data_val.edge_type),
            (data_test.edge_index, data_test.edge_type),
        ]:
            head_mask[heads[(tails == dst) & (types == rel)]] = False

        if entities_idx:
            head_mask[not_relevant_entities_idx] = False

        head = torch.arange(num_entities)[head_mask]
        head = torch.cat([torch.tensor([src]), head])
        head_features = entity_features[head].to(DEVICE)

        out = out_model.repeat(len(head), 1)

        out = eval_loss_function(out, head_features)  # , -torch.ones(len(head)).to(DEVICE))
        out = - torch.mean(out, dim=1)  # added for MSE loss

        rank = compute_rank(out)
        ranks.append(rank)

    num_ranks = len(ranks)
    ranks = torch.tensor(ranks, dtype=torch.float)
    return (1. / ranks).mean(), ranks.mean(), ranks[ranks <= 10].size(0) / num_ranks, \
           ranks[ranks <= 5].size(0) / num_ranks, \
           ranks[ranks <= 3].size(0) / num_ranks, \
           ranks[ranks <= 1].size(0) / num_ranks


def get_ilpc_entities_relations():
    # load data
    entities_ilpc = set()
    relations_ilpc = set()
    entities_ilpc_train = set()
    entitites_ilpc_inference = set()

    for graph in ['train', 'inference']:
        with open(f'./data/ilpc/raw/large/{graph}.txt') as triples_in:
            for line in triples_in:
                head, relation, tail = line[:-1].split('\t')
                entities_ilpc.add(head)
                entities_ilpc.add(tail)
                relations_ilpc.add(relation)
                if graph == 'train':
                    entities_ilpc_train.add(head)
                    entities_ilpc_train.add(tail)
                else:
                    entitites_ilpc_inference.add(head)
                    entitites_ilpc_inference.add(tail)

    return list(entities_ilpc), list(relations_ilpc), list(entities_ilpc_train), list(entitites_ilpc_inference)


# todo implement for ray
def train(config):
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='rdf2vec link prediction')
    # specify dataset
    parser.add_argument('--dataset', type=str, default='fb15k', help='fb15k or fb15-237 or ilpc')
    parser.add_argument('--architecture', type=str, default='ClassicLinkPredNet',
                        help="ClassicLinkPredNet or VectorReconstructionNet or DistMultNet or ComplExNet")
    parser.add_argument('--relationfeatures', type=str, default='standard',
                        help="standard (use the ones trained by RDF2Vec) or derived (derive them form the entity features automatically)")
    parser.add_argument('--lr', type=float, default=.001, help="learning rate")
    parser.add_argument('--bs', type=int, default=1000, help="learning rate")
    parser.add_argument('--hidden', type=int, default=200, help="hidden dim")
    parser.add_argument('--wv', type=str, default='ilpc_rebel',
                        help="only for ilpc: ilpc_rebel or ilpc_joint2vec or ilpc_hybrid2vec")

    parser.add_argument('--device', type=str, default='cpu',
                        help="cpu, cuda, cuda:0, cuda:1")

    args = parser.parse_args()
    dataset = args.dataset
    inductive = dataset == 'ilpc'
    architecture = args.architecture
    relation_embeddings = True if args.relationfeatures == 'standard' else False
    lr = args.lr
    BATCH_SIZE = args.bs  # 1000
    HIDDEN_DIM = args.hidden
    run_name = 'rdf2vec_link_pred' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    DEVICE = torch.device(args.device)

    # only required for inductive link prediction
    entities_train_idx = None
    entities_inference_idx = None

    if dataset in ['fb15k', 'fb15k-237']:
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

        num_entities = len(entities)

        print('num_entities:', num_entities)

        # load RDF2Vec models for features
        wv_model = Word2Vec.load(f'./data/{dataset}_rdf2vec/model')

        if not os.path.isfile(f'./data/{dataset}/train.pt'):
            data_train = read_lp_data(path=f'./data/{dataset}/', entities=entities, relations=relations,
                                      data_sample='train', wv_model=wv_model,
                                      relation_embeddings=relation_embeddings)
            torch.save(data_train, f'./data/{dataset}/train.pt')

        if not os.path.isfile(f'./data/{dataset}/val.pt'):
            data_val = read_lp_data(path=f'./data/{dataset}/', entities=entities, relations=relations,
                                    data_sample='valid',
                                    wv_model=wv_model, relation_embeddings=relation_embeddings)
            torch.save(data_val, f'./data/{dataset}/val.pt')

        if not os.path.isfile(f'./data/{dataset}/test.pt'):
            data_test = read_lp_data(path=f'./data/{dataset}/', entities=entities, relations=relations,
                                     data_sample='test',
                                     wv_model=wv_model, relation_embeddings=relation_embeddings)
            torch.save(data_test, f'./data/{dataset}/test.pt')

        data_train = torch.load(f'./data/{dataset}/train.pt')
        data_val = torch.load(f'./data/{dataset}/val.pt')
        data_test = torch.load(f'./data/{dataset}/test.pt')

    if dataset == 'ilpc':
        wv_type = args.wv

        # start loading data
        entities_ilpc, relations_ilpc, entities_ilpc_train, entitites_ilpc_inference = get_ilpc_entities_relations()
        num_entities = len(entities_ilpc)

        entities_train_idx = [entities_ilpc.index(uri) for uri in entities_ilpc_train]
        entities_inference_idx = [entities_ilpc.index(uri) for uri in entitites_ilpc_inference]
        print('num_entities (train/inf):', num_entities,
              f'({len(entities_train_idx)}/{len(entities_inference_idx)})')

        # load RDF2Vec models for features
        print('start loading data')
        if not os.path.isfile(f'./data/{wv_type}/train.pt') or \
                not os.path.isfile(f'./data/{wv_type}/val.pt') or \
                not os.path.isfile(f'./data/{wv_type}/test.pt'):
            wv_model = Word2Vec.load(f'./data/{wv_type}/model')

        if not os.path.isfile(f'./data/{wv_type}/train.pt'):
            data_train = read_lp_data(path=f'./data/ilpc/raw/large', entities=entities_ilpc, relations=relations_ilpc,
                                      data_sample='train', wv_model=wv_model, relation_embeddings=False)
            torch.save(data_train, f'./data/{wv_type}/train.pt')

        if not os.path.isfile(f'./data/{wv_type}/val.pt'):
            data_val = read_lp_data(path=f'./data/ilpc/raw/large', entities=entities_ilpc, relations=relations_ilpc,
                                    data_sample='inference_validation', wv_model=wv_model, relation_embeddings=False)
            torch.save(data_val, f'./data/{wv_type}/val.pt')

        if not os.path.isfile(f'./data/{wv_type}/test.pt'):
            data_test = read_lp_data(path=f'./data/ilpc/raw/large', entities=entities_ilpc, relations=relations_ilpc,
                                     data_sample='inference_test', wv_model=wv_model, relation_embeddings=False)
            torch.save(data_test, f'./data/{wv_type}/test.pt')

        data_train = torch.load(f'./data/{wv_type}/train.pt')
        data_val = torch.load(f'./data/{wv_type}/val.pt')
        data_test = torch.load(f'./data/{wv_type}/test.pt')

    print('data loaded')

    # load features to GPU
    x_entity = data_train.x_entity.float().to(DEVICE)
    x_relation = data_train.x_relation.float().to(DEVICE)

    # load train data to GPU s.t. this does not have to happen every epoch / every evaluation
    train_edge_index_t = data_train.edge_index.t().to(DEVICE)
    train_edge_index = data_train.edge_index.to(DEVICE)
    train_edge_type = data_train.edge_type.to(DEVICE)

    val_edge_index = data_val.edge_index.to(DEVICE)
    val_edge_type = data_val.edge_type.to(DEVICE)

    test_edge_index = data_test.edge_index.to(DEVICE)
    test_edge_type = data_test.edge_type.to(DEVICE)

    # ClassicLinkPredNet or DistMultNet
    if architecture in ['ClassicLinkPredNet', 'DistMultNet', 'ComplExNet']:
        # todo implement ray hyperparameter optimization
        search_space = {
            "lr": tune.loguniform(1e-5, 1e-3),
            "bs": tune.grid_search([32, 64, 128]),  # batch size
            "rel_dim": tune.grid_search([50, 100]),
            "hid_dim": tune.grid_search([50, 100, 200, 300, 400]),
            "num_hid_lay": tune.grid_search([1, 2, 3, 4]),
            "wandb": {"project": run_name},
        }


        if architecture == 'ClassicLinkPredNet':
            model = ClassicLinkPredNet(EMBEDDING_DIM, HIDDEN_DIM)
        if architecture == 'DistMultNet':
            model = DistMultNet(EMBEDDING_DIM, HIDDEN_DIM)
        if architecture == 'ComplExNet':
            model = ComplExNet(EMBEDDING_DIM, HIDDEN_DIM, num_entities, len(relations))
        model.to(DEVICE)
        loss_function = torch.nn.BCELoss()  # torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # , weight_decay=1e-4
        model.train()

        for epoch in range(0, 1000):
            train_triple_scoring(model, optimizer, model_type=architecture)
            if epoch % 50 == 0:
                mrr, mr, hits10, hits5, hits3, hits1 = compute_mrr_triple_scoring(model,
                                                                                  val_edge_index,
                                                                                  val_edge_type,
                                                                                  fast=True,
                                                                                  entities_idx=entities_inference_idx,
                                                                                  model_type=architecture)
                print('mrr:', mrr, 'mr:', mr, 'hits@10:', hits10, 'hits@5:', hits5, 'hits@3:', hits3, 'hits@1:', hits1)

        torch.save(model.state_dict(), 'model_ClassicLinkPredNet.pt')
        mrr, mr, hits10, hits5, hits3, hits1 = compute_mrr_triple_scoring(model,
                                                                          val_edge_index,
                                                                          val_edge_type,
                                                                          fast=False,
                                                                          entities_idx=entities_inference_idx)
        print('val mrr:', mrr, 'mr:', mr, 'hits@10:', hits10, 'hits@5:', hits5, 'hits@3:', hits3, 'hits@1:', hits1)
        mrr, mr, hits10, hits5, hits3, hits1 = compute_mrr_triple_scoring(model,
                                                                          data_test.edge_index,
                                                                          data_test.edge_type,
                                                                          fast=False,
                                                                          entities_idx=entities_inference_idx)
        print('test mrr:', mrr, 'mr:', mr, 'hits@10:', hits10, 'hits@5:', hits5, 'hits@3:', hits3, 'hits@1:', hits1)

    # VectorReconstructionNet
    elif architecture == 'VectorReconstructionNet':
        torch.backends.cudnn.benchmark = True
        model_tail_pred = VectorReconstructionNet(EMBEDDING_DIM, HIDDEN_DIM)
        model_head_pred = VectorReconstructionNet(EMBEDDING_DIM, HIDDEN_DIM)

        model_tail_pred.to(DEVICE)
        model_head_pred.to(DEVICE)
        optimizer_tail_pred = torch.optim.Adam(model_tail_pred.parameters(), lr=lr)
        optimizer_head_pred = torch.optim.Adam(model_head_pred.parameters(), lr=lr)

        loss_function = MSELoss()  # CosineEmbeddingLoss()

        for epoch in range(1, 5000):
            train_vector_reconstruction(model_tail_pred, optimizer_tail_pred, 'head')
            train_vector_reconstruction(model_head_pred, optimizer_head_pred, 'tail')
            if epoch % 50 == 0:
                mrr, mr, hits10, hits5, hits3, hits1 = compute_mrr_vector_reconstruction(model_tail_pred,
                                                                                         model_head_pred,
                                                                                         x_entity,
                                                                                         x_relation,
                                                                                         data_val.edge_index,
                                                                                         data_val.edge_type,
                                                                                         fast=True,
                                                                                         entities_idx=entities_inference_idx)
                print('mrr:', mrr, 'mr:', mr, 'hits@10:', hits10, 'hits@5:', hits5, 'hits@3:', hits3, 'hits@1:', hits1)

        mrr, mr, hits10, hits5, hits3, hits1 = compute_mrr_vector_reconstruction(model_tail_pred,
                                                                                 model_head_pred,
                                                                                 x_entity,
                                                                                 x_relation,
                                                                                 data_val.edge_index,
                                                                                 data_val.edge_type,
                                                                                 fast=False,
                                                                                 entities_idx=entities_inference_idx)
        print('val mrr:', mrr, 'mr:', mr, 'hits@10:', hits10, 'hits@5:', hits5, 'hits@3:', hits3, 'hits@1:', hits1)
        mrr, mr, hits10, hits5, hits3, hits1 = compute_mrr_vector_reconstruction(model_tail_pred,
                                                                                 model_head_pred,
                                                                                 x_entity,
                                                                                 x_relation,
                                                                                 data_test.edge_index,
                                                                                 data_test.edge_type,
                                                                                 fast=False,
                                                                                 entities_idx=entities_inference_idx)
        print('test mrr:', mrr, 'mr:', mr, 'hits@10:', hits10, 'hits@5:', hits5, 'hits@3:', hits3, 'hits@1:', hits1)
        torch.save(model_tail_pred.state_dict(), 'model_VectorReconstructionNet_tail_pred.pt')
        torch.save(model_head_pred.state_dict(), 'model_VectorReconstructionNet_head_pred.pt')
