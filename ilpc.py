import argparse
import os.path as osp
import torch
import numpy as np
from gensim.models import Word2Vec
from torch.nn import MSELoss
from torch_geometric.data import Data
import os

from main import read_lp_data, train_triple_scoring, compute_mrr_triple_scoring, train_vector_reconstruction, \
    compute_mrr_vector_reconstruction
from models import ClassicLinkPredNet, VectorReconstructionNet

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cuda:1')


def get_ilpc_entities_relations():
    # load data
    entities_ilpc = set()
    relations_ilpc = set()


    for graph in ['train', 'inference']:
        with open(f'./data/ilpc/raw/large/{graph}.txt') as triples_in:
            for line in triples_in:
                head, relation, tail = line[:-1].split('\t')
                entities_ilpc.add(head)
                entities_ilpc.add(tail)
                relations_ilpc.add(relation)

    return list(entities_ilpc), list(relations_ilpc)


if __name__ == '__main__':
    EMBEDDING_DIM = 200

    parser = argparse.ArgumentParser(description='rdf2vec inductive link prediction with background knowledge')
    # specify dataset
    parser.add_argument('--architecture', type=str, default='ClassicLinkPredNet',
                        help="ClassicLinkPredNet or VectorReconstructionNet")
    parser.add_argument('--lr', type=float, default=.001, help="learning rate")
    parser.add_argument('--bs', type=int, default=1000, help="learning rate")
    parser.add_argument('--hidden', type=int, default=200, help="hidden dim")
    parser.add_argument('--wv', type=str, default='ilpc_rebel', help="ilpc_rebel or ilpc_joint2vec or ilpc_hybrid2vec")

    args = parser.parse_args()
    architecture = args.architecture
    lr = args.lr
    BATCH_SIZE = args.bs  # 1000
    HIDDEN_DIM = args.hidden
    wv_type = args.wv

    # start loading data
    entities_ilpc, relations_ilpc = get_ilpc_entities_relations()
    num_entities = len(entities_ilpc)

    # load RDF2Vec models for features

    wv_model = Word2Vec.load(f'./data/ilpc/{wv_type}/model')

    if not os.path.isfile(f'./data/ilpc/{wv_type}/train.pt'):
        data_train = read_lp_data(path=f'./data/ilpc/{wv_type}/', entities=entities_ilpc, relations=relations_ilpc,
                                  data_sample='train', wv_model=wv_model, relation_embeddings=False)
        torch.save(data_train, f'./data/ilpc/{wv_type}/train.pt')

    if not os.path.isfile(f'./data/ilpc/{wv_type}/val.pt'):
        data_val = read_lp_data(path=f'./data/ilpc/{wv_type}/', entities=entities_ilpc, relations=relations_ilpc,
                                data_sample='valid', wv_model=wv_model, relation_embeddings=False)
        torch.save(data_val, f'./data/ilpc/{wv_type}/val.pt')

    if not os.path.isfile(f'./data/ilpc/{wv_type}/test.pt'):
        data_test = read_lp_data(path=f'./data/ilpc/{wv_type}/', entities=entities_ilpc, relations=relations_ilpc,
                                 data_sample='test', wv_model=wv_model, relation_embeddings=False)
        torch.save(data_test, f'./data/ilpc/{wv_type}/test.pt')

    data_train = torch.load(f'./data//ilpc/{wv_type}/train.pt')
    data_val = torch.load(f'./data/ilpc/{wv_type}/val.pt')
    data_test = torch.load(f'./data/ilpc/{wv_type}/test.pt')
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

    # ClassicLinkPredNet
    if architecture == 'ClassicLinkPredNet':
        model = ClassicLinkPredNet(EMBEDDING_DIM, HIDDEN_DIM)
        model.to(DEVICE)
        loss_function = torch.nn.BCELoss() #torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # , weight_decay=1e-4
        model.train()

        for epoch in range(0, 1000):
            train_triple_scoring(model, optimizer)
            if epoch % 50 == 0:
                mrr, mr, hits10, hits5, hits3, hits1 = compute_mrr_triple_scoring(model, num_entities, val_edge_index,
                                                             val_edge_type, fast=True)
                print('mrr:', mrr, 'mr:', mr, 'hits@10:', hits10, 'hits@5:', hits5, 'hits@3:', hits3, 'hits@1:', hits1)

        torch.save(model.state_dict(), 'model_ClassicLinkPredNet.pt')
        mrr, mr, hits10, hits5, hits3, hits1 = compute_mrr_triple_scoring(model, num_entities, val_edge_index, val_edge_type,
                                                     fast=False)
        print('val mrr:', mrr, 'mr:', mr, 'hits@10:', hits10, 'hits@5:', hits5, 'hits@3:', hits3, 'hits@1:', hits1)
        mrr, mr, hits10, hits5, hits3, hits1 = compute_mrr_triple_scoring(model, num_entities, data_test.edge_index, data_test.edge_type,
                                                     fast=False)
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

        for epoch in range(0, 5000):
            train_vector_reconstruction(model_tail_pred, optimizer_tail_pred, 'head')
            train_vector_reconstruction(model_head_pred, optimizer_head_pred, 'tail')
            if epoch % 50 == 0:
                mrr, mr, hits10, hits5, hits3, hits1 = compute_mrr_vector_reconstruction(model_tail_pred, model_head_pred,
                                                                    x_entity, x_relation,
                                                                    data_val.edge_index, data_val.edge_type, fast=True)
                print('mrr:', mrr, 'mr:', mr, 'hits@10:', hits10, 'hits@5:', hits5, 'hits@3:', hits3, 'hits@1:', hits1)

        mrr, mr, hits10, hits5, hits3, hits1 = compute_mrr_vector_reconstruction(model_tail_pred, model_head_pred, x_entity,
                                                            x_relation,
                                                            data_val.edge_index, data_val.edge_type, fast=False)
        print('val mrr:', mrr, 'mr:', mr, 'hits@10:', hits10, 'hits@5:', hits5, 'hits@3:', hits3, 'hits@1:', hits1)
        mrr, mr, hits10, hits5, hits3, hits1 = compute_mrr_vector_reconstruction(model_tail_pred, model_head_pred, x_entity,
                                                            x_relation,
                                                            data_test.edge_index, data_test.edge_type, fast=False)
        print('test mrr:', mrr, 'mr:', mr, 'hits@10:', hits10, 'hits@5:', hits5, 'hits@3:', hits3, 'hits@1:', hits1)
        torch.save(model_tail_pred.state_dict(), 'model_VectorReconstructionNet_tail_pred.pt')
        torch.save(model_head_pred.state_dict(), 'model_VectorReconstructionNet_head_pred.pt')


