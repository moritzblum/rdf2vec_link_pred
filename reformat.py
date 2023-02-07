from tqdm import tqdm
import os
import os.path as osp
import json
import gzip
import shutil


def reformat_lp_dataset(dataset):
    with open(f'./data/{dataset}/train_rdf.nt', 'w') as kg_out:
        with open(f'./data/{dataset}/train.txt') as triples_in:
            for line in triples_in:
                head, relation, tail = line[:-1].split('\t')
                triple_line = f'<http://wd/{head}> <http://wd/{relation}> <http://wd/{tail}> . \n'
                kg_out.write(triple_line)


def reformat_rebel():
    with open('./data/ilpc_rebel/ilpc_rebel.txt', 'w') as sentences_out:

        for file in tqdm(os.listdir('./data/REBEL')):
            if file.startswith('en_'):
                with open(osp.join('./data/REBEL', file)) as rebel_in:
                    for line in rebel_in:
                        rebel_doc = json.loads(line)
                        text = rebel_doc['text'].rstrip().lstrip().replace('\n', ' ')
                        for entity in rebel_doc['entities']:
                            uri = entity['uri']
                            if uri.startswith('Q'):
                                text = text.replace(entity['surfaceform'], ' http://wd/' + uri + ' ')
                        sentences_out.write(text + '\n')

    with open('./data/ilpc_rebel/ilpc_rebel.txt', 'rb') as f_in:
        with gzip.open(osp.join('./data/ilpc_rebel/ilpc_rebel.txt.gz'), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


if __name__ == '__main__':
    reformat_lp_dataset('fb15k')
    reformat_lp_dataset('fb15k-237')
    reformat_lp_dataset('ilpc/raw/large')
    reformat_rebel()