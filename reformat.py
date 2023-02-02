def reformat_dataset(dataset):
    with open(f'./data/{dataset}/train_rdf.nt', 'w') as kg_out:
        for data_sample in ['train']:
            with open(f'./data/{dataset}/{data_sample}.txt') as triples_in:
                for line in triples_in:
                    head, relation, tail = line[:-1].split('\t')
                    triple_line = f'<http://wd/{head}> <http://wd/{relation}> <http://wd/{tail}> . \n'
                    kg_out.write(triple_line)


if __name__ == '__main__':
    reformat_dataset('fb15k')
    reformat_dataset('fb15k-237')
