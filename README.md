# Evaluation of Link Prediction Neural Networks that use RDF2Vec features

## Set-up 
We recommend Python 3.8 using conda. To install all requirements run `pip install -r requirements.txt` (dependencies) 
and `./setup.sh` (datasets).

## Train RDF2Vec embeddings
First the datasets have to be reformatted to RDF n-triple format. Therefore, just call the script `python reformat.py`. 
To train the RDF2Vec embeddings, we use the jRDF2Vec framework. Just call the following commands: 
* `java -Xmx10g -jar jrdf2vec-1.3-SNAPSHOT.jar -graph ./data/fb15k/train_rdf.nt -numberOfWalks 2000 -depth 4 -walkGenerationMode "MID_WALKS" -walkDirectory ./data/fb15k_rdf2vec -dimension 200 -epochs 25 -window 5`
* `java -Xmx10g -jar jrdf2vec-1.3-SNAPSHOT.jar -graph ./data/fb15k-237/train_rdf.nt -numberOfWalks 2000 -depth 4 -walkGenerationMode "MID_WALKS" -walkDirectory ./data/fb15k-237_rdf2vec -dimension 200 -epochs 25 -window 5`

To train the inductive link prediction (ILPC) embeddings, run the following commands:

* `mkdir ./data/ilpc-large_rebel`
* `mkdir ./data/ilpc-large_joint2vec`
* `mkdir ./data/ilpc-large_hybrid2vec`
* `python reformat.py`
* `java -Xmx200g -jar jrdf2vec-1.2-SNAPSHOT.jar -onlyTraining -walkDirectory ./data/ilpc_rebel -dimension 200 -epochs 25 -window 5`
* `cp ./data/ilpc_rebel/ilpc_rebel.txt.gz ./data/ilpc_joint2vec`
* `java -Xmx200g -jar jrdf2vec-1.3-SNAPSHOT.jar -graph ./data/ilpc/raw/large/train_rdf.nt -numberOfWalks 2000 -depth 4 -walkGenerationMode "MID_WALKS" -walkDirectory ./data/ilpc_rdf2vec -dimension 200 -epochs 25 -window 5`
* `cp ./data/ilpc_rdf2vec/walk_*  ./data/ilpc_joint2vec`
* `java -Xmx200g -jar jrdf2vec-1.2-SNAPSHOT.jar -onlyTraining -walkDirectory ./data/ilpc_joint2vec -dimension 200 -epochs 25 -window 5`







## Training

### Models
The **VectorReconstructionNet** model takes as input head and relation features (or tail and relation features) as is 
trained to predict the tail feature (or head feature). The Link Prediction plausibility scoring for the triple is then the distance of 
this predicted vector to the actual feature vector. This architecture was originally proposed by Paulheim et al.

The **ClassicLinkPredNet** model takes as input head, relation, and tail features and is trained to predict the 
Link Prediction plausibility score directly. 


### Relation Features
As relation features, either the standard RDF2Vec embeddings of the relations can be used (**standard**), or the 
relation features can be derived by computing the mean difference between the entities a relation usually 
occurs (**derived**). The **derived** features were initially proposed by Paulheim et al. for a Link Prediction approach 
with RDF2Vec embeddings that does not need a NN. 




## Run 
## FB15k
* `nohup python -u main.py --dataset fb15k --architecture VectorReconstructionNet --relationfeatures derived --bs 1000 --device cuda:0 > ./data/fb15k_vr_der.out &`
* `nohup python -u main.py --dataset fb15k --architecture VectorReconstructionNet --relationfeatures standard --bs 1000 --device cuda:0 > ./data/fb15k_vr_std.out &`
* `nohup python -u main.py --dataset fb15k --architecture ClassicLinkPredNet --relationfeatures derived --bs 128 --device cuda:0 --lr .0001 > ./data/fb15k_clp_der.out &`
* `nohup python -u main.py --dataset fb15k --architecture ClassicLinkPredNet --relationfeatures standard --bs 128 --device cuda:0 > ./data/fb15k_clp_std.out &`
* `nohup python -u main.py --dataset fb15k --architecture DistMultNet --relationfeatures standard --bs 128 --device cuda:1 > ./data/fb15k_dm_std.out &`
* `nohup python -u main.py --dataset fb15k --architecture DistMultNet --relationfeatures derived --bs 128 --device cuda:1  > ./data/fb15k_dm_der.out &`
* `nohup python -u main.py --dataset fb15k --architecture ComplExNet --relationfeatures standard --bs 128 --device cuda:1 > ./data/fb15k_cp_std.out &`
* `nohup python -u main.py --dataset fb15k --architecture ComplExNet --relationfeatures derived --bs 128 --device cuda:0 > ./data/fb15k_cp_der.out &`


* `nohup python -u main.py --dataset fb15k --architecture DistMultNet --relationfeatures standard --bs 128 --device cuda:1 --batchnorm > ./data/fb15k_dm_std_bn.out &`
* `nohup python -u main.py --dataset fb15k --architecture DistMultNet --relationfeatures derived --bs 128 --device cuda:1  --batchnorm > ./data/fb15k_dm_der_bn.out &`
* `nohup python -u main.py --dataset fb15k --architecture ComplExNet --relationfeatures standard --bs 128 --device cuda:0 --batchnorm > ./data/fb15k_cp_std_bn.out &`
* `nohup python -u main.py --dataset fb15k --architecture ComplExNet --relationfeatures derived --bs 128 --device cuda:0 --batchnorm > ./data/fb15k_cp_der_bn.out &`



## FB15k-237
* `nohup python -u main.py --dataset fb15k-237 --architecture VectorReconstructionNet --relationfeatures derived --bs 1000 --device cuda:0 > ./data/fb15k-237_vr_der.out &` -> histogram 
* `nohup python -u main.py --dataset fb15k-237 --architecture VectorReconstructionNet --relationfeatures standard --bs 1000 --device cuda:0 > ./data/fb15k-237_vr_std.out &`
* `nohup python -u main.py --dataset fb15k-237 --architecture ClassicLinkPredNet --relationfeatures derived --bs 128 --device cuda:0 > ./data/fb15k-237_clp_der.out &` -> histogram 
* `nohup python -u main.py --dataset fb15k-237 --architecture ClassicLinkPredNet --relationfeatures standard --bs 128 --device cuda:0 > ./data/fb15k-237_clp_std.out &`

## ILPC 
* `nohup python -u main.py --dataset ilpc --architecture VectorReconstructionNet --bs 1000 --wv ilpc_rebel --device cuda:0 > ./data/ilpc_vr_rebel.out &`
* `nohup python -u main.py --dataset ilpc --architecture VectorReconstructionNet --bs 1000 --wv ilpc_joint2vec --device cuda:0 > ./data/ilpc_vr_joint.out &`
* `nohup python -u main.py --dataset ilpc --architecture ClassicLinkPredNet --bs 1000 --wv ilpc_joint2vec --device cuda:0 > ./data/ilpc_vr_joint.out &`



## Results


### FB15k
| Models                  | Batch Norm. | Edge Features | MR        | MRR    | Hits@10 | Hits@3 | Hits@1 |
|-------------------------|-------------|---------------|-----------|--------|---------|--------|--------|
| VectorReconstructionNet | false       | derived       | 482.4171  | 0.2710 | 0.409   | 0.296  | 0.1971 |
| VectorReconstructionNet | false       | standard      | 481.3459  | 0.2705 | 0.411   | 0.297  | 0.196  |
| ClassicLinkPredNet      | false       | derived       | 147.4021  | 0.2036 | 0.420   | 0.224  | 0.101  |
| ClassicLinkPredNet      | false       | standard      | 160.4563  | 0.1742 | 0.357   | 0.186  | 0.086  |



### FB15k-237
| Models                  | Batch Norm. | Edge Features  | MR        | MRR    | Hits@10 | Hits@3 | Hits@1 |
|-------------------------|-------------|----------------|-----------|--------|---------|--------|--------|
| VectorReconstructionNet | false       | derived        | 1146.7583 | 0.2054 | 0.281   | 0.222  | 0.162  |-> histogram
| VectorReconstructionNet | false       | standard       | 1153.2833 | 0.2083 | 0.286   | 0.226  | 0.165  |
| ClassicLinkPredNet      | false       | derived        | 427.9076  | 0.1228 | 0.258   | 0.127  | 0.057  |-> histogram
| ClassicLinkPredNet      | false       | standard       | 419.6589  | 0.1172 | 0.252   | 0.119  | 0.053  |



### WN18 







