# Evaluation of Link Prediction Neural Networks that use RDF2Vec features

## Set-up 
We recommend Python 3.8 using conda. To install all requirements run `pip install -r requirements.txt` (dependencies) 
and `./setup.sh` (datasets).

## Train RDF2Vec embeddings
First the datasets have to be reformatted to RDF n-triple format. Therefore, just call the script `python reformat.py`. 
To train the RDF2Vec embeddings, we use the jRDF2Vec framework. Just call the following commands: 
* `java -Xmx10g -jar jrdf2vec-1.3-SNAPSHOT.jar -graph ./data/fb15k/train_rdf.nt -numberOfWalks 2000 -depth 4 -walkGenerationMode "MID_WALKS" -walkDirectory ./data/fb15k_rdf2vec -dimension 200 -epochs 25 -window 5`
* `java -Xmx10g -jar jrdf2vec-1.3-SNAPSHOT.jar -graph ./data/fb15k-237/train_rdf.nt -numberOfWalks 2000 -depth 4 -walkGenerationMode "MID_WALKS" -walkDirectory ./data/fb15k-237_rdf2vec -dimension 200 -epochs 25 -window 5`


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




### Run 
* VectorReconstructionNet `python main.py --dataset fb15k --architecture VectorReconstructionNet --relationfeatures derived`
* ClassicLinkPredNet `python main.py --dataset fb15k --architecture ClassicLinkPredNet --relationfeatures derived`


nohup python main.py --dataset fb15k --architecture VectorReconstructionNet --relationfeatures derived > ./data/fb15k_vr_der.out &
nohup python main.py --dataset fb15k --architecture VectorReconstructionNet --relationfeatures standard > ./data/fb15k_vr_std.out &
nohup python main.py --dataset fb15k --architecture ClassicLinkPredNet --relationfeatures derived > ./data/fb15k_clp_der.out &
nohup python main.py --dataset fb15k --architecture ClassicLinkPredNet --relationfeatures standard > ./data/fb15k_clp_std.out &
