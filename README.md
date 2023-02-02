# Evaluation of Link Prediction Neural Networks that use RDF2Vec features

## Set-up 
We recommend Python 3.8 using conda. To install all requirements run `pip install -r requirements.txt` (dependencies) 
and `./setup.sh` (datasets).

## Train RDF2Vec embeddings
First the datasets have to be reformatted to RDF n-triple format. Therefore, just call the script `python reformat.py`. 
To train the RDF2Vec embeddings, we use the jRDF2Vec framework. Just call the following commands: 
* `java -Xmx10g -jar jrdf2vec-1.3-SNAPSHOT.jar -graph ./data/fb15k/train_rdf.nt -numberOfWalks 2000 -depth 4 -walkGenerationMode "MID_WALKS" -walkDirectory ./data/fb15k_rdf2vec -dimension 200 -epochs 25 -window 5`
* `java -Xmx10g -jar jrdf2vec-1.3-SNAPSHOT.jar -graph ./data/fb15k-237/train_rdf.nt -numberOfWalks 2000 -depth 4 -walkGenerationMode "MID_WALKS" -walkDirectory ./data/fb15k-237_rdf2vec -dimension 200 -epochs 25 -window 5`
