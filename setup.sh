mkdir data
wget https://github.com/dwslab/jRDF2Vec/raw/jars/jars/jrdf2vec-1.3-SNAPSHOT.jar

wget "https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz" -P ./data
tar -xzf ./data/fetch.php?media=en:fb15k.tgz --directory ./data
mv ./data/FB15k ./data/fb15k
mv ./data/fb15k/freebase_mtr100_mte100-test.txt ./data/fb15k/test.txt
mv ./data/fb15k/freebase_mtr100_mte100-train.txt ./data/fb15k/train.txt
mv ./data/fb15k/freebase_mtr100_mte100-valid.txt ./data/fb15k/valid.txt
rm ./data/fetch.php?media=en:fb15k.tgz

mkdir fb15k-237
wget "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237/train.txt" -P ./data/fb15k-237
wget "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237/valid.txt" -P ./data/fb15k-237
wget "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237/test.txt" -P ./data/fb15k-237