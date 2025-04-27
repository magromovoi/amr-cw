# Clean bbcsport raw texts csv file

#rm -f data/raw_data/csvs/bbcsport_raw_texts.csv

# Clean bbcsport graphs

#rm -f data/datasets/bbcsport_graphs_dataset/processed/*


# Clean bbcsport model checkpoints and node embeddings

#rm -f checkpoints/*
#rm -f word2vec-google-news-300.*
#rm -f vertex_embeddings.pkl

# Clean images folder

#rm -f images/*


#Create bbcsport raw texts csv file

#python main.py --dataset bbcsport --operation text_index_construction

# Train bbcsport text classification

#python main.py --dataset bbcsport --operation text_classification --mode train

# Predict bbcsport text classification

#python main.py --dataset bbcsport --operation text_classification --mode predict

# Extract bbcsport document embeddings

#python main.py --dataset bbcsport --operation text_classification --mode embeddings

# Train bbcsport graph classification

#python main.py --dataset bbcsport --operation graph_classification --mode train

# Predict bbcsport graph classification

#python main.py --dataset bbcsport --operation graph_classification --mode predict

# Extract bbcsport graph embeddings

#python main.py --dataset bbcsport --operation graph_classification --mode embeddings

# Extract bbcsport hybrid data embeddings

#python main.py --dataset bbcsport --operation hybrid_data_classification --mode embeddings

# Train bbcsport hybrid data classification

#python main.py --dataset bbcsport --operation hybrid_data_classification --mode train

# Predict bbcsport hybrid data classification

#python main.py --dataset bbcsport --operation hybrid_data_classification --mode predict
