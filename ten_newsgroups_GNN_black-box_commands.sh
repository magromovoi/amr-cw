# Clean ten newsgroups raw texts csv file

#rm -f data/raw_data/csvs/ten_newsgroups_raw_texts.csv


# Clean ten newsgroups graphs

#rm -f data/datasets/ten_newsgroups_graphs_dataset/processed/*


# Clean ten newsgroups model checkpoints and node embeddings

#rm -f checkpoints/*
#rm -f word2vec-google-news-300.*
#rm -f vertex_embeddings.pkl


# Clean images folder

#rm -f images/*


#Create ten newsgroups raw texts csv file

#python main.py --dataset ten_newsgroups --operation text_index_construction

# Train ten newsgroups text classification

#python main.py --dataset ten_newsgroups --operation text_classification --mode train

# Predict ten newsgroups text classification

#python main.py --dataset ten_newsgroups --operation text_classification --mode predict

# Extract ten newsgroups document embeddings

#python main.py --dataset ten_newsgroups --operation text_classification --mode embeddings

# Train ten newsgroups graph classification

#python main.py --dataset ten_newsgroups --operation graph_classification --mode train

# Predict ten newsgroups graph classification

#python main.py --dataset ten_newsgroups --operation graph_classification --mode predict

# Extract ten newsgroups graph embeddings

#python main.py --dataset ten_newsgroups --operation graph_classification --mode embeddings

# Extract ten newsgroups hybrid data embeddings

#python main.py --dataset ten_newsgroups --operation hybrid_data_classification --mode embeddings

# Train ten newsgroups hybrid data classification

#python main.py --dataset ten_newsgroups --operation hybrid_data_classification --mode train

# Predict ten newsgroups hybrid data classification

#python main.py --dataset ten_newsgroups --operation hybrid_data_classification --mode predict
