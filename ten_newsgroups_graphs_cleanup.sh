# conda activate chemical-concept-whitening

# Clean ten newsgroups raw texts csv file

#rm -f data/raw_data/csvs/ten_newsgroups_raw_texts.csv

# Clean ten newsgroups graphs

rm -f data/datasets/ten_newsgroups_graphs_dataset/processed/*

# Clean ten newsgroups weighted frequent subgraphs

: '
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_frequent_subgraphs/business/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_frequent_subgraphs/business/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_frequent_subgraphs/entertainment/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_frequent_subgraphs/entertainment/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_frequent_subgraphs/food/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_frequent_subgraphs/food/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_frequent_subgraphs/graphics/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_frequent_subgraphs/graphics/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_frequent_subgraphs/historical/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_frequent_subgraphs/historical/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_frequent_subgraphs/medical/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_frequent_subgraphs/medical/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_frequent_subgraphs/politics/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_frequent_subgraphs/politics/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_frequent_subgraphs/space/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_frequent_subgraphs/space/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_frequent_subgraphs/sport/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_frequent_subgraphs/sport/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_frequent_subgraphs/technologie/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_frequent_subgraphs/technologie/processed/*
'

# Clean ten newsgroups weighted concepts

: '
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_concepts/business/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_concepts/business/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_concepts/entertainment/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_concepts/entertainment/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_concepts/food/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_concepts/food/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_concepts/graphics/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_concepts/graphics/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_concepts/historical/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_concepts/historical/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_concepts/medical/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_concepts/medical/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_concepts/politics/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_concepts/politics/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_concepts/space/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_concepts/space/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_concepts/sport/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_concepts/sport/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_concepts/technologie/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_concepts/technologie/processed/*
'


# Clean ten newsgroups weighted equivalence classes

: '
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_equivalence_classes/business/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_equivalence_classes/business/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_equivalence_classes/entertainment/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_equivalence_classes/entertainment/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_equivalence_classes/food/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_equivalence_classes/food/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_equivalence_classes/graphics/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_equivalence_classes/graphics/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_equivalence_classes/historical/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_equivalence_classes/historical/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_equivalence_classes/medical/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_equivalence_classes/medical/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_equivalence_classes/politics/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_equivalence_classes/politics/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_equivalence_classes/space/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_equivalence_classes/space/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_equivalence_classes/sport/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_equivalence_classes/sport/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_equivalence_classes/technologie/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_equivalence_classes/technologie/processed/*
'

# Clean ten newsgroups weighted closed subgraphs

: '
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_closed_subgraphs/business/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_closed_subgraphs/business/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_closed_subgraphs/entertainment/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_closed_subgraphs/entertainment/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_closed_subgraphs/food/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_closed_subgraphs/food/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_closed_subgraphs/graphics/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_closed_subgraphs/graphics/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_closed_subgraphs/historical/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_closed_subgraphs/historical/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_closed_subgraphs/medical/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_closed_subgraphs/medical/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_closed_subgraphs/politics/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_closed_subgraphs/politics/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_closed_subgraphs/space/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_closed_subgraphs/space/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_closed_subgraphs/sport/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_closed_subgraphs/sport/processed/*

rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_closed_subgraphs/technologie/raw/*
rm -f data/datasets/ten_newsgroups_graph_concepts_dataset_weighted_closed_subgraphs/technologie/processed/*
'

# Clean ten newsgroups model checkpoints and node embeddings


#rm -f checkpoints/*
rm -f word2vec-google-news-300.*
rm -f vertex_embeddings.pkl

#Create ten newsgroups raw texts csv file

#python main.py --dataset ten_newsgroups --operation text_index_construction

# Train ten newsgroups text classification

#python main.py --dataset ten_newsgroups --operation text_classification --mode train

# Extract ten newsgroups document embeddings

#python main.py --dataset ten_newsgroups --operation text_classification --mode embeddings

# Create ten newsgroups weighted concepts

#python main.py --dataset ten_newsgroups --operation graph_concept_construction --concept_type weighted_concepts

# Create ten newsgroups weighted equivalence classes

#python main.py --dataset ten_newsgroups --operation graph_concept_construction --concept_type weighted_equivalence_classes

# Create ten newsgroups weighted closed subgraphs

#python main.py --dataset ten_newsgroups --operation graph_concept_construction --concept_type weighted_closed_subgraphs

# Create ten newsgroups weighted closed subgraphs

#python main.py --dataset ten_newsgroups --operation graph_concept_construction --concept_type weighted_closed_subgraphs

# Train ten newsgroups graph classification

#python main.py --dataset ten_newsgroups --operation graph_classification --mode train

# Extract ten newsgroups graph embeddings

#python main.py --dataset ten_newsgroups --operation graph_classification --mode embeddings

# Extract ten newsgroups hybrid data embeddings

#python main.py --dataset ten_newsgroups --operation hybrid_data_classification --mode embeddings

# Train ten newsgroups hybrid data classification

#python main.py --dataset ten_newsgroups --operation hybrid_data_classification --mode train

# Train ten newsgroups concept whitening using weighted frequent subgraphs concepts

#python main.py --dataset ten_newsgroups --operation graph_concept_whitening --mode train --concept_type weighted_frequent_subgraphs

# Train ten newsgroups concept whitening using weighted concepts

#python main.py --dataset ten_newsgroups --operation graph_concept_whitening --mode train --concept_type weighted_concepts

# Train ten newsgroups concept whitening using weighted equivalence classes concepts

#python main.py --dataset ten_newsgroups --operation graph_concept_whitening --mode train --concept_type weighted_equivalence_classes

# Train ten newsgroups concept whitening using weighted closed subgraphs concepts

#python main.py --dataset ten_newsgroups --operation graph_concept_whitening --mode train --concept_type weighted_closed_subgraphs

# Visualize ten newsgroups concept gradient importance

#python main.py --dataset ten_newsgroups --operation evaluation --mode concept_gradient_importance
