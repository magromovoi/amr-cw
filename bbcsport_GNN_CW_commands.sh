# Clean bbcsport weighted frequent subgraphs

: '
rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_frequent_subgraphs/athletics/raw/*
rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_frequent_subgraphs/athletics/processed/*

rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_frequent_subgraphs/cricket/raw/*
rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_frequent_subgraphs/cricket/processed/*

rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_frequent_subgraphs/football/raw/*
rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_frequent_subgraphs/football/processed/*

rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_frequent_subgraphs/rugby/raw/*
rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_frequent_subgraphs/rugby/processed/*

rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_frequent_subgraphs/tennis/raw/*
rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_frequent_subgraphs/tennis/processed/*
'


# Clean bbcsport weighted pattern concepts

: '
rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_pattern_concepts/athletics/raw/*
rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_pattern_concepts/athletics/processed/*

rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_pattern_concepts/cricket/raw/*
rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_pattern_concepts/cricket/processed/*

rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_pattern_concepts/football/raw/*
rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_pattern_concepts/football/processed/*

rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_pattern_concepts/rugby/raw/*
rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_pattern_concepts/rugby/processed/*

rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_pattern_concepts/tennis/raw/*
rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_pattern_concepts/tennis/processed/*
'

# Clean bbcsport weighted filtered equivalence classes

: '
rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_filtered_equivalence_classes/athletics/raw/*
rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_filtered_equivalence_classes/athletics/processed/*

rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_filtered_equivalence_classes/cricket/raw/*
rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_filtered_equivalence_classes/cricket/processed/*

rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_filtered_equivalence_classes/football/raw/*
rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_filtered_equivalence_classes/football/processed/*

rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_filtered_equivalence_classes/rugby/raw/*
rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_filtered_equivalence_classes/rugby/processed/*

rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_filtered_equivalence_classes/tennis/raw/*
rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_filtered_equivalence_classes/tennis/processed/*
'

# Clean bbcsport weighted closed subgraphs

: '
rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_closed_subgraphs/athletics/raw/*
rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_closed_subgraphs/athletics/processed/*

rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_closed_subgraphs/cricket/raw/*
rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_closed_subgraphs/cricket/processed/*

rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_closed_subgraphs/football/raw/*
rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_closed_subgraphs/football/processed/*

rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_closed_subgraphs/rugby/raw/*
rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_closed_subgraphs/rugby/processed/*

rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_closed_subgraphs/tennis/raw/*
rm -f data/datasets/bbcsport_graph_concepts_dataset_weighted_closed_subgraphs/tennis/processed/*
'

# Create bbcsport weighted frequent subgraphs

#python main.py --dataset bbcsport --operation graph_concept_construction --concept_type weighted_frequent_subgraphs

# Create bbcsport weighted pattern concepts

#python main.py --dataset bbcsport --operation graph_concept_construction --concept_type weighted_pattern_concepts

# Create bbcsport weighted filtered equivalence classes

#python main.py --dataset bbcsport --operation graph_concept_construction --concept_type weighted_filtered_equivalence_classes

# Create bbcsport weighted closed subgraphs

#python main.py --dataset bbcsport --operation graph_concept_construction --concept_type weighted_closed_subgraphs


# Clean bbcsport CW model checkpoints

#find checkpoints/ -type f -name "*bbcsport*" -name "*whitened_graph_model*" -name "*gcn_conv*" -exec rm {} +
#find checkpoints/ -type f -name "*bbcsport*" -name "*whitened_graph_model*" -name "*attention_conv*" -exec rm {} +
#find checkpoints/ -type f -name "*bbcsport*" -name "*whitened_graph_model*" -name "*gated_graph_conv*" -exec rm {} +
#find checkpoints/ -type f -name "*bbcsport*" -name "*whitened_graph_model*" -name "*sage_conv*" -exec rm {} +


# Train bbcsport concept whitening using weighted frequent subgraphs

#python main.py --dataset bbcsport --operation graph_concept_whitening --mode train --concept_type weighted_frequent_subgraphs

# Train bbcsport concept whitening using weighted pattern concepts

#python main.py --dataset bbcsport --operation graph_concept_whitening --mode train --concept_type weighted_pattern_concepts

# Train bbcsport concept whitening using weighted filtered equivalence classes

#python main.py --dataset bbcsport --operation graph_concept_whitening --mode train --concept_type weighted_filtered_equivalence_classes

# Train bbcsport concept whitening using weighted closed subgraphs

#python main.py --dataset bbcsport --operation graph_concept_whitening --mode train --concept_type weighted_closed_subgraphs


# Predict bbcsport concept whitening using weighted frequent subgraphs

#python main.py --dataset bbcsport --operation graph_concept_whitening --mode predict --concept_type weighted_frequent_subgraphs

# Predict bbcsport concept whitening using weighted pattern concepts

#python main.py --dataset bbcsport --operation graph_concept_whitening --mode predict --concept_type weighted_pattern_concepts

# Predict bbcsport concept whitening using weighted filtered equivalence classes

#python main.py --dataset bbcsport --operation graph_concept_whitening --mode predict --concept_type weighted_filtered_equivalence_classes

# Predict bbcsport concept whitening using weighted closed subgraphs

#python main.py --dataset bbcsport --operation graph_concept_whitening --mode predict --concept_type weighted_closed_subgraphs


# Visualize bbcsport concept gradient importance

#python main.py --dataset bbcsport --operation evaluation --mode concept_gradient_importance


# Visualize bbcsport concept similarities

#python main.py --dataset bbcsport --operation evaluation --mode concept_dot_product


# Calculate bbcsport macro-averaged F1 CAP

python main.py --dataset bbcsport --operation evaluation --mode concept_axis_visualization

# Extract bbcsport top-activating graph concepts

python main.py --dataset bbcsport --operation evaluation --mode top_activation_subgraphs
