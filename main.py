import yaml
from pathlib import Path
from text_classification import classify_texts
from graph_classification import classify_graphs
from hybrid_data_classification import classify_hybrid_data
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from text_index_construction import construct_text_indices
from graphs_concepts_construction import construct_graph_concepts
from graph_concept_whitening import whiten_graph_concepts


if __name__ == '__main__':

    parser = ArgumentParser(description='Main script', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', type=Path, default='config.yaml', help='configuration file path.')

    parser.add_argument('--dataset', type=str, default='ten_newsgroups', choices=['ten_newsgroups', 'bbcsport'],
                        help='dataset to process.')

    parser.add_argument('--operation', type=str, default='graph_concept_whitening',
                        choices=['text_index_construction', 'text_classification', 'graph_concept_construction',
                                 'graph_classification', 'hybrid_data_classification', 'graph_concept_whitening',
                                 'evaluation'],
                        help='operation to be performed.')

    parser.add_argument('--mode', type=str, default='concept_gradient_importance',
                        choices=['train', 'predict', 'embeddings', 'concept_gradient_importance', 'concept_dot_product'],
                        help='mode of the operation to be performed.')

    parser.add_argument('--concept_type', type=str, default='weighted_frequent_subgraphs',
                        choices=['weighted_frequent_subgraphs', 'weighted_concepts',
                                 'weighted_closed_subgraphs', 'weighted_equivalence_classes'],
                        help='choice of the concept type for the graph concept construction '
                             'and graph concept whitening operations.')

    args, unknown = parser.parse_known_args()

    with args.config.open() as y:
        config = yaml.load(y, Loader=yaml.FullLoader)

    operation = args.operation
    mode = args.mode
    concept_type = args.concept_type

    raw_texts_csv_prefix = config['raw_texts_csv_prefix']

    embeddings_prefix = config['embeddings_prefix']

    concept_types = ['weighted_frequent_subgraphs', 'weighted_concepts', 'weighted_equivalence_classes',
                     'weighted_closed_subgraphs']

    graph_conv_type = '_' + config['graph_conv_type']

    if config['graph_residual_connections']:
        graph_residual_connections = '_residual'
    else:
        graph_residual_connections = '_non_residual'

    if args.dataset == 'ten_newsgroups':
        dataset = 'ten_newsgroups'
        classes = config['ten_newsgroups_classes']

        raw_texts_prefix = config['ten_newsgroups_raw_texts_prefix']

        text_model_name = dataset + '_BERT_' + '_text_model.pt'
        text_model_path = config['checkpoints_prefix'] + '/' + text_model_name

        graphs_dataset_prefix = config['ten_newsgroups_graphs_dataset_prefix']

        graph_model_name = dataset + graph_conv_type + graph_residual_connections + '_graph_model.pt'
        graph_model_path = config['checkpoints_prefix'] + '/' + graph_model_name

        hybrid_model_name = dataset + '_BERT_text_' + graph_conv_type + graph_residual_connections + '_hybrid_model.pt'
        hybrid_model_path = config['checkpoints_prefix'] + '/' + hybrid_model_name

        graph_concepts_dataset_prefix = config['ten_newsgroups_graph_concepts_dataset_prefix']
        concepts = config['ten_newsgroups_graph_concepts']
        concepts_name = '_'.join(concepts)
        concepts_name = '_' + concepts_name

        whitened_graph_model_paths = {}
        whitened_graph_model_name = (dataset + concepts_name + graph_conv_type + graph_residual_connections +
                                     '_' + concept_type + '_whitened_graph_model.pt')

        whitened_graph_model_path = config['checkpoints_prefix'] + '/' + whitened_graph_model_name
        whitened_graph_model_paths[concept_type] = whitened_graph_model_path

        negative_concept_types = [concept_type_temp for concept_type_temp in concept_types
                                  if concept_type_temp != concept_type]

        for negative_concept_type in negative_concept_types:
            negative_whitened_graph_model_name = (dataset + concepts_name + graph_conv_type +
                                                  graph_residual_connections + '_' + negative_concept_type
                                                  + '_whitened_graph_model.pt')

            negative_whitened_graph_model_path = config['checkpoints_prefix'] + '/' + negative_whitened_graph_model_name
            whitened_graph_model_paths[negative_concept_type] = negative_whitened_graph_model_path

        target_class = config['ten_newsgroups_target_class']

    if operation == 'text_index_construction':
        construct_text_indices(dataset, classes, raw_texts_prefix, raw_texts_csv_prefix)

    elif operation == 'text_classification':
        classify_texts(dataset, classes, raw_texts_csv_prefix, embeddings_prefix, text_model_path, mode)

    elif operation == 'graph_concept_construction':
        construct_graph_concepts(graph_concepts_dataset_prefix, dataset, classes, concept_type,
                                 config['min_support'], config['max_penalty'])

    elif operation == 'graph_classification':
        classify_graphs(dataset, classes, graphs_dataset_prefix, embeddings_prefix,
                        graph_model_path, config['graph_conv_type'], config['graph_residual_connections'], mode)

    elif operation == 'hybrid_data_classification':
        classify_hybrid_data(dataset, classes, embeddings_prefix, hybrid_model_path, config['graph_conv_type'],
                             config['graph_residual_connections'], mode)

    elif operation == 'graph_concept_whitening':
        whiten_graph_concepts(dataset, classes, graphs_dataset_prefix, graph_concepts_dataset_prefix,
                              concepts, graph_model_path, config['graph_conv_type'],
                              config['graph_residual_connections'], whitened_graph_model_paths,
                              concept_type, negative_concept_types, target_class, mode)

    elif operation == 'evaluation':
        whiten_graph_concepts(dataset, classes, graphs_dataset_prefix, graph_concepts_dataset_prefix,
                              concepts, graph_model_path, config['graph_conv_type'],
                              config['graph_residual_connections'], whitened_graph_model_paths,
                              concept_type, negative_concept_types, target_class, mode)
