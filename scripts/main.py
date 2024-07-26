import yaml
from pathlib import Path
from utils import get_project_root
from text_classification import classify_texts
from graph_classification import classify_graphs
from embeddings_visualization import visualize_embeddings
from text_index_construction import construct_text_indices
from hybrid_data_classification import classify_hybrid_data
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def do_operations(root, config, dataset, operation):

    if dataset == 'all':
        if operation == 'text_indexing':
            print("Indexing all texts...")
            construct_text_indices('ten_newsgroups', config['ten_newsgroups_classes'],
                                   str(root) + '/' + config['ten_newsgroups_raw_texts_prefix'],
                                   str(root) + '/' + config['raw_texts_csv_prefix'])

            construct_text_indices('bbcsport', config['bbcsport_classes'],
                                   str(root) + '/' + config['bbcsport_raw_texts_prefix'],
                                   str(root) + '/' + config['raw_texts_csv_prefix'])

        if operation == 'text_classification':
            print("Classifying all texts...")
            classify_texts('ten_newsgroups', config['ten_newsgroups_classes'],
                           str(root) + '/' + config['raw_texts_csv_prefix'],
                           str(root) + '/' + config['embeddings_prefix'],
                           str(root) + '/' + config['saved_models_prefix'] + '/' + 'ten_newsgroups_text_model.pt',
                           mode='training')

            classify_texts('bbcsport', config['bbcsport_classes'],
                           str(root) + '/' + config['raw_texts_csv_prefix'],
                           str(root) + '/' + config['embeddings_prefix'],
                           str(root) + '/' + config['saved_models_prefix'] + '/' + 'bbcsport_text_model.pt',
                           mode='training')

    if dataset == 'ten_newsgroups':
        if operation == 'text_indexing':
            print("Indexing ten newsgroups texts...")
            construct_text_indices('ten_newsgroups', config['ten_newsgroups_classes'],
                                   str(root) + '/' + config['ten_newsgroups_raw_texts_prefix'],
                                   str(root) + '/' + config['raw_texts_csv_prefix'])

        if operation == 'text_classification':
            print("Classifying ten newsgroups texts...")
            classify_texts('ten_newsgroups', config['ten_newsgroups_classes'],
                           str(root) + '/' + config['raw_texts_csv_prefix'],
                           str(root) + '/' + config['embeddings_prefix'],
                           str(root) + '/' + config['saved_models_prefix'] + '/' + 'ten_newsgroups_text_model.pt',
                           mode='get_text_embeddings')

        if operation == 'graph_classification':
            print("Classifying ten newsgroups graphs...")
            classify_graphs('ten_newsgroups', config['ten_newsgroups_classes'],
                            str(root) + '/' + config['ten_newsgroups_graphs_dataset_prefix'],
                            str(root) + '/' + config['ten_newsgroups_graphs_index_prefix'],
                            str(root) + '/' + config['embeddings_prefix'],
                            str(root) + '/' + config['saved_models_prefix'] + '/' + 'ten_newsgroups_graph_model.pt',
                            mode='inference')

        if operation == 'hybrid_classification':
            print("Classifying ten newsgroups hybrid data...")
            classify_hybrid_data('ten_newsgroups', config['ten_newsgroups_classes'],
                                 str(root) + '/' + config['embeddings_prefix'],
                                 str(root) + '/' + config['saved_models_prefix']
                                 + '/' + 'ten_newsgroups_hybrid_model.pt',
                                 mode='inference')

        if operation == 'visualization':
            print("Visualizing ten newsgroups embeddings...")
            visualize_embeddings('ten_newsgroups', config['ten_newsgroups_classes'],
                                 str(root) + '/' + config['raw_texts_csv_prefix'],
                                 str(root) + '/' + config['ten_newsgroups_graphs_dataset_prefix'],
                                 str(root) + '/' + config['embeddings_prefix'],
                                 str(root) + '/' + config['saved_models_prefix'] + '/' + 'ten_newsgroups_text_model.pt',
                                 str(root) + '/' + config['saved_models_prefix'] + '/' + 'ten_newsgroups_graph_model.pt',
                                 str(root) + '/' + config['saved_models_prefix'] + '/' + 'ten_newsgroups_hybrid_model.pt',
                                 mode='text_embeddings')

    if dataset == 'bbcsport':
        if operation == 'text_indexing':
            print("Indexing BBC sport texts...")
            construct_text_indices('bbcsport', config['bbcsport_classes'],
                                   str(root) + '/' + config['bbcsport_raw_texts_prefix'],
                                   str(root) + '/' + config['raw_texts_csv_prefix'])

        if operation == 'text_classification':
            print("Classifying BBC sport texts...")
            classify_texts('bbcsport', config['bbcsport_classes'],
                           str(root) + '/' + config['raw_texts_csv_prefix'],
                           str(root) + '/' + config['embeddings_prefix'],
                           str(root) + '/' + config['saved_models_prefix'] + '/' + 'bbcsport_text_model.pt',
                           mode='get_text_embeddings')

        if operation == 'graph_classification':
            print("Classifying BBC sport graphs...")
            classify_graphs('bbcsport', config['bbcsport_classes'],
                            str(root) + '/' + config['bbcsport_graphs_dataset_prefix'],
                            str(root) + '/' + config['bbcsport_graphs_index_prefix'],
                            str(root) + '/' + config['embeddings_prefix'],
                            str(root) + '/' + config['saved_models_prefix'] + '/' + 'bbcsport_graph_model.pt',
                            mode='inference')

        if operation == 'hybrid_classification':
            print("Classifying BBC sport hybrid data...")
            classify_hybrid_data('bbcsport', config['bbcsport_classes'],
                                 str(root) + '/' + config['embeddings_prefix'],
                                 str(root) + '/' + config['saved_models_prefix']
                                 + '/' + 'bbcsport_hybrid_model.pt',
                                 mode='training')

        if operation == 'visualization':
            print("Visualizing BBC sport embeddings...")
            visualize_embeddings('bbcsport', config['bbcsport_classes'],
                                 str(root) + '/' + config['raw_texts_csv_prefix'],
                                 str(root) + '/' + config['bbcsport_graphs_dataset_prefix'],
                                 str(root) + '/' + config['embeddings_prefix'],
                                 str(root) + '/' + config['saved_models_prefix'] + '/' + 'bbcsport_text_model.pt',
                                 str(root) + '/' + config['saved_models_prefix'] + '/' + 'bbcsport_graph_model.pt',
                                 str(root) + '/' + config['saved_models_prefix'] + '/' + 'bbcsport_hybrid_model.pt',
                                 mode='hybrid_embeddings')


if __name__ == '__main__':
    root = get_project_root()

    parser = ArgumentParser(description='Main script', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', type=Path, default=root/'config/config.yaml',
                        help='Enter the config file path.')

    parser.add_argument('--dataset', type=str, default='ten_newsgroups', choices=['all', 'ten_newsgroups', 'bbcsport'],
                        help='Choose the dataset.')

    parser.add_argument('--operation', type=str, default='graph_classification',
                        choices=['all', 'text_indexing', 'text_classification', 'graph_classification',
                                 'hybrid_classification', 'visualization'],
                        help='Choose the operation to be performed.')

    args, unknown = parser.parse_known_args()

    with args.config.open() as y:
        config = yaml.load(y, Loader=yaml.FullLoader)

    do_operations(root=root,
                  config=config,
                  dataset=args.dataset,
                  operation=args.operation)
