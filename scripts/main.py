import yaml
from pathlib import Path
from utils import get_project_root
from text_classification import classify_texts
from text_index_construction import construct_text_indices
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def do_operations(root, config, dataset, operation):

    if dataset == 'all':
        if operation == 'text_indexing':
            print("Indexing texts...")
            construct_text_indices('bbcsport', config['bbcsport_classes'],
                                   str(root) + '/' + config['bbcsport_raw_texts_prefix'],
                                   str(root) + '/' + config['raw_texts_csv_prefix'])
            '''
            construct_text_indices('ten_newsgroups', config['ten_newsgroups_classes'],
                                                  str(root) + '/' + config['ten_newsgroups_raw_texts_prefix'],
                                                  str(root) + '/' + config['raw_texts_csv_prefix'])
            '''

        if operation == 'text_classification':
            '''
            classify_texts('ten_newsgroups', config['ten_newsgroups_classes'],
                           str(root) + '/' + config['raw_texts_csv_prefix'])
            '''
            classify_texts('bbcsport', config['bbcsport_classes'],
                           str(root) + '/' + config['raw_texts_csv_prefix'])


if __name__ == '__main__':
    root = get_project_root()

    parser = ArgumentParser(description='Main script', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', type=Path, default=root/'config/config.yaml',
                        help='Enter the config file path.')

    parser.add_argument('--dataset', type=str, default='all', choices=['all', 'ten_newsgroups', 'bbcsport'],
                        help='Choose the dataset.')

    parser.add_argument('--operation', type=str, default='text_classification',
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
