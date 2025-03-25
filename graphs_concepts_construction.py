import shutil
import pickle
import random
import numpy as np
import networkx as nx


def calc_percentile_val(concepts, attribute, percentile):
    if attribute == 'supports':
        attribute_vals = [concept[attribute][0] for concept in concepts]
        return np.percentile(attribute_vals, percentile)

    elif attribute == 'extent':
        attribute_vals = [len(concept[attribute]) for concept in concepts]
        return np.percentile(attribute_vals, percentile)

    else:
        attribute_vals = [concept[attribute] for concept in concepts]
        return np.percentile(attribute_vals, percentile)


def merge_subgraphs(subgraphs):

    merged_graph = nx.MultiDiGraph()

    for subgraph in subgraphs:
        for edge in subgraph.edges(data=True):
            if edge not in merged_graph.edges(data=True):
                merged_graph.add_edge(edge[0], edge[1], label=edge[2]['label'])

    return merged_graph


def save_graph_concepts(collected_graph_concepts, graph_concepts_dataset_prefix, concept_type, class_name):
    indices = list(np.arange(len(collected_graph_concepts)))

    train_subset = len(indices) * 0.9

    train_indices = random.sample(indices, int(train_subset))

    test_indices = list(set(indices) - set(train_indices))

    for train_index in train_indices:

        train_graph_concept_filename = collected_graph_concepts[train_index]['graph_concept_name'] + '_train.gml'

        nx.write_gml(collected_graph_concepts[train_index]['graph_concept'], train_graph_concept_filename)

        train_graph_concept_filename_path = (graph_concepts_dataset_prefix + '_'
                                             + concept_type + '/'
                                             + class_name + '/' + 'raw/'
                                             + train_graph_concept_filename)

        shutil.move(train_graph_concept_filename,
                    train_graph_concept_filename_path)

    for test_index in test_indices:

        test_graph_concept_filename = collected_graph_concepts[test_index]['graph_concept_name'] + '_test.gml'

        nx.write_gml(collected_graph_concepts[test_index]['graph_concept'], test_graph_concept_filename)

        test_graph_concept_filename_path = (graph_concepts_dataset_prefix + '_'
                                            + concept_type + '/'
                                            + class_name + '/' + 'raw/'
                                            + test_graph_concept_filename)

        shutil.move(test_graph_concept_filename,
                    test_graph_concept_filename_path)


def construct_graph_concepts(graph_concepts_dataset_prefix, dataset, classes, concept_type, min_support, max_penalty):

    if concept_type == 'weighted_frequent_subgraphs':

        print(f"Constructing {concept_type} for {dataset}")

        for class_name in classes:
            weighted_frequent_subgraphs_file_name = (graph_concepts_dataset_prefix + '_' + concept_type
                                                     + '/' + class_name + '/' + class_name + '_' + concept_type + '.pickle')

            collected_weighted_frequent_subgraphs = []

            with (open(weighted_frequent_subgraphs_file_name, 'rb') as f):
                weighted_frequent_subgraphs = pickle.load(f)

                class_min_support = calc_percentile_val(weighted_frequent_subgraphs, 'supports', min_support)
                class_max_penalty = calc_percentile_val(weighted_frequent_subgraphs, 'penalty_5', max_penalty)

                for weighted_frequent_subgraph in weighted_frequent_subgraphs:
                    assert len(weighted_frequent_subgraph['subgraphs']) == 1, \
                        f"There's supposed to be only 1 frequent subgraph per concept"

                    if weighted_frequent_subgraph['penalty_5'] <= class_max_penalty \
                       and weighted_frequent_subgraph['supports'][0] >= class_min_support:

                        collected_weighted_frequent_subgraphs.append({'graph_concept_name': weighted_frequent_subgraph['id'],
                                                                      'graph_concept': weighted_frequent_subgraph['subgraphs'][0]})

                        '''
                        weighted_frequent_subgraph_filename = weighted_frequent_subgraph['id'] + '.gml'

                        nx.write_gml(weighted_frequent_subgraph['subgraphs'][0], weighted_frequent_subgraph_filename)

                        weighted_frequent_subgraph_concept_filename_path = (graph_concepts_dataset_prefix + '_'
                                                                            + concept_type + '/'
                                                                            + class_name + '/' + 'raw/'
                                                                            + weighted_frequent_subgraph_filename)

                        shutil.move(weighted_frequent_subgraph_filename,
                                    weighted_frequent_subgraph_concept_filename_path)
                        '''

            save_graph_concepts(collected_weighted_frequent_subgraphs, graph_concepts_dataset_prefix,
                                concept_type, class_name)

    elif concept_type == 'weighted_concepts':

        print(f"Constructing {concept_type} for {dataset}")

        for class_name in classes:
            weighted_concepts_file_name = (graph_concepts_dataset_prefix + '_' + concept_type
                                           + '/' + class_name + '/' + class_name + '_' + concept_type + '.pickle')

            collected_weighted_concepts = []

            with (open(weighted_concepts_file_name, 'rb') as f):
                weighted_concepts = pickle.load(f)

                class_min_support = calc_percentile_val(weighted_concepts, 'extent', min_support)
                class_max_penalty = calc_percentile_val(weighted_concepts, 'penalty_5', max_penalty)

                for weighted_concept in weighted_concepts:

                    if weighted_concept['penalty_5'] <= class_max_penalty \
                       and weighted_concept['supports'][0] >= class_min_support:

                        weighted_concept_graph = merge_subgraphs(weighted_concept['subgraphs'])

                        collected_weighted_concepts.append({'graph_concept_name': weighted_concept['id'],
                                                            'graph_concept': weighted_concept_graph})

                        '''
                        weighted_concept_filename = weighted_concept['id'] + '.gml'

                        weighted_concept_graph = merge_subgraphs(weighted_concept['subgraphs'])

                        nx.write_gml(weighted_concept_graph, weighted_concept_filename)

                        weighted_concept_graph_filename_path = (graph_concepts_dataset_prefix + '_' + concept_type + '/'
                                                                + class_name + '/' + 'raw/' + weighted_concept_filename)

                        shutil.move(weighted_concept_filename,
                                    weighted_concept_graph_filename_path)
                        '''

            save_graph_concepts(collected_weighted_concepts, graph_concepts_dataset_prefix,
                                concept_type, class_name)

    elif concept_type == 'weighted_equivalence_classes':

        print(f"Constructing {concept_type} for {dataset}")

        for class_name in classes:
            weighted_equivalence_classes_file_name = (graph_concepts_dataset_prefix + '_' + concept_type
                                                      + '/' + class_name + '/' + class_name + '_' + concept_type + '.pickle')

            collected_weighted_equivalence_classes = []

            with (open(weighted_equivalence_classes_file_name, 'rb') as f):
                weighted_equivalence_classes = pickle.load(f)

                class_min_support = calc_percentile_val(weighted_equivalence_classes, 'extent', min_support)
                class_max_penalty = calc_percentile_val(weighted_equivalence_classes, 'penalty_5', max_penalty)

                for weighted_equivalence_class in weighted_equivalence_classes:

                    if weighted_equivalence_class['penalty_5'] <= class_max_penalty \
                            and weighted_equivalence_class['supports'][0] >= class_min_support:

                        weighted_equivalence_class_graph = merge_subgraphs(weighted_equivalence_class['subgraphs'])

                        collected_weighted_equivalence_classes.append({'graph_concept_name': weighted_equivalence_class['id'],
                                                                       'graph_concept': weighted_equivalence_class_graph})

                        '''
                        weighted_equivalence_class_filename = weighted_equivalence_class['id'] + '.gml'

                        weighted_equivalence_class_graph = merge_subgraphs(weighted_equivalence_class['subgraphs'])

                        nx.write_gml(weighted_equivalence_class_graph, weighted_equivalence_class_filename)

                        weighted_concept_graph_filename_path = (graph_concepts_dataset_prefix + '_' + concept_type + '/'
                                                                + class_name + '/' + 'raw/' + weighted_equivalence_class_filename)

                        shutil.move(weighted_equivalence_class_filename,
                                    weighted_concept_graph_filename_path)
                        '''
            save_graph_concepts(collected_weighted_equivalence_classes, graph_concepts_dataset_prefix,
                                concept_type, class_name)

    elif concept_type == 'weighted_closed_subgraphs':

        print(f"Constructing {concept_type} for {dataset}")

        for class_name in classes:
            weighted_closed_subgraphs_file_name = (graph_concepts_dataset_prefix + '_' + concept_type
                                                   + '/' + class_name + '/' + class_name + '_'
                                                   + concept_type + '.pickle')

            collected_weighted_closed_subgraphs = []

            with (open(weighted_closed_subgraphs_file_name, 'rb') as f):
                weighted_concepts = pickle.load(f)

                class_min_support = calc_percentile_val(weighted_concepts, 'extent', min_support)
                class_max_penalty = calc_percentile_val(weighted_concepts, 'penalty_5', max_penalty)

                for weighted_concept in weighted_concepts:

                    if weighted_concept['penalty_5'] <= class_max_penalty \
                            and weighted_concept['supports'][0] >= class_min_support:

                        for weighted_closed_subgraph_idx, weighted_closed_subgraph in enumerate(weighted_concept['subgraphs']):

                            collected_weighted_closed_subgraphs.append({'graph_concept_name': weighted_concept['id'] + '_' + str(weighted_closed_subgraph_idx),
                                                                        'graph_concept': weighted_closed_subgraph})

                            '''
                            weighted_closed_subgraph_filename = weighted_concept['id'] + '_' + str(weighted_closed_subgraph_idx) + '.gml'

                            nx.write_gml(weighted_closed_subgraph, weighted_closed_subgraph_filename)

                            weighted_closed_subgraph_filename_path = (graph_concepts_dataset_prefix + '_' + concept_type
                                                                      + '/' + class_name + '/' + 'raw/'
                                                                      + weighted_closed_subgraph_filename)

                            shutil.move(weighted_closed_subgraph_filename, weighted_closed_subgraph_filename_path)
                            '''

            save_graph_concepts(collected_weighted_closed_subgraphs, graph_concepts_dataset_prefix,
                                concept_type, class_name)
