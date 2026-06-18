import os
import shutil
import pickle
import random
import numpy as np
import networkx as nx

from amr_cw.diversity import select


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

    concept_dir = graph_concepts_dataset_prefix + '_' + concept_type + '/' + class_name
    raw_dir = concept_dir + '/raw/'
    processed_dir = concept_dir + '/processed/'
    if os.path.exists(raw_dir):
        shutil.rmtree(raw_dir)
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(raw_dir)

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


def construct_graph_concepts(concept_pickles_prefix, graph_concepts_dataset_prefix, dataset, classes, concept_type,
                             min_support, max_penalty,
                             diversity_method='none', diversity_lambda=0.5, diversity_min_k=30,
                             diversity_min_k_proportional=0.0, diversity_tau=0.0,
                             diversity_skip_below=0, diversity_sim='jaccard'):

    if concept_type == 'weighted_frequent_subgraphs':

        print(f"Constructing {concept_type} for {dataset}")

        for class_name in classes:
            weighted_frequent_subgraphs_file_name = (concept_pickles_prefix + '_' + concept_type
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
                                                                      'graph_concept': weighted_frequent_subgraph['subgraphs'][0],
                                                                      'extent': weighted_frequent_subgraph['extent'],
                                                                      'support': weighted_frequent_subgraph['supports'][0],
                                                                      'penalty': weighted_frequent_subgraph['penalty_5']})

            assert (len(collected_weighted_frequent_subgraphs)/len(weighted_frequent_subgraphs)) >= 0.01

            pool_too_small = diversity_skip_below > 0 and len(collected_weighted_frequent_subgraphs) < diversity_skip_below
            if diversity_method != 'none' and not pool_too_small:
                effective_min_k = max(diversity_min_k, int(diversity_min_k_proportional * len(collected_weighted_frequent_subgraphs)))
                collected_weighted_frequent_subgraphs = select(
                    diversity_method, collected_weighted_frequent_subgraphs, diversity_lambda,
                    tau=diversity_tau, min_k=effective_min_k, sim=diversity_sim)

            save_graph_concepts(collected_weighted_frequent_subgraphs, graph_concepts_dataset_prefix,
                                concept_type, class_name)

    elif concept_type == 'weighted_pattern_concepts':

        print(f"Constructing {concept_type} for {dataset}")

        for class_name in classes:
            weighted_concepts_file_name = (concept_pickles_prefix + '_' + concept_type
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

            assert (len(collected_weighted_concepts) / len(weighted_concepts)) >= 0.01

            save_graph_concepts(collected_weighted_concepts, graph_concepts_dataset_prefix,
                                concept_type, class_name)

    elif concept_type == 'weighted_filtered_equivalence_classes':

        print(f"Constructing {concept_type} for {dataset}")

        for class_name in classes:
            weighted_equivalence_classes_file_name = (concept_pickles_prefix + '_' + concept_type
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

            assert (len(collected_weighted_equivalence_classes) / len(weighted_equivalence_classes)) >= 0.01

            save_graph_concepts(collected_weighted_equivalence_classes, graph_concepts_dataset_prefix,
                                concept_type, class_name)

    elif concept_type == 'weighted_closed_subgraphs':

        print(f"Constructing {concept_type} for {dataset}")

        for class_name in classes:
            weighted_closed_subgraphs_file_name = (concept_pickles_prefix + '_' + concept_type
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

            assert (len(collected_weighted_closed_subgraphs) / len(weighted_concepts)) >= 0.01

            save_graph_concepts(collected_weighted_closed_subgraphs, graph_concepts_dataset_prefix,
                                concept_type, class_name)
