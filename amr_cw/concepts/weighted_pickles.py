import networkx as nx


def triples_to_graph(triples):
    g = nx.MultiDiGraph()
    for u, v, role in triples:
        g.add_edge(u, v, label=role)
    return g


def other_class_doc_count(triples, other_class_graph_triples):
    triple_set = set(triples)
    return sum(1 for ts in other_class_graph_triples if triple_set <= ts)


def make_concept(concept_id, subgraph_triples_list, extent, other_class_graph_triples):
    subgraphs = [triples_to_graph(t) for t in subgraph_triples_list]
    representative = set(subgraph_triples_list[0]) if subgraph_triples_list else set()
    penalty_5 = other_class_doc_count(representative, other_class_graph_triples)

    concept = {
        'id': concept_id,
        'supports': [len(extent)],
        'subgraphs': subgraphs,
        'extent': sorted(extent),
        'penalty_5': penalty_5,
    }
    return concept


def build_frequent_subgraphs(type1_patterns, class_name, other_class_graph_triples):
    concepts = []
    for i, (pattern, extent) in enumerate(type1_patterns):
        concepts.append(make_concept(
            f'{class_name}_frequent_subgraph_{i}',
            [list(pattern)],
            extent,
            other_class_graph_triples,
        ))
    return concepts
