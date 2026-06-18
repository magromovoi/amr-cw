import penman
import spacy
import networkx as nx


def normalize_constant(target):
    return str(target).strip('"').lower()


def penman_to_graph(penman_str):
    g = penman.decode(penman_str)

    var_to_label = {}
    for inst in g.instances():
        var_to_label[inst.source] = inst.target

    graph = nx.MultiDiGraph()
    for label in var_to_label.values():
        graph.add_node(label)

    seen_triples = set()

    def add_typed_edge(src_label, tgt_label, role):
        triple = (src_label, tgt_label, role)
        if triple in seen_triples:
            return
        seen_triples.add(triple)
        graph.add_edge(src_label, tgt_label, label=role)

    for edge in g.edges():
        src = var_to_label.get(edge.source)
        tgt = var_to_label.get(edge.target)
        if src is None or tgt is None:
            continue
        add_typed_edge(src, tgt, edge.role)

    for attr in g.attributes():
        src = var_to_label.get(attr.source)
        if src is None:
            continue
        tgt = normalize_constant(attr.target)
        graph.add_node(tgt)
        add_typed_edge(src, tgt, attr.role)

    return graph


def merge_by_label(graphs):
    document_graph = nx.MultiDiGraph()
    seen_triples = set()

    for graph in graphs:
        for node in graph.nodes():
            document_graph.add_node(node)

        for src, tgt, data in graph.edges(data=True):
            role = data['label']
            triple = (src, tgt, role)
            if triple in seen_triples:
                continue
            seen_triples.add(triple)
            document_graph.add_edge(src, tgt, label=role)

    return document_graph


def split_sentences(text, nlp):
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


def text_to_graph(text, stog_model, nlp):
    sentences = split_sentences(text, nlp)
    penman_strs = stog_model.parse_sents(sentences, add_metadata=False)

    sentence_graphs = []
    for penman_str in penman_strs:
        if penman_str is None:
            continue
        sentence_graphs.append(penman_to_graph(penman_str))

    return merge_by_label(sentence_graphs)


def load_sentence_splitter():
    nlp = spacy.blank('en')
    nlp.add_pipe('sentencizer')
    return nlp
