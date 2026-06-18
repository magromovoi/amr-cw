import os
import re
import copy
import torch
import pickle
import numpy as np
import gensim.downloader as api
from gensim.models import KeyedVectors

vertex_labels = {}
vertex_label_counter = 0

edge_labels = {}
edge_label_counter = 0

vertex_embeddings = {}
edge_embeddings = []

try:
    word_vectors = KeyedVectors.load("data/input/word2vec/word2vec-google-news-300.kv")

except FileNotFoundError:
    wv = api.load('word2vec-google-news-300')
    wv.save("data/input/word2vec/word2vec-google-news-300.kv")

    word_vectors = KeyedVectors.load("data/input/word2vec/word2vec-google-news-300.kv")

string_pattern = "([A-Z]|[a-z])*"


def load_vertex_embeddings():
    global vertex_embeddings

    try:
        with open("data/output/vertex_embeddings.pkl", "rb") as f:
            vertex_embeddings = pickle.load(f)
    except FileNotFoundError:
        vertex_embeddings = {}


def find_vertex_embedding(vertex_label):
    global string_pattern, vertex_embeddings, word_vectors

    vertex_regex_match = str(re.search(string_pattern, vertex_label).group(0))

    if vertex_regex_match in vertex_embeddings:
        return vertex_embeddings[vertex_regex_match]

    else:

        try:
            vertex_embeddings[vertex_regex_match] = copy.deepcopy(word_vectors[vertex_regex_match])
        except KeyError:
            vertex_embeddings[vertex_regex_match] = copy.deepcopy(word_vectors['UNK'])

        return vertex_embeddings[vertex_regex_match]


def save_embeddings():
    global vertex_embeddings

    os.makedirs("data/output", exist_ok=True)
    with open("data/output/vertex_embeddings.pkl", "wb") as f:
        pickle.dump(vertex_embeddings, f)


def find_indices(labels, raw_path):
    labels_pattern = "(" + ")|(".join(labels) + ")"

    label_match = re.search(labels_pattern, raw_path).group(0)

    label_index = int(labels.index(label_match))

    return label_index


def edge_label_lookup(edge):
    global edge_labels, edge_label_counter

    discovery_flag = 1

    if edge in edge_labels:
        return discovery_flag
    else:
        edge_labels[edge] = copy.deepcopy(edge_label_counter)
        edge_label_counter += 1
        discovery_flag = 0
        return discovery_flag


def vertex_label_lookup(vertex):
    global vertex_labels, vertex_label_counter

    discovery_flag = 1

    if vertex in vertex_labels:
        return vertex_labels[vertex], discovery_flag

    else:
        vertex_labels[vertex] = copy.deepcopy(vertex_label_counter)
        vertex_label_counter += 1
        discovery_flag = 0
        return vertex_labels[vertex], discovery_flag


def find_graph_info(graph):
    global vertex_labels, vertex_label_counter, edge_labels, edge_label_counter

    load_vertex_embeddings()

    vertex_features = []

    head_vertices = []
    tail_vertices = []

    for vertex1, vertex2, data in graph.edges(data=True):

        edge_dis_1 = edge_label_lookup((vertex1, vertex2))

        if edge_dis_1 == 0:

            head_vertex, head_dis = vertex_label_lookup(vertex1)
            tail_vertex, tail_dis = vertex_label_lookup(vertex2)

            if head_dis == 0:
                head_vertex_embedding = find_vertex_embedding(vertex1)
                vertex_features.append(head_vertex_embedding)

            if tail_dis == 0:
                tail_vertex_embedding = find_vertex_embedding(vertex2)
                vertex_features.append(tail_vertex_embedding)

            head_vertices.append(head_vertex)
            tail_vertices.append(tail_vertex)

    edge_indices = [head_vertices, tail_vertices]

    final_vertex_features = torch.tensor(np.asarray(vertex_features), dtype=torch.float)
    final_edge_indices = torch.tensor(np.asarray(edge_indices), dtype=torch.long)

    vertex_labels = {}
    vertex_label_counter = 0

    edge_labels = {}
    edge_label_counter = 0

    save_embeddings()

    return final_vertex_features, final_edge_indices
