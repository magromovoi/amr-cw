from collections import defaultdict

from amr_cw.concepts.subgraph_miner import frequent_single_triples, triple_sets


def single_triple_extents(freq1, graph_triples):
    extents = {}
    for t in freq1:
        extents[t] = frozenset(i for i, ts in enumerate(graph_triples) if t in ts)
    return extents


def fca_concept_extents(freq1, graph_triples, min_support):
    single_extents = single_triple_extents(freq1, graph_triples)

    concepts = set()
    for t in freq1:
        e = single_extents[t]
        if len(e) >= min_support:
            concepts.add(e)
    concepts.add(frozenset(range(len(graph_triples))))

    work = list(concepts)
    i = 0
    while i < len(work):
        base = work[i]
        i += 1
        for other in list(concepts):
            inter = base & other
            if len(inter) >= min_support and inter not in concepts:
                concepts.add(inter)
                work.append(inter)

    return concepts


def delta_stability(concept_extents):
    exts = sorted(concept_extents, key=lambda e: -len(e))
    sets = [set(e) for e in exts]
    n = len(sets)
    sizes = [len(s) for s in sets]

    proper_subsets = [[] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            if sizes[b] < sizes[a] and sets[b] < sets[a]:
                proper_subsets[a].append(b)

    deltas = {}
    for a in range(n):
        subs = proper_subsets[a]
        if not subs:
            deltas[exts[a]] = sizes[a]
            continue
        direct = [x for x in subs if not any(x != y and sets[x] < sets[y] for y in subs)]
        deltas[exts[a]] = min(sizes[a] - sizes[x] for x in direct)

    return deltas


def stable_concepts(graphs, min_support, stability_threshold):
    graph_triples = triple_sets(graphs)
    freq1 = frequent_single_triples(graph_triples, min_support)
    concept_extents = fca_concept_extents(freq1, graph_triples, min_support)
    deltas = delta_stability(concept_extents)
    return {e for e, d in deltas.items() if d >= stability_threshold}, graph_triples, freq1


def closed_connected_patterns(mined_patterns, stable_extents):
    by_extent = defaultdict(list)
    for pattern, extent in mined_patterns.items():
        if extent in stable_extents:
            by_extent[extent].append(pattern)

    type1 = []
    for extent, patterns in by_extent.items():
        pattern_sets = [set(p) for p in patterns]
        for k, pattern in enumerate(patterns):
            is_maximal = not any(j != k and pattern_sets[k] < pattern_sets[j]
                                 for j in range(len(patterns)))
            if is_maximal:
                type1.append((pattern, extent))

    return type1, by_extent
