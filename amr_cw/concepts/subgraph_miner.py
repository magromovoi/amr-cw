import re


def human_sort_key(s):
    return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', s)]


def triple_sets(graphs):
    return [{(u, v, d['label']) for u, v, d in g.edges(data=True)} for g in graphs]


def frequent_single_triples(graph_triples, min_support):
    counts = {}
    for triples in graph_triples:
        for t in triples:
            counts[t] = counts.get(t, 0) + 1
    return {t for t, n in counts.items() if n >= min_support}


def extent(triple_set, graph_triples):
    return frozenset(i for i, ts in enumerate(graph_triples) if triple_set <= ts)


def adjacent_triples(current, allowed):
    nodes = set()
    for u, v, _ in current:
        nodes.add(u)
        nodes.add(v)
    out = set()
    for t in allowed:
        if t in current:
            continue
        u, v, _ = t
        if u in nodes or v in nodes:
            out.add(t)
    return out


def mine_frequent_subgraphs(graphs, min_support):
    graph_triples = triple_sets(graphs)
    freq1 = frequent_single_triples(graph_triples, min_support)

    single_extents = {t: extent(frozenset([t]), graph_triples) for t in freq1}

    results = {}
    seen = set()

    def expand(current_frozen, current_extent):
        for t in adjacent_triples(current_frozen, freq1):
            candidate = current_frozen | {t}
            if candidate in seen:
                continue
            seen.add(candidate)

            cand_extent = current_extent & single_extents[t]
            if len(cand_extent) < min_support:
                continue

            results[candidate] = cand_extent
            expand(candidate, cand_extent)

    for t in freq1:
        fs = frozenset([t])
        seen.add(fs)
        results[fs] = single_extents[t]
        expand(fs, single_extents[t])

    return results, graph_triples
