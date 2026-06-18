from math import inf, sqrt


def jaccard(set_a, set_b):
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return intersection / union


def cosine_sim(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sqrt(sum(a * a for a in vec_a))
    norm_b = sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def normalized_quality(candidates):
    supports = [c['support'] for c in candidates]
    penalties = [c['penalty'] for c in candidates]

    sup_min, sup_max = min(supports), max(supports)
    pen_min, pen_max = min(penalties), max(penalties)
    sup_range = sup_max - sup_min if sup_max != sup_min else 1.0
    pen_range = pen_max - pen_min if pen_max != pen_min else 1.0

    norm_sup = [(s - sup_min) / sup_range for s in supports]
    norm_pen = [(p - pen_min) / pen_range for p in penalties]
    raw_quality = [s - p for s, p in zip(norm_sup, norm_pen)]
    q_min, q_max = min(raw_quality), max(raw_quality)
    q_range = q_max - q_min if q_max != q_min else 1.0
    return [(q - q_min) / q_range for q in raw_quality]


def mmr_select(candidates, lam, tau=0.0, min_k=0):
    if len(candidates) <= 1:
        return candidates

    quality = normalized_quality(candidates)

    extents = [set(map(str, c['extent'])) for c in candidates]

    selected_indices = []
    remaining = list(range(len(candidates)))

    best_idx = max(remaining, key=lambda i: quality[i])
    selected_indices.append(best_idx)
    remaining.remove(best_idx)

    while remaining:
        best_score = -inf
        best_i = -1
        for i in remaining:
            max_sim = max(jaccard(extents[i], extents[j]) for j in selected_indices)
            score = lam * quality[i] - (1 - lam) * max_sim
            if score > best_score:
                best_score = score
                best_i = i
        if best_score < tau and len(selected_indices) >= min_k:
            break
        selected_indices.append(best_i)
        remaining.remove(best_i)

    return [candidates[i] for i in selected_indices]
