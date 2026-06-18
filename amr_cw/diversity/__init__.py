from amr_cw.diversity.mmr import mmr_select, jaccard
from amr_cw.diversity.selectors import (
    facility_location, set_cover, greedy_map_dpp, extent_features)
from amr_cw.diversity.activation_space import select_precomputed

SIM_NOTIONS = {
    'jaccard': (jaccard, extent_features),
}


def select(method, candidates, lam=0.5, tau=0.0, min_k=0, sim='jaccard'):
    if method in ('none', None):
        return candidates
    if method in ('mmr', 'mrmr'):
        return mmr_select(candidates, lam, tau=tau, min_k=min_k)
    if sim not in SIM_NOTIONS:
        raise ValueError(f"unknown similarity notion '{sim}'")
    sim_fn, feature_fn = SIM_NOTIONS[sim]
    if method == 'facility_location':
        return facility_location(candidates, min_k, sim_fn, feature_fn)
    if method == 'set_cover':
        if sim != 'jaccard':
            raise ValueError("set_cover is extensional (document sets) and ignores sim; "
                             f"got sim='{sim}'")
        return set_cover(candidates, min_k)
    if method == 'dpp':
        return greedy_map_dpp(candidates, min_k, sim_fn, feature_fn)
    if method == 'activation_space':
        return select_precomputed(candidates, min_k)
    raise ValueError(f"unknown diversity method '{method}'")
