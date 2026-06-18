# Diversity Selection for Verifiable Concept-Based Explanations Under a Review Time Budget

## Abstract

Interpretation of an ML model's output is only useful if a human can verify it, which matters most 
when the output is used to make important decisions. A correct explanation that a reviewer cannot 
verify within their limited time provides no real benefit: when the explanation is redundant, the 
reviewer spends their time budget on near-duplicates instead of distinct findings. We study this on a 
pipeline that interprets a neural text classifier using the semantic structure of documents: text is 
parsed into Abstract Meaning Representation (AMR) graphs, frequent subgraphs are mined as FCA 
concepts, and Concept Whitening (CW) aligns latent axes to concepts. In this setting concepts are 
correct, but many are redundant. We build on the base pipeline (see citation below), which keeps 
one concept per class using a support-and-penalty filter. We add diversity selection at the concept 
selection step before CW. We measure verifiability @ B: the number of distinct ground-truth 
techniques a reviewer recovers within a time budget of B concepts. Diversity recovers 2-3 more 
distinct techniques than the base pipeline's quality-only rule across budgets (p<0.01), and on BBC 
Sport, the base paper's own dataset, decreases inter-concept similarity from 0.72 to 0.38 (lower is 
better) while preserving the classifier's predictive power. 
Our main contribution is an evaluation protocol for the verifiability of concept-based 
explanations: a deterministic simulated auditor with decoy and null falsification controls, which we run on a 
controlled synthetic corpus with injected ground-truth communication techniques. 
We further show that diversity selection reduces inter-concept similarity at the selection step, 
add an activation-space selector that computes similarity in CW's whitened-activation space, 
and a document-level variant of the Concept Alignment Performance (CAP) metric.

## Built on Parakal et al.

The AMR + Concept Whitening pipeline is from:

> E. G. Parakal, S. O. Kuznetsov, I. Makarov, N. Severin.
> *Explainable Document Classification via Concept Whitening and Stable Graph Patterns.*
> IEEE Access, 2024.
> https://github.com/ericparakal/stable-AMR-graphs-concept-whitening

## Setup

The project uses a `uv` virtual environment.

```
uv venv
uv pip install -e .
source .venv/bin/activate
```

## Generating the corpus

Public datasets have no concept-level ground truth, so the experiments run on a synthetic corpus
with ground-truth techniques injected into the text. Generation and annotation 
write to `data/output/synthetic_corpus/<corpus>/`.

The generator we used is Claude Opus 4.8; it's possible to pick a backend with `CW_LLM_BACKEND`. 
The `api` backend needs `ANTHROPIC_API_KEY`:

```
# using Anthropic API
CW_LLM_BACKEND=api ANTHROPIC_API_KEY=... \
python -m amr_cw.scripts.generate_corpus --corpus calibration --n 65 --seed 42 --model claude-opus-4-8
```

Ground truth is derived from an independent annotation using a different model from
a different provider, namely OpenAI's `gpt-4.1-2025-04-14`, which needs `OPENAI_API_KEY`:

```
OPENAI_API_KEY=... \
python -m amr_cw.scripts.annotate_techniques --corpus eval \
  --in data/output/synthetic_corpus/eval/eval_raw_texts.csv \
  --out data/output/synthetic_corpus/eval/annotations.json
```

Calibration is used to tune generation parameters and the technique set, which we then freeze to
`amr_cw.scripts.freeze_params`. The evaluation corpus is generated fresh using the frozen parameters. 


## Running the synthetic pipeline

`amr_cw/scripts/run_synthetic_pipeline.py` processes one corpus from text to CW outputs with diversity. 
Pass `--dry-run` to build the step plan without executing it:

```
python -m amr_cw.scripts.run_synthetic_pipeline --variant smoke --min-support 3 --stability 3 --dry-run
```

Drop `--dry-run` to execute the steps. 
Pass `--rules` (allows multiple args) to choose selection rules (default: `none quality_only facility_location`, 
also available: `set_cover`, `dpp`, `mmr`). Each step can be used as an entrypoint: 
`build_amr_graphs` (text to AMR graphs), 
`mine_concepts` (FCA stable concept mining), 
`amr_cw.main` (train base GNN), 
per-rule `amr_cw.main` (CW). 
Results are written to `data/output/synthetic_corpus/<variant>/runs/<rule>/`.

## Analysis and figures

The analysis steps read the outputs, perform analysis, and create figures:

```
ROOT=data/output/synthetic_corpus/eval
MAP=$ROOT/mined_concepts_weighted_frequent_subgraphs_concept_techniques.json

# Concept -> technique mapping
python -m amr_cw.evaluation.concept_techniques --pickles_prefix $ROOT/mined_concepts \
  --graphs_dir $ROOT/graphs_dataset/synthetic_graphs_dataset/raw \
  --annotations $ROOT/annotations.json --threshold 0.6 --out $MAP

# Primary metric: verifiability @ budget k
for K in 4 6 8; do
  python -m amr_cw.evaluation.matched_k --pickles_prefix $ROOT/mined_concepts \
    --concept_techniques $MAP --k $K --seeds 0 1 2 3 4 \
    --diversity_rule facility_location --out $ROOT/matched_k_k$K.json
done

# Null control: shuffle the technique mappings
python -m amr_cw.evaluation.build_null_map --pickles_prefix $ROOT/mined_concepts \
  --concept_techniques $MAP --seed 0 --out $ROOT/null_concept_techniques_seed0.json
```
