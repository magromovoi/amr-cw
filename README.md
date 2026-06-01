# stable-AMR-graphs-concept-whitening
This repository contains the official code for the paper **"Explainable Document Classification via Concept Whitening and Stable Graph Patterns"** by Eric George Parakal, Sergei O. Kuznetsov, Ilya Makarov, and Nikita Severin.

Parts of this implementation are adapted from [ConceptWhitening](https://github.com/zhiCHEN96/ConceptWhitening) and [GraphCW](https://github.com/KRLGroup/GraphCW).

## Installation

Run the following commands to install the required libraries:

```
pip3 install torch torchvision torchaudio
pip install torch_geometric
pip install transformers
pip install numpy pandas networkx scikit-learn scipy
```

## Usage

Due to their large size, model checkpoints are stored at **[[checkpoints]](https://www.dropbox.com/scl/fo/liabqakl4w4ftqf6mw4ok/AMY2tqbkHWegCKQQehGKUDs?rlkey=yyeets245smzgv6c5ev1irx9v&st=3r3qevtt&dl=0)**.  
Please download them and place the files into the `checkpoints/` directory.

Word2Vec node embeddings are stored separately at **[[word2vec embeddings]](https://www.dropbox.com/scl/fo/uvbpdblfhq88zg6dansbq/AI70P4hzE_T1_2xcVlIM6jY?rlkey=oaq8jzaaoloh44260qe86m44o&st=qiuss65t&dl=0)**.  
Please download them and place the files in the main directory (alongside the `.py` scripts).

### Running the experiments

- For the **10 Newsgroups** dataset:
  - Use the `ten_newsgroups_GNN_black-box_commands.sh` file to run experiments with **black-box GNNs** (without the CW module).
  - Use the `ten_newsgroups_CW_commands.sh` file to run experiments with **white-box GNNs** (with the CW module).

- For the **BBC Sport** dataset:
  - Use the `bbcsport_GNN_black-box_commands.sh` file for black-box GNN experiments.
  - Use the `bbcsport_GNN_CW_commands.sh` file for white-box GNN experiments.

To change the GNN model and toggle residual connections, please modify the `graph_conv_type` and `graph_residual_connections` variables in the `config.yaml` file.  

Additionally, the target classes for computing concept gradient importances can be adjusted:
- For the **10 Newsgroups** dataset: modify the `ten_newsgroups_target_class` field.
- For the **BBC Sport** dataset: modify the `bbcsport_target_class` field.

### Multi-seed runs

`run_multiseed.sh` runs the full pipeline on BBC Sport: concept construction, concept-whitening training, then doc-CAP evaluation. It repeats this for the baseline and exp configs across seeds 42, 123, 456, 789, and 1024. From the repository root:

```
./run_multiseed.sh
```

Each run writes to `data/output/runs/bbcsport_<config>_seed<N>/`, containing its own `log.txt`, `checkpoints/`, `config.yaml`, and `run_meta.json`.

To run one stage at a time for a single seed and config:

```
RUN=data/output/runs/bbcsport_exp_seed42

.venv/bin/python main.py --seed 42 --config config.yaml --dataset bbcsport --operation graph_concept_construction --concept_type weighted_frequent_subgraphs --run_dir "$RUN"

.venv/bin/python main.py --seed 42 --config config.yaml --dataset bbcsport --operation graph_concept_whitening --mode train --concept_type weighted_frequent_subgraphs --run_dir "$RUN"

.venv/bin/python main.py --seed 42 --config config.yaml --dataset bbcsport --operation evaluation --mode doc_cap --concept_type weighted_frequent_subgraphs --run_dir "$RUN"
```

Use `--config config_baseline.yaml` for the baseline.
