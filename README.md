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

Due to their large size, model checkpoints are stored at **[LINK_1]**.  
Please download them and place the files into the `checkpoints/` directory.

Word2Vec node embeddings are stored separately at **[LINK_2]**.  
Please download them and place the files in the main directory (alongside the `.py` scripts).

### Running the experiments

- For the **10 Newsgroups** dataset:
  - Use the `ten_newsgroups_GNN_black-box_commands.sh` file to run experiments with **black-box GNNs** (without the CW module).
  - Use the `ten_newsgroups_CW_commands.sh` file to run experiments with **white-box GNNs** (with the CW module).

- For the **BBC Sport** dataset:
  - Use the `bbcsport_GNN_black-box_commands.sh` file for black-box GNN experiments.
  - Use the `bbcsport_GNN_CW_commands.sh` file for white-box GNN experiments.
