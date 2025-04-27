# stable-AMR-graphs-concept-whitening
This repository contains the code for classifying documents using their document graphs and generating explanations for the classifications, using Concept Whitening (CW). The graph concepts used here for concept whitening, are based on stable AMR graph patterns that are obtained from the training document graphs of each document class, using methods based on Formal Concept Analysis (FCA).

Run the following commands to install the required libraries:

```
pip3 install torch torchvision torchaudio
pip install torch_geometric
pip install transformers
pip install numpy pandas networkx scikit-learn scipy
```
Refer to the commands inside the shell script files to perform the various operations.
