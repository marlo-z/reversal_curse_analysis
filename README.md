# Analysis of 'Reversal Curse'

This repo contains the code for the experiments in our [paper](https://arxiv.org/abs/2405.04669): 

**Towards a Theoretical Understanding of the 'Reversal Curse' via Training Dynamics**

[Hanlin Zhu](https://hanlinzhu.com), 
[Baihe Huang](https://scholar.google.com/citations?user=chICXXMAAAAJ&hl=en),
[Shaolun Zhang](https://marlo-z.github.io/shaolun-zhang.github.io/),
[Michael Jordan](https://people.eecs.berkeley.edu/~jordan),
[Jiantao Jiao](https://people.eecs.berkeley.edu/~jiantao),
[Yuandong Tian](https://scholar.google.com/citations?user=0mgEF28AAAAJ&hl=en),
[Stuart Russell](https://people.eecs.berkeley.edu/~russell)

## Installation
```sh
conda create -n reversal_curse python=3.10
conda activate reversal_curse
pip install -r requirements.txt
```

## Usage

### Reversal Logic Experiments

To run the standard reversal logic experiment of the form $A_i \rightarrow B_i$ and $B_i \leftarrow A_i$ and plot training and validation log probabilities:
```sh
CUDA_VISIBLE_DEVICES=0 python3 train_reverse.py
```

To run the standard reversal logic experiment of the form $A_i \rightarrow B_i$ and $B_i \leftarrow A_i$ and visualize the logits of token $A_i$ given $B_i$ and the logits of $B_i$ given $A_i$:
```sh
CUDA_VISIBLE_DEVICES=0 python3 train_reverse_logits.py
```

To run the standard reversal logic experiment of the form $A_i \rightarrow B_i$ and $B_i \leftarrow A_i$ and visualize cosine similarity between embedding vectors of tokens $A_i$ and $B_i$:
```sh
CUDA_VISIBLE_DEVICES=0 python3 train_reverse_embed.py
```

To run the reversal logic experiment with In-Context Learning of the form $A_i R B_i \Longleftrightarrow B_i R^{-1} A_i$ and and plot training and validation log probabilities:
```sh
CUDA_VISIBLE_DEVICES=0 python3 train_reverse_ICL.py
```

### Chain-of-Thought Experiments

To run the standard Chain-of-Thought experiment of the form $A_i \rightarrow B_i$, $B_i \rightarrow C_i$, $A_i \leadsto C_i$ and plot the training and validation probabilities:
```sh
CUDA_VISIBLE_DEVICES=0 python3 train_chain.py
```

To run the standard Chain-of-Thought experiment of the form $A_i \rightarrow B_i$, $B_i \rightarrow C_i$, $A_i \leadsto C_i$ and visualize the logits of tokens 1) $B_i$ given $A_i$, 2) $C_i$ given $B_i$ and 3) $C_i$ given $A_i$:
```sh
CUDA_VISIBLE_DEVICES=0 python3 train_reverse_logits.py
```

To run the alternative version of Chain-of-Thought experiment with correlated tokens of the form $Ai \rightarrow Bi$, $Bi \rightarrow Ci$, $Ai \leadsto Ci$, where each entity is comprised of 2 tokens with $A,B,C$ tokens fixed and plot the training and validation probabilities:
```sh
CUDA_VISIBLE_DEVICES=0 python3 train_chain_related_tokens.py
```

### Command Line Arguments

When running the scripts for each experiment, there are several command line argument that can be passed to customize model configuration, training hyperparameters and dataset generation. The following example demonstrates how these arguments can be passed via command line, and their respective default values:
```
CUDA_VISIBLE_DEVICES=0 python3 train_reverse.py \
    --pos_encode_type 'absolute' \      # positional embedding: 'null', 'absolute', 'rotary'
    --n_layers 24 \                     # number of transformer layers
    --embed_dim 768 \                   # dimension of the token embedding vectors
    --vocab_size 800 \                  # size of the vocabulary from which the datasets are constructed
    --word_size 1 \                     # number of tokens that forms an entity
    --seed 1234 \
    --num_epochs 3000 \
    --batch_size 64 \
    --lr 0.01 \
    --decay 0.9
    --betas (0.9, 0.999) \
    --loss_whole_sequence \             # see note
    --freeze_wte_wpe \                  # see note
    --output_dir 'exp_reverse' \        # where plots will be saved
```

**NOTE**: The defautl setting is without passing the follow flags: \
With `--loss_whole_sequence` flag, loss is applied to entire input sequence, otherwise only applied to tokens corresponding to the last entity.\
With `--freeze_wte_wpe` flag, the token embedding matrix and the positional embedding matrices of the model are frozen, otherwise they are trainable.

## Acknowledgement
We used [this](https://github.com/lucidrains/rotary-embedding-torch) implementation of the Rotary Embeddings in out experiments.
