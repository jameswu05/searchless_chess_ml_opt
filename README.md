# Variational Latent Regularization for Amortized Chess Planning: A KL-Divergence Bottleneck Approach to Searchless Value Prediction

Authors: Harman Aujla, Vasilije Dragovic, Calissa Tyrrell, James Wu

This repository provides an implementation of our CSCI-4140 Final Course Project. It extends upon the paper _Amortized Planning with Large-Scale Transformers: A Case Study on Chess_ by Ruoss et al. 2024. Our abstract is below.

> Transformer-based chess models have recently demonstrated grandmaster-level chess play without explicit search by distilling the value estimates of Stockfish 16 into a feedforward network via supervised learning. While these models achieve strong performance at scale, they exhibit known failure modes including overconfidence in winning positions and limited generalization to novel board states. In this work, we propose an architectural modification to the searchless chess transformer that introduces a variational autoencoder bottleneck between the main transformer stack and the output head. Specifically, after encoding the board and action tokens through a standard decoder-only transformer, we compress the resulting representations through a Multi-Head Latent Attention module into a structured latent space parameterized by a Gaussian distribution, from which a sampled latent vector is injected back into a shallow decoder to produce the final action-value predictions. Training incorporates a KL-divergence regularization term weighted by a scalar $\beta$, encouraging the latent distribution to remain close to a standard normal prior. We train our model on a single shard of the ChessBench dataset and evaluate on puzzle accuracy, action accuracy, and Kendall's $\tau$ against Stockfish ground truth. Our experiments remain in progress.

## Contents ##

```
.
|
├── BayesElo                        - Elo computation (need to be installed)
|
├── checkpoints                     - Model checkpoints (need to be downloaded)
|   ├── 136M
|   ├── 270M
|   └── 9M
|
├── data                            - Datasets (need to be downloaded)
|   ├── eco_openings.csv
|   ├── test
|   ├── train
|   └── puzzles.csv
|
├── lc0                             - Leela Chess Zero (needs to be installed)
|
├── src
|   ├── engines
|   |   ├── constants.py            - Engine constants
|   |   ├── engine.py               - Engine interface
|   |   ├── lc0_engine.py           - Leela Chess Zero engine
|   |   ├── neural_engines.py       - Neural engines
|   |   └── stockfish_engine.py     - Stockfish engine
|   |
|   ├── bagz.py                     - Readers for our .bag data files
|   ├── config.py                   - Experiment configurations
|   ├── constants.py                - Constants, interfaces, and types
|   ├── data_loader.py              - Data loader
|   ├── metrics_evaluator.py        - Metrics (e.g., Kendall's tau) evaluator
|   ├── puzzles.py                  - Puzzle evaluation script
|   ├── searchless_chess.ipynb      - Model analysis notebook
|   ├── tokenizer.py                - Chess board tokenization
|   ├── tournament.py               - Elo tournament script
|   ├── train.py                    - Example training + evaluation script
|   ├── training.py                 - Training loop
|   ├── training_utils.py           - Training utility functions
|   ├── transformer.py              - Decoder-only Transformer
|   └── utils.py                    - Utility functions
|
├── Stockfish                       - Stockfish (needs to be installed)
|
├── README.md
└── requirements.txt                - Dependencies
```

## Dataset

We train and evaluate on **ChessBench**, a large-scale chess dataset introduced by Ruoss et al. (2024). ChessBench consists of 10 million human games sourced from [Lichess](https://lichess.org), annotated by Stockfish 16 with state-values, best actions, and action-values for all legal moves in each board state. The full dataset contains approximately 530 million board states and 15.3 billion action-value estimates, corresponding to roughly 8,864 days of unparallelized Stockfish evaluation time.

For our experiments, we train on a single shard of the action-value training split (~7 million state-action pairs) due to compute constraints. Evaluation is performed on the held-out test set and a curated set of 10,000 Lichess puzzles rated by Elo difficulty.

The dataset is publicly available at: https://storage.googleapis.com/searchless_chess/data/

## Acknowledgements ##

If you use ChessBench or build on the searchless chess framework, please cite the original work:

```bibtex
@inproceedings{ruoss2024amortized,
  author    = {Anian Ruoss and
               Gr{\'{e}}goire Del{\'{e}}tang and
               Sourabh Medapati and
               Jordi Grau{-}Moya and
               Li Kevin Wenliang and
               Elliot Catt and
               John Reid and
               Cannada A. Lewis and
               Joel Veness and
               Tim Genewein},
  title     = {Amortized Planning with Large-Scale Transformers:
               A Case Study on Chess},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024}
}
```
