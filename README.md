# DS_cone

## Introduction

DS_cone is a Lorentz-hyperbolic knowledge graph completion model built on the FlorE backbone. In addition to Lorentz relation transformations and scoring, it introduces a semantic-guided Dynamic Sequential Lorentz Entailment Cone for dynamic semantic cone constraints, multi-hop neighborhood reasoning, boundary-constrained path regularization, and Lorentz-constrained message aggregation.

The default training entrypoint is `main.py`. During training, the model selects the best checkpoint on the validation split and reports filtered `MRR`, `Hits@1`, `Hits@3`, and `Hits@10` on the test split.

## Module Directory

```text
DS_cone/
├── main.py                    # CLI entrypoint for training and evaluation
├── Experiment.py              # Training loop, validation, testing, logging, checkpointing
├── LorentzModel.py            # FlorE backbone, semantic cone reasoning, scoring function
├── SemanticConeProjector.py   # Dynamic semantic cone aperture and direction module
├── load_data.py               # Data loading, reverse-relation augmentation, entity/relation indexing
├── case.py                    # Case-study path visualization from logs or checkpoints
├── requirements.txt           # Python dependencies exported from the project environment
├── best_fb15k.sh              # Best-configuration launcher for FB15k-237
├── best_wn18rr.sh             # Best-configuration launcher for WN18RR
├── run_codexm.sh              # CoDEx-M launcher
├── run_codexl.sh              # CoDEx-L launcher
├── data/                      # Dataset folders: FB15k-237, wn18rr, codex-m, codex-l
├── manifolds/                 # Lorentz manifold operations and math utilities
└── optim/                     # Riemannian optimizers
```

## Installation

Create or activate a Python 3.11 environment, then install dependencies:

```bash
pip install -r requirements.txt
```

To use the existing local environment, pass the Python interpreter through `PYTHON`:

```bash
PYTHON="/warehouse/ruinanli_resource/py311/bin/python" ./best_fb15k.sh
```

The GPU device can be overridden through `DEVICE`:

```bash
PYTHON="/warehouse/ruinanli_resource/py311/bin/python" DEVICE=cuda:1 ./best_wn18rr.sh
```

## Quick Start

Train and evaluate on FB15k-237:

```bash
./best_fb15k.sh
```

Train and evaluate on WN18RR:

```bash
./best_wn18rr.sh
```

Train and evaluate on CoDEx-M or CoDEx-L:

```bash
./run_codexm.sh
./run_codexl.sh
```

By default, training logs and checkpoints are written under `logs/`. Use `--no_log` or `--no_checkpoint` to disable them.

## Best Results

This repository includes best-configuration launchers for each dataset, but no committed final result logs were found. The table below records the best available reproduction configurations. Fill in the actual `MRR` and `Hits@K` values after running the corresponding script and reading the printed `Test Result`.

| Dataset | Launcher | Dim | Epochs | Batch | Negatives | LR | Valid Step | Best MRR | Hits@1 | Hits@3 | Hits@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| FB15k-237 | `best_fb15k.sh` | 256 | 200 | 512 | 100 | 0.00717548 | 25 | TBD | TBD | TBD | TBD |
| WN18RR | `best_wn18rr.sh` | 256 | 200 | 512 | 200 | 0.0343 | 50 | TBD | TBD | TBD | TBD |
| CoDEx-M | `run_codexm.sh` | 256 | 200 | 512 | 200 | 0.005 | 25 | TBD | TBD | TBD | TBD |
| CoDEx-L | `run_codexl.sh` | 256 | 200 | 512 | 200 | 0.05 | 25 | TBD | TBD | TBD | TBD |

Test metrics are printed in this format:

```text
Test Result
Best it:<epoch>
Hit@10:<value>
Hit@3:<value>
Hit@1:<value>
MRR:<value>
PrunedNodes:<value>
```

## Main Options

Common training options:

- `--dataset`: dataset name under `data/`, such as `FB15k-237`, `wn18rr`, `codex-m`, or `codex-l`.
- `--dim`: embedding dimensionality.
- `--nneg`: number of negative samples.
- `--lr`: learning rate.
- `--optimizer`: optimizer, one of `radam`, `rsgd`, or `adam`.
- `--valid_steps`: validation interval.
- `--device`: PyTorch device, such as `cuda:0` or `cpu`.

Semantic cone and multi-hop reasoning options:

- `--use_projector`: enable the semantic cone projector.
- `--no_logic_cone`: disable semantic logic cone guidance.
- `--static_cone`: freeze the dynamic cone aperture/axis for ablation.
- `--no_bc_regularization`: disable Boundary-Constrained Path Regularization.
- `--no_an_modulation`: disable Adaptive Neighborhood Modulation.
- `--no_lc_aggregation`: disable Lorentz-Constrained Message Aggregation.
- `--no_multi_hop`: disable multi-hop reasoning.
- `--K_max`: maximum hop depth.
- `--agg_lambda`: residual coefficient for neighborhood aggregation.

## Data Format

Each dataset folder should contain:

```text
train.txt
valid.txt
test.txt
entities.dict
relations.dict
relation_names.txt
```

Triple files use whitespace-separated `(head, relation, tail)` rows. `load_data.py` automatically augments each split with reverse relations by appending `_reverse` to relation names.
