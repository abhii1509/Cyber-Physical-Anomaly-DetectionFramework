## Privacy-Preserving Cross-Layer Anomaly Detection in Smart Homes Using Federated Learning with Gradient Leakage Mitigation

Anomaly detection in smart-home environments by fusing physical, device, and network (cyber) sensor data. The three modalities are encoded independently, compared for cross-layer consistency, fused through an autoencoder, and scored by reconstruction error. The fusion model is trained via Federated Learning across simulated clients, with an optional Differential Privacy mode for gradient leakage mitigation.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Method](#method)
- [Federated Learning & Differential Privacy](#federated-learning--differential-privacy)
- [Results](#results)
- [Dataset](#dataset)
- [Reproducibility](#reproducibility)
- [Requirements](#requirements)
- [Citation](#citation)
- [License](#license)

## Overview

Three modalities — physical sensors, device state, and network traffic — are each encoded into a 32-dimensional representation (`Cp`, `Cd`, `Cy`). Pairwise similarities between these embeddings form a 3-dimensional consistency vector `CS`, giving a 99-dimensional fused representation that passes through a learnable modality gate into a fusion autoencoder. Reconstruction error is the anomaly score. Training uses Federated Learning (FedProx) across 3 clients, with an optional Differential Privacy mode (Opacus) evaluated as an ablation.

Full methodology → [`docs/methodology.md`](docs/methodology.md)

## Quick Start

```bash
git clone https://github.com/abhii1509/Cyber-Physical-Anomaly-DetectionFramework.git
cd Cyber-Physical-Anomaly-DetectionFramework
pip install -r requirements.txt

# 1. Prepare data and train modality encoders
python scripts/01_create_8020split.py
python scripts/02_preprocess.py
python scripts/03_phyencoder.py
python scripts/04_devicencoder.py
python scripts/05_cyberencoder.py
python scripts/06_embeddings.py
python scripts/07_trainfusion.py
python scripts/08_createclients.py

# 2. Run the federated experiment
cd experiments
python run.py
```

`run.py` prompts you to select the configuration:

```text
Select experiment:
1 - With Differential Privacy
0 - Without Differential Privacy
```

Full data-preparation prerequisites → [`docs/dataset.md`](docs/dataset.md)

## Repository Structure

```text
Cyber-Physical-Anomaly-DetectionFramework/
│   .gitignore
│   LICENSE
│   README.md
│   requirements.txt
│
├── data/                 # raw/processed data (generated files not committed)
├── docs/
│   ├── dataset.md        # data sources, preparation, generated files
│   ├── methodology.md    # full architecture and training objective
│   └── experiments.md    # FL/DP config, ablation design, output layout
├── experiments/
│   ├── config.py         # experiment, training, path, DP settings
│   ├── model.py           # modality gate, fusion AE, FedProx training/eval
│   ├── client.py          # loads a client partition, local FL training
│   ├── server.py          # Flower server, FedProx aggregation
│   ├── evaluate.py        # evaluates global model on held-out set
│   └── run.py             # starts server/clients, DP on/off prompt
├── figures/               # plots and diagrams used in docs/paper
    ├── Architecture.png
    ├── ConfusionMatrix_8020.png
    ├── ErrorDistribution_8020.png
    ├── ROC_8020.png
├── models/                # trained checkpoints (not committed)
├── results/               # metrics and figures (not committed)
└── scripts/
    ├── 01_create_8020split.py
    ├── 02_preprocess.py
    ├── 03_phyencoder.py
    ├── 04_devicencoder.py
    ├── 05_cyberencoder.py
    ├── 06_embeddings.py
    ├── 07_trainfusion.py
    ├── 08_createclients.py
    ├── client.py
    ├── evaluate.py
    ├── model.py
    └── server.py
```

## Method

- **Encoding:** physical, device, and cyber data are each compressed to a 32-D embedding.
- **Cross-layer consistency:** pairwise similarities `Cp–Cd`, `Cp–Cy`, `Cd–Cy` form a 3-D vector `CS`.
- **Fusion:** `[Cp | Cd | Cy | CS]` (99-D) passes through a learnable modality gate, then a `99 → 64 → 32 → 16 → 32 → 64 → 99` autoencoder.
- **Anomaly score:** per-observation reconstruction MSE.

Full architecture diagrams and training-objective details → [`docs/methodology.md`](docs/methodology.md)

## Federated Learning & Differential Privacy

- **FL:** 3 clients, FedProx aggregation, 10 communication rounds, local epochs = 1, μ = 0.01.
- **DP (optional):** Opacus, noise multiplier 0.8, max gradient norm 1.0, δ = 1×10⁻⁵.
- **Ablation:** same architecture, data partition, and evaluation procedure; only DP on/off differs.

Full experiment configuration and output layout → [`docs/experiments.md`](docs/experiments.md)

## Results

Evaluation on the held-out 20% partition (8,161 observations; 864 anomalous, ~10.6%):

| Configuration | Accuracy | Precision | Recall | F1-Score |    ROC-AUC |
| ------------- | -------: | --------: | -----: | -------: | ---------: |
| FL (No DP)    |   0.6640 |    0.1164 | 0.3299 |   0.1721 |     0.6145 |
| FL + DP       |   0.6768 |    0.1371 | 0.3877 |   0.2025 | **0.6622** |

Confusion matrix (FL + DP):

```text
[[5188, 2109],
 [ 529,  335]]
```

The ROC-AUC of 0.6622 indicates the continuous reconstruction-error scores provide useful, though imperfect, discrimination between normal and anomalous behaviour.

## Dataset

Experiments use the **Cyber-Physical Anomaly Detection in Smart Homes** dataset (Majib et al., 2023), combining network traffic, smart-device, and environmental sensor data collected over ~4 weeks in a smart-home environment. The dataset is **not redistributed** in this repository — see [`docs/dataset.md`](docs/dataset.md) for sources, preparation steps, and required generated files.

## Reproducibility

Both ablation configurations use the same 80:20 data partition, client structure, and evaluation procedure. The centralized fusion model produced by `scripts/07_trainfusion.py` is used as the initial model for federated training in both cases; only the DP setting during local optimization differs.

## Requirements

Python 3.x with:

- PyTorch
- Flower
- Opacus
- NumPy
- Pandas
- Scikit-learn
- Joblib

See [`requirements.txt`](requirements.txt) for pinned versions.

## Citation

```text
Majib, Y., Alosaimi, M., Asaturyan, A., and Perera, C.
Dataset for cyber–physical anomaly detection in smart homes.
Frontiers in the Internet of Things, 2023.
DOI: 10.3389/friot.2023.1275080
```

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file in this repository.
