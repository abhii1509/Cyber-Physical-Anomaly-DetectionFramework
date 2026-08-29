# Experiments

## Federated Learning

The fusion model is trained via Federated Learning across three clients. Each client receives the current global model parameters and performs local optimization on its own training partition; local data never leaves the client. After local training, each client sends updated parameters to the server, which aggregates them into the next global model.

Implementation: [Flower](https://flower.ai/) for federated orchestration, FedProx for local regularization.

```text
Number of clients     : 3
Communication rounds  : 10
Batch size            : 128
Local epochs          : 1
Learning rate         : 2 × 10⁻⁴
FedProx μ             : 0.01
```

## Differential Privacy

Differential Privacy is applied during local federated training using [Opacus](https://opacus.ai/).

```text
Noise multiplier      : 0.8
Maximum gradient norm : 1.0
Delta                 : 1 × 10⁻⁵
```

Gradient clipping limits the contribution of individual training examples; noise is added during private optimization. Privacy accounting uses the selected delta.

A single implementation supports both configurations:

```text
0 → Federated Learning without Differential Privacy
1 → Federated Learning with Differential Privacy
```

`run.py` prompts the user for this selection at runtime.

## Ablation Study

The ablation study evaluates the effect of Differential Privacy on the federated anomaly detection framework:

| Configuration | Federated Learning | Differential Privacy |
| ------------- | ------------------- | --------------------- |
| FL (No DP)    | Yes                  | No                     |
| FL + DP       | Yes                  | Yes                    |

Both configurations use the same architecture, data partition, federated client structure, and evaluation procedure. The only intended experimental difference is whether Differential Privacy is enabled during local training. This is implemented as one codebase in `experiments/` rather than maintained as separate copies.

## Experiment Files

| File          | Purpose                                                                                              |
| ------------- | ------------------------------------------------------------------------------------------------------ |
| `config.py`   | Experiment, training, path, and Differential Privacy settings                                          |
| `model.py`    | Modality gate, fusion autoencoder, parameter utilities, FedProx training, and evaluation                |
| `client.py`   | Loads a client partition and performs local federated training                                          |
| `server.py`   | Runs the Flower server and FedProx aggregation                                                          |
| `evaluate.py` | Evaluates the selected global model on the held-out evaluation set                                       |
| `run.py`      | Starts the server and clients; asks whether Differential Privacy should be enabled                       |

## Running the Experiment

Prerequisite: the processed data, client partitions, test representations, and initial fusion model must already be prepared (see the `scripts/` pipeline in the top-level [README](../README.md#quick-start) and [`dataset.md`](dataset.md#generated-data-not-committed)).

```bash
cd experiments
python run.py
```

```text
PCCF DIFFERENTIAL PRIVACY ABLATION STUDY

Select experiment:
1 - With Differential Privacy
0 - Without Differential Privacy

Enter choice [1/0]:
```

Enter `0` to run FL without DP, or `1` to run FL with DP. The selected configuration is passed automatically to the server and clients.

## Output Organization

Model checkpoints and result files for each configuration are kept separate:

```text
results/
└── metrics/
    ├── E1_No_DP/
    │   ├── global_models/
    │   │   ├── global_round_1.pth
    │   │   ├── ...
    │   │   └── global_round_10.pth
    │   └── E1_No_DP_results.pkl
    │
    └── E2_FL_DP/
        ├── global_models/
        │   ├── global_round_1.pth
        │   ├── ...
        │   └── global_round_10.pth
        └── E2_FL_DP_results.pkl
```

## Evaluation Metrics

Both threshold-independent and threshold-dependent measures are used:

- Accuracy, Precision, Recall, F1-score
- ROC-AUC
- Confusion matrix / classification report
- Reconstruction-error distributions
- ROC curve

ROC-AUC measures how well continuous reconstruction scores separate normal and anomalous observations independently of any threshold. For threshold-dependent metrics, candidate thresholds are swept over the reconstruction-error distribution and the threshold with the highest F1-score is selected.

## Reproducibility

Both ablation configurations use the same 80:20 partition and evaluation procedure. The centralized fusion model from `scripts/07_trainfusion.py` initializes federated training in both cases.

```text
FL (No DP)                          FL + DP

Client data                         Client data
    ↓                                    ↓
Local FedProx training              Private local FedProx training
    ↓                                    ↓
Client model updates                Gradient clipping + noise
    ↓                                    ↓
FedProx aggregation                 Client model updates
    ↓                                    ↓
Global model                        FedProx aggregation
    ↓                                    ↓
Evaluation                          Global model
                                          ↓
                                     Evaluation
```

See the top-level [README](../README.md#results) for the reported results table and confusion matrix.
