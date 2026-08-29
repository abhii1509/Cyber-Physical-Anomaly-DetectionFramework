# Methodology

## Framework Overview

PCCF-GLM combines information from the physical, device, and cyber layers of a smart-home environment to identify abnormal system behaviour while supporting privacy-preserving model training.

The three modalities are processed independently to obtain compact representations. Cross-layer consistency is then calculated between the modality representations. These representations are combined using a learnable modality gate and passed through a fusion autoencoder. The reconstruction error produced by the autoencoder is used as the anomaly score.

Federated Learning trains the fusion model across multiple clients without directly sharing local training data. FedProx is incorporated into local optimization to reduce divergence between client models. Differential Privacy is additionally evaluated using gradient clipping, noise addition, and privacy accounting (see [`experiments.md`](experiments.md)).

## System Architecture

Processing flow:

1. Physical, device, and cyber data preparation
2. Individual modality encoding
3. Cross-layer consistency calculation
4. Cross-layer feature concatenation
5. Learnable modality weighting
6. Fusion autoencoder
7. Reconstruction-error based anomaly scoring
8. Local federated optimization
9. FedProx aggregation
10. Differential Privacy during private training
11. Global model evaluation

Local multimodal processing pipeline:

```text
Physical Sensor Data ──→ Physical Encoder ──→ Cp ──┐
                                                    │
Device State Data ─────→ Device Encoder ────→ Cd ──┤
                                                    ├──→ Cross-Layer
Network Traffic ───────→ Cyber Encoder ─────→ Cy ──┤    Consistency
                                                    │
Cp, Cd, Cy ─────────────────────────────────────────┘
                         │
                         ↓
              Consistency Vector CS
                         │
                         ↓
             99-D Cross-Layer Representation
                         │
                         ↓
                Learnable Modality Gate
                         │
                         ↓
                 Fusion Autoencoder
                         │
                         ↓
                Reconstruction Error
                         │
                         ↓
                  Anomaly Score
```

Federated training extends this local pipeline across three clients:

```text
                         Global Model
                              │
                    Global Model Parameters
                              │
              ┌───────────────┼───────────────┐
              ↓               ↓               ↓
           Client 1        Client 2        Client 3
              │               │               │
        Local Training  Local Training  Local Training
              │               │               │
              └───────────────┼───────────────┘
                              ↓
                       FedProx Aggregation
                              │
                              ↓
                         Global Model
```

## Data Processing Pipeline

The implementation follows an 80:20 training/evaluation protocol:

1. Prepare the physical, device, and cyber data.
2. Inspect and clean the individual data sources.
3. Recover required information from raw files when missing data is identified.
4. Synchronize the three modalities on their common timeline.
5. Create the common multimodal representation.
6. Split into training and testing partitions.
7. Train the modality-specific encoders.
8. Generate physical, device, and cyber embeddings.
9. Calculate cross-layer consistency.
10. Prepare the final representation for multimodal fusion.
11. Create the federated client partitions.
12. Train and evaluate the fusion model.

This pipeline is implemented in `scripts/` (see the mapping in the [top-level README](../README.md#repository-structure)).

## Modality Representations

Each modality is encoded to a 32-dimensional representation:

```text
Cp — physical embedding (32)
Cd — device embedding   (32)
Cy — cyber embedding    (32)
```

Pairwise similarities between embeddings form the consistency vector:

```text
CS = [Cp-Cd, Cp-Cy, Cd-Cy]   (3)
```

Final fused representation:

```text
Cp (32) + Cd (32) + Cy (32) + CS (3) = 99 dimensions
```

## Cross-Layer Consistency

Cross-layer consistency is explicitly modelled using pairwise similarities between the modality embeddings, concatenated with the embeddings themselves:

```text
[Cp | Cd | Cy | CS]
```

This gives the fusion stage both modality-specific information and information describing relationships between the physical, device, and cyber layers.

## Learnable Modality Gate

The 99-dimensional representation passes through a learnable gate: a fully connected layer followed by softmax, producing four weights corresponding to the physical, device, cyber, and consistency components. The weighted components are concatenated and passed to the fusion autoencoder — letting the model learn relative contribution per component during training rather than using fixed weights.

## Fusion Autoencoder

```text
99 → 64 → 32 → 16 → 32 → 64 → 99
```

The encoder compresses the 99-D representation to a 16-D latent space; the decoder reconstructs the original 99-D representation. Reconstruction error per observation:

```text
MSE = mean((reconstruction - fusion representation)²)
```

Reconstruction error is the anomaly score — larger error indicates greater deviation from learned normal behaviour.

## Training Objective

Local training objective combines mean reconstruction error with its dispersion:

```text
Loss = Mean(MSE) + 0.5 × Std(MSE)
```

FedProx regularization is added during federated optimization to constrain local updates relative to the current global model:

```text
μ = 0.01
```

See [`experiments.md`](experiments.md) for full FL/DP configuration and the ablation design.
