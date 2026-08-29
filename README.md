# PCCF-GLM

## Privacy-Preserving Cross-Layer Cyber-Physical Anomaly Detection Using Federated Learning and Differential Privacy for Gradient Leakage Mitigation

PCCF-GLM is a cyber-physical anomaly detection framework designed for smart-home environments. The framework combines information from the physical, device, and cyber layers to identify abnormal system behaviour while supporting privacy-preserving model training.

The three modalities are processed independently to obtain compact representations. Cross-layer consistency is then calculated between the modality representations. These representations are combined using a learnable modality gate and passed through a fusion autoencoder. The reconstruction error produced by the autoencoder is used as the anomaly score.

Federated Learning is used to train the fusion model across multiple clients without directly sharing their local training data. FedProx is incorporated into local optimization to reduce divergence between client models. Differential Privacy is additionally evaluated using gradient clipping, noise addition, and privacy accounting.

---

## Framework Overview

The framework uses three sources of information from the smart-home environment:

* Physical sensor data
* Device state data
* Network traffic data

Each modality is processed separately and converted into a compact 32-dimensional representation.

The resulting modality embeddings are:

* `Cp` — physical embedding
* `Cd` — device embedding
* `Cy` — cyber embedding

Pairwise similarities between these embeddings are used to represent cross-layer consistency:

* `Cp-Cd`
* `Cp-Cy`
* `Cd-Cy`

These three values form the consistency representation `CS`.

The final multimodal representation is therefore:

```text
Cp (32) + Cd (32) + Cy (32) + CS (3) = 99 dimensions
```

This 99-dimensional representation is passed through a learnable modality gate before being processed by the fusion autoencoder.

---

## System Architecture

The overall processing flow consists of:

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

The local multimodal processing pipeline can be summarized as:

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

The federated training process extends this local pipeline across three clients:

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

---

## Data Processing Pipeline

The implementation follows an 80:20 training and evaluation protocol.

The preprocessing pipeline performs the following steps:

1. Prepare the physical, device, and cyber data.
2. Inspect and clean the individual data sources.
3. Recover required information from corresponding raw files when missing data are identified.
4. Synchronize the three modalities using their common timeline.
5. Create the common multimodal representation.
6. Split the synchronized observations into training and testing partitions.
7. Train the modality-specific encoders.
8. Generate physical, device, and cyber embeddings.
9. Calculate cross-layer consistency.
10. Prepare the final representation for multimodal fusion.
11. Create the federated client partitions.
12. Train and evaluate the fusion model.

The main implementation pipeline is contained in the `scripts/` directory.

---

## Modality Representations

The final synchronized multimodal dataset contains 40,802 observations represented by:

```text
Physical features : 572
Device features   : 2,780
Cyber features    : 12
```

Modality-specific deep learning models are used to obtain compact 32-dimensional representations.

The physical, device, and cyber representations are then used to construct the cross-layer consistency vector.

The resulting representation is:

```text
Physical embedding     : Cp (32)
Device embedding       : Cd (32)
Cyber embedding        : Cy (32)
Consistency vector     : CS (3)

Final representation   : 99 dimensions
```

---

## Cross-Layer Consistency

Cross-layer consistency is explicitly modelled using pairwise similarities between the modality embeddings.

The three pairwise relationships are:

```text
Cp-Cd
Cp-Cy
Cd-Cy
```

Together they form:

```text
CS = [Cp-Cd, Cp-Cy, Cd-Cy]
```

The consistency representation is concatenated with the three modality embeddings:

```text
[Cp | Cd | Cy | CS]
```

This provides the fusion stage with both modality-specific information and information describing the relationships between the different layers of the cyber-physical environment.

---

## Learnable Modality Gate

The 99-dimensional representation is passed through a learnable modality gate.

The gate uses a fully connected layer followed by a softmax operation to obtain four weights.

The four weights correspond to:

* Physical representation
* Device representation
* Cyber representation
* Cross-layer consistency representation

The weighted components are concatenated and passed to the fusion autoencoder.

This allows the model to learn the relative contribution of the different components during training instead of assigning fixed weights to the modalities.

---

## Fusion Autoencoder

The fusion stage uses an autoencoder with the following architecture:

```text
99 → 64 → 32 → 16 → 32 → 64 → 99
```

The encoder compresses the 99-dimensional representation into a 16-dimensional latent representation.

The decoder reconstructs the original 99-dimensional representation.

For each observation, reconstruction error is calculated as the mean squared error between the fused representation and its reconstruction:

```text
MSE = mean((reconstruction - fusion representation)²)
```

The reconstruction error is used as the anomaly score.

A larger reconstruction error indicates that the observation differs more strongly from the behaviour learned by the model.

---

## Training Objective

The local training objective combines the average reconstruction error with the dispersion of reconstruction errors.

The loss used during training is:

```text
Loss = Mean(MSE) + 0.5 × Std(MSE)
```

FedProx regularization is then added during federated optimization to constrain local model updates with respect to the current global model.

The FedProx coefficient used in the implementation is:

```text
μ = 0.01
```

---

## Federated Learning

The fusion model is trained using Federated Learning across three clients.

Each client receives the current global model parameters and performs local optimization using its own training partition. The local data remain on the client and are not sent to the aggregation server.

After local training, the client sends its updated model parameters to the server. The server aggregates the client updates to produce the next global model.

The implementation uses Flower for the federated learning process and FedProx for local regularization.

The main federated configuration is:

```text
Number of clients : 3
Communication rounds : 10
Batch size : 128
Local epochs : 1
Learning rate : 2 × 10⁻⁴
FedProx μ : 0.01
```

---

## Differential Privacy

Differential Privacy is applied during local federated training using Opacus.

The private training configuration uses:

```text
Noise multiplier     : 0.8
Maximum gradient norm: 1.0
Delta                : 1 × 10⁻⁵
```

Gradient clipping limits the contribution of individual training examples, while noise is introduced during private optimization. Privacy accounting is performed using the selected delta value.

The repository uses a single federated experiment implementation for both configurations:

```text
0 → Federated Learning without Differential Privacy
1 → Federated Learning with Differential Privacy
```

The experiment runner automatically asks the user which configuration should be executed.

---

## Ablation Study

The ablation study evaluates the effect of Differential Privacy on the federated anomaly detection framework.

Two configurations are compared:

| Configuration | Federated Learning | Differential Privacy |
| ------------- | ------------------ | -------------------- |
| FL (No DP)    | Yes                | No                   |
| FL + DP       | Yes                | Yes                  |

The same multimodal architecture, data partition, federated client structure, and evaluation procedure are used for the comparison.

The only intended experimental difference is whether Differential Privacy is enabled during local training.

The experiment implementation is contained in the `experiments/` directory rather than maintaining separate copies of the model and training code.

---

## Dataset

The experiments use the **Cyber-Physical Anomaly Detection in Smart Homes** dataset introduced by Yasar Majib, Mohammed Alosaimi, Andre Asaturyan, and Charith Perera.

The dataset combines information from network traffic, smart devices, and environmental sensors collected in a smart-home environment. The original dataset contains both individual source datasets and a merged timeline intended for cyber-physical analysis. 

The dataset was collected over approximately four weeks and includes activity from two actors. The primary actor represents normal activity, while the second actor was introduced for a shorter period and can be used for anomalous behaviour analysis. 

The original dataset includes physical sensor measurements, network traffic, and smart-device information. 

---

## Data Preparation and Missing Cyber Data

The original dataset provides raw and processed representations of the different data sources. 

During preparation of the data for this project, missing portions were identified in the available cyber master data. The corresponding raw cyber files were obtained, inspected, and used to recover the required records.

The recovered information was converted and merged using the corresponding master-data structure and field format before being used in the project preprocessing pipeline.

This additional preparation was required to obtain a consistent cyber representation for synchronization with the physical and device data.

The original cyber dataset itself is based on hourly packet-capture files. The dataset documentation describes converting these `.pcap` files to CSV, adding timestamps, joining the resulting files, and normalizing the resulting data. 

The physical dataset also contains documented capture-related issues. The original dataset processing includes cleaning and repairing affected physical sensor values. 

---

## Data Used in the Implementation

The project uses information from the following data categories.

### Physical Data

The physical component uses the physical sensor data and its processed representations.

The relevant files include:

```text
SensorData.csv

SensorDataMasterTransposed.csv

PhysicalData_Clean_1.csv
PhysicalData_Clean_2.csv
PhysicalData_Clean_3.csv
PhysicalData_Clean_4.csv
PhysicalData_Clean_5.csv

PhysicalData_Clean_Repaired_1.csv
PhysicalData_Clean_Repaired_2.csv
PhysicalData_Clean_Repaired_3.csv
PhysicalData_Clean_Repaired_4.csv
PhysicalData_Clean_Repaired_5.csv
```

These files correspond to the raw, master, cleaned, and repaired physical representations available during data preparation. 

### Device Data

The device component uses the smart-device data represented through the CU and BRE sources.

The main master representations are:

```text
CUMaster
BREMaster
```

Supporting device information and mappings are also used during preparation and synchronization.

### Cyber Data

The cyber component uses the network-traffic data and the corresponding cyber/cyber-physical representations.

The available raw cyber/cyber-physical sources include:

```text
BRECyPhy_W1
BRECyPhy_W2
BRECyPhy_W3
BRECyPhy_W4
BRECyPhy_W5

CUCyPhy_1711_Latest
CUCyPhy_141122-1628
CUCyPhy_151122-1740
CUCyPhy_161122-1615
CUCyPhy_281022

CUCyPhy_states1
CUCyPhy_states2
CUCyPhy_states3
CUCyPhy_states4
```

The corresponding cyber/device master representations include:

```text
BREMaster
CUMaster
```

The cyber data are ultimately transformed into the feature representation used by the cyber encoder. The original dataset describes packet-level fields including frame number, timestamp, source, destination, protocol, packet length, and packet information. 

### Supporting Metadata

Supporting metadata used during data preparation include:

```text
ActivityLabels.json
ActivityLogR1.csv
ActivityLogR2.csv

BRE_CyPhyDevices.json
BRE_Devices_Location_Sensors.json

CU_CyPhyDevices.json
CUStatesMap.json

CyberDevices.json
PIDMap.json
ProtocolMap.json

SmartDevices_Data_and_Frequency.csv
```

These files provide activity information, device mappings, protocol mappings, and other information required to interpret and synchronize the source data. 

---

## Generated Data Used by the Experiments

The preprocessing pipeline generates the synchronized representations used by the learning stages.

The main synchronized feature files are:

```text
Xp_common.csv
Xd_common.csv
Xc_common.csv
```

These represent the synchronized physical, device, and cyber feature spaces.

The later stages generate the compact representations used by the fusion model:

```text
Cp
Cd
Cy
CS
```

The federated experiment uses client partitions and test representations derived from these processed datasets.

The main generated files required by the federated experiment are:

```text
client_1.pkl
client_2.pkl
client_3.pkl

Cp_test.pkl
Cd_test.pkl
Cy_test.pkl
CS_test.pkl

TestLabels.csv
```

These generated files are not committed to the repository.

---

## Evaluation Protocol

The synchronized dataset contains 40,802 observations.

The evaluation uses a separate 20% partition that was not used during federated model training.

The resulting partition contains:

```text
Training observations : 32,641
Evaluation observations: 8,161
```

The evaluation set contains:

```text
Normal observations    : 7,297
Anomalous observations : 864
```

Anomalous observations therefore represent approximately 10.6% of the evaluation set.

The trained global model produces a reconstruction error for every evaluation observation. These continuous reconstruction errors are used as anomaly scores.

---

## Evaluation Metrics

The model is evaluated using both threshold-independent and threshold-dependent measures.

The evaluation includes:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrix
* Classification report
* Reconstruction-error distributions
* ROC curve

ROC-AUC is used to examine the ability of the continuous reconstruction scores to distinguish normal and anomalous observations independently of a particular classification threshold.

For threshold-dependent evaluation, candidate thresholds are examined using the reconstruction-error distribution and the threshold producing the highest F1-score is selected.

---

## Results

The final reported comparison between the two federated configurations is:

| Configuration | Accuracy | Precision | Recall | F1-Score |    ROC-AUC |
| ------------- | -------: | --------: | -----: | -------: | ---------: |
| FL (No DP)    |   0.6640 |    0.1164 | 0.3299 |   0.1721 |     0.6145 |
| FL + DP       |   0.6768 |    0.1371 | 0.3877 |   0.2025 | **0.6622** |

For the reported FL + DP configuration, the ROC-AUC is **0.6622**.

The corresponding confusion matrix is:

```text
[[5188, 2109],
 [ 529,  335]]
```

This gives:

```text
True Negatives  : 5188
False Positives : 2109
False Negatives : 529
True Positives  : 335
```

For the anomaly class, the selected threshold gives a precision of approximately 0.1371 and a recall of approximately 0.3877, resulting in an F1-score of 0.2025.

The overlap between normal and anomalous reconstruction errors results in both false-positive and false-negative predictions. The ROC-AUC of 0.6622 indicates that the continuous anomaly scores provide useful discrimination between the two classes, although the classes are not completely separable.

---

## Repository Structure

```text
Cyber-Physical-Anomaly-DetectionFramework/
│
├── data/
│   └── .gitkeep
│
├── docs/
│   ├── dataset.md
│   ├── experiments.md
│   └── methodology.md
│
├── experiments/
│   ├── client.py
│   ├── config.py
│   ├── evaluate.py
│   ├── model.py
│   ├── run.py
│   └── server.py
│
├── models/
│   └── .gitkeep
│
├── results/
│   └── .gitkeep
│
├── scripts/
│   ├── 01_create_8020split.py
│   ├── 02_preprocess.py
│   ├── 03_phyencoder.py
│   ├── 04_devicencoder.py
│   ├── 05_cyberencoder.py
│   ├── 06_embeddings.py
│   ├── 07_trainfusion.py
│   ├── 08_createclients.py
│   ├── client.py
│   ├── evaluate.py
│   ├── model.py
│   └── server.py
│
├── .gitignore
└── README.md
```

The `scripts/` directory contains the main data preparation, encoding, embedding, fusion-model, and client-generation pipeline.

The `experiments/` directory contains the federated learning implementation and the Differential Privacy ablation study.

The `data/`, `models/`, and `results/` directories are retained in the repository structure, while generated datasets and trained model checkpoints are excluded from version control.

---

## Pipeline Scripts

The main preprocessing and training pipeline is divided into the following stages:

| Script                   | Purpose                                                       |
| ------------------------ | ------------------------------------------------------------- |
| `01_create_8020split.py` | Creates the 80:20 data partition                              |
| `02_preprocess.py`       | Preprocesses and synchronizes the source data                 |
| `03_phyencoder.py`       | Processes the physical modality                               |
| `04_devicencoder.py`     | Processes the device modality                                 |
| `05_cyberencoder.py`     | Processes the cyber modality                                  |
| `06_embeddings.py`       | Generates modality embeddings and cross-layer representations |
| `07_trainfusion.py`      | Trains the centralized fusion model used as the initial model |
| `08_createclients.py`    | Creates the federated client partitions                       |

These scripts represent the main project pipeline and are separate from the federated Differential Privacy ablation implementation.

---

## Federated Experiment Files

The `experiments/` directory contains a single implementation that supports both the DP and No-DP configurations.

| File          | Purpose                                                                                              |
| ------------- | ---------------------------------------------------------------------------------------------------- |
| `config.py`   | Stores experiment, training, path, and Differential Privacy settings                                 |
| `model.py`    | Defines the modality gate, fusion autoencoder, parameter utilities, FedProx training, and evaluation |
| `client.py`   | Loads a client partition and performs local federated training                                       |
| `server.py`   | Runs the Flower server and FedProx aggregation                                                       |
| `evaluate.py` | Evaluates the selected global model on the held-out evaluation set                                   |
| `run.py`      | Starts the server and clients and asks whether Differential Privacy should be enabled                |

The same implementation is used for both ablation configurations. The selected configuration determines whether Opacus-based Differential Privacy is enabled during client training.

---

## Running the Federated Experiment

After the required processed data, client partitions, test representations, and initial fusion model have been prepared, the federated experiment can be started from the `experiments/` directory:

```bash
python run.py
```

The program asks the user to select the configuration:

```text
PCCF DIFFERENTIAL PRIVACY ABLATION STUDY

Select experiment:
1 - With Differential Privacy
0 - Without Differential Privacy

Enter choice [1/0]:
```

Enter:

```text
0
```

to run Federated Learning without Differential Privacy.

Enter:

```text
1
```

to run Federated Learning with Differential Privacy.

The selected configuration is passed automatically to the server and clients.

---

## Output Organization

The experiment configuration keeps the two runs separate so that their model checkpoints and result files are not mixed.

The output structure is:

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
    └── E4_FL_DP/
        ├── global_models/
        │   ├── global_round_1.pth
        │   ├── ...
        │   └── global_round_10.pth
        └── E4_FL_DP_results.pkl
```

The DP and No-DP configurations therefore use separate experiment-specific output directories while sharing the same implementation.

---

## Reproducibility

The reported comparison uses the same 80:20 data partition and evaluation procedure for both federated configurations.

The training pipeline first prepares the synchronized multimodal representations and creates the client partitions. The centralized fusion model produced by the main pipeline is then used as the initial model for federated training.

The two ablation configurations differ in whether Differential Privacy is enabled during local optimization.

```text
FL (No DP)

Client data
    ↓
Local FedProx training
    ↓
Client model updates
    ↓
FedProx aggregation
    ↓
Global model
    ↓
Evaluation
```

```text
FL + DP

Client data
    ↓
Private local FedProx training
    ↓
Gradient clipping + noise
    ↓
Client model updates
    ↓
FedProx aggregation
    ↓
Global model
    ↓
Evaluation
```

The evaluation procedure remains the same for both configurations.

---

## Data Availability

The original **Cyber-Physical Anomaly Detection in Smart Homes** dataset is publicly available through the repository associated with the original dataset publication. The dataset itself is not redistributed in this project.

The repository does not include the original raw data, generated synchronized datasets, serialized client partitions, or trained model checkpoints.

The data used by the implementation must therefore be obtained and prepared separately before running the complete pipeline.

The original dataset publication provides both raw and processed datasets and describes the data collection, cleaning, normalization, and merging procedures. 

During preparation for this project, missing information was identified in the available cyber master data. The corresponding raw cyber files were inspected and used to recover the required records. The recovered records were then converted and merged into the corresponding master-data format so that the cyber data could be synchronized with the physical and device modalities.

Generated files such as the synchronized datasets, modality embeddings, client partitions, test representations, and trained model checkpoints are excluded from version control.

---

## Documentation

Additional documentation is available in the `docs/` directory.

| Document                                     | Description                                             |
| -------------------------------------------- | ------------------------------------------------------- |
| [`docs/dataset.md`](docs/dataset.md)         | Dataset sources, structure, and preparation             |
| [`docs/methodology.md`](docs/methodology.md) | Detailed framework methodology and processing pipeline  |
| [`docs/experiments.md`](docs/experiments.md) | Federated Learning and Differential Privacy experiments |

---

## Requirements

The implementation uses Python and the following major libraries:

* PyTorch
* Flower
* Opacus
* NumPy
* Pandas
* Scikit-learn
* Joblib

A compatible Python environment with these dependencies is required to run the preprocessing and federated experiments.

---

## Citation

The dataset used in this project was introduced by Majib et al.:

```text
Majib, Y., Alosaimi, M., Asaturyan, A., and Perera, C.
Dataset for cyber–physical anomaly detection in smart homes.
Frontiers in the Internet of Things, 2023.
DOI: 10.3389/friot.2023.1275080
```

The dataset publication describes the smart-home cyber, physical, and smart-device sources and their unified timeline. 

---

## License

This repository is intended for research and academic use.
