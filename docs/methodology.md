\# Methodology



\## Overview



The proposed Cyber-Physical Context Fusion (PCCF) pipeline combines physical, device, and cyber information for anomaly detection in a cyber-physical environment.



The complete dataset is divided into training and testing data using an 80:20 split. The training data is used for preprocessing, encoder training, fusion-model training, and federated learning. The test data is used for evaluating the final model.



\## 1. Dataset Preparation



Three types of data are used in the pipeline:



\- Physical data (`Xp`)

\- Device data (`Xd`)

\- Cyber data (`Xc`)



The three modalities are aligned using their common samples/windows so that the corresponding physical, device, and cyber information belongs to the same sample.



\## 2. Train-Test Split



The data is divided into:



\- 80% training data

\- 20% testing data



The split is performed using `train\_test\_split` with:



\- `test\_size = 0.20`

\- `random\_state = 42`

\- Stratification based on the `GroundTruth` label



The same indices are used for the physical, device, and cyber data so that the three modalities remain aligned.



The training data is used for model development, while the test data is used for final evaluation.



\## 3. Preprocessing



The `Window\_Start` timestamp column is removed before training.



Missing values are replaced with zero.



Each modality is standardized separately using `StandardScaler`.



The scaler for each modality is fitted using only its training data. The same scaler is then applied to the corresponding test data.



\## 4. Modality-Specific Representation Learning



Three separate neural encoders are trained to obtain compact representations of the three modalities.



\### 4.1 Physical Encoder



The physical encoder is an autoencoder based on fully connected layers.



The encoder converts the physical feature vector into a 32-dimensional representation. The decoder reconstructs the original physical feature vector.



The model is trained using mean squared reconstruction error.



\### 4.2 Device Encoder



The device encoder uses a GRU-based architecture.



The GRU processes the device features and produces a hidden representation, which is converted into a 32-dimensional device embedding.



A decoder is used to reconstruct the original device features.



\### 4.3 Cyber Encoder



The cyber encoder uses one-dimensional convolutional layers followed by adaptive average pooling.



The resulting representation is converted into a 32-dimensional cyber embedding.



A decoder reconstructs the original cyber features.



\## 5. Embedding Generation



After training the three modality-specific encoders, their encoder outputs are extracted as:



\- `Cp` — physical embedding

\- `Cd` — device embedding

\- `Cy` — cyber embedding



Each embedding has 32 dimensions.



The embeddings are L2-normalized before they are used for fusion.



\## 6. Consistency Vector



A three-dimensional consistency vector is calculated using pairwise cosine similarity between the modality embeddings:



\- Physical–Device similarity

\- Physical–Cyber similarity

\- Device–Cyber similarity



Therefore, the consistency vector has 3 dimensions.



The complete fusion representation contains:



\- Physical embedding: 32 dimensions

\- Device embedding: 32 dimensions

\- Cyber embedding: 32 dimensions

\- Consistency vector: 3 dimensions



Therefore, the total fusion input has:



`32 + 32 + 32 + 3 = 99 dimensions`



\## 7. Cross-Modal Fusion



The three modality embeddings and the consistency vector are concatenated to form a 99-dimensional representation.



A learnable modality gate assigns weights to:



\- Physical information

\- Device information

\- Cyber information

\- Consistency information



The gate uses a softmax function to obtain normalized weights for the four components.



The weighted representations are then concatenated and passed to the fusion autoencoder.



\## 8. Fusion Autoencoder



The fusion network contains an autoencoder that takes the 99-dimensional fused representation and maps it to a lower-dimensional latent representation before reconstructing the input.



The reconstruction error is calculated for every sample using mean squared error between the fused representation and its reconstructed representation.



This reconstruction error is used as the anomaly score.



\## 9. Federated Learning



For the federated experiments, the training embeddings are divided among three clients.



Each client performs local training using its own training data. The raw client data is not sent to the central server.



The Flower framework is used to manage communication between the clients and the server.



All three clients participate in the federated training rounds in the implemented setup.



\## 10. FedProx



FedProx is used during local client training to reduce differences between the local model and the global model.



The local training objective consists of the reconstruction loss, the standard deviation component, and the FedProx regularization term.



The implementation uses:



\- `λ = 0.5` for the standard deviation component

\- `μ = 0.01` for the FedProx term



Gradient clipping is also applied during local training.



\## 11. Differential Privacy



Differential privacy is used in the full FL + DP implementation during local client training.



The implementation uses the Opacus `PrivacyEngine` with:



\- Noise multiplier = 0.8

\- Maximum gradient norm = 1.0

\- Delta = `1e-5` for privacy accounting



The privacy mechanism is applied during local model optimization.



\## 12. Anomaly Detection



For every test sample, the reconstruction error from the fusion autoencoder is used as the anomaly score.



The reconstruction errors are normalized before calculating ROC-AUC and selecting the classification threshold.



The threshold is selected by testing percentile-based thresholds and choosing the threshold that gives the highest F1-score.



\## 13. Evaluation



The model is evaluated using the test data.



The following metrics are calculated:



\- ROC-AUC

\- Accuracy

\- Precision

\- Recall

\- F1-score

\- Confusion Matrix



\## 14. Ablation Study



Four experiments are used to study the contribution of the main components of the proposed approach:



| Experiment | Configuration |

|---|---|

| E1 — No DP | Federated learning without differential privacy |

| E2 — No FL | Centralized training without federated learning or differential privacy |

| E3 — No Consistency | Federated configuration without the consistency vector |

| E4 — FL + DP | Full federated learning and differential privacy configuration |



The results of these experiments are compared using the same evaluation metrics to study the effect of federated learning, differential privacy, and the consistency vector on anomaly detection performance.

