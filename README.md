\# PCCF-GLM



\## Privacy-Preserving Cross-Layer Cyber-Physical Anomaly Detection with Gradient Leakage Mitigation



This repository contains the implementation of our cyber-physical anomaly detection framework for smart-home environments.



The framework uses information from three layers of the environment:



\- Physical

\- Device

\- Cyber



The three modalities are processed separately, converted into compact representations, and then combined for anomaly detection. The framework also uses Federated Learning and Differential Privacy to support privacy-preserving training and reduce the risk of information leakage from model updates.



\## What the Framework Does



The overall pipeline consists of the following steps:



1\. Prepare and synchronize the physical, device, and cyber data.

2\. Split the synchronized data into training and testing sets using an 80:20 ratio.

3\. Train a separate encoder for each modality.

4\. Generate physical, device, and cyber embeddings.

5\. Calculate cross-layer consistency between the embeddings.

6\. Combine the representations using a learnable modality gate.

7\. Train the fusion network using reconstruction error.

8\. Distribute the fusion model across three federated clients.

9\. Apply FedProx during local federated training.

10\. Apply Differential Privacy using gradient clipping and noise addition.

11\. Use the reconstruction error to identify anomalous observations.



A more detailed description of the pipeline is provided in

\[`docs/methodology.md`](docs/methodology.md).



\## Dataset



The experiments use the \*\*Cyber-Physical Anomaly Detection in Smart Homes\*\* dataset introduced by Majib et al.



The dataset contains physical, device, and cyber information collected from a smart-home environment.



After preprocessing and synchronization, the dataset contains 40,802 common observations. The data is divided into 32,641 training samples (80%) and 8,161 testing samples (20%).



More information about the dataset and preprocessing is available in

\[`docs/dataset.md`](docs/dataset.md).



\## 80:20 Pipeline



The main implementation used in the experiments is located in:



```text

src/pipeline\_8020/

