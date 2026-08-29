\# Ablation Study



\## Overview



An ablation study was carried out to understand the contribution of the main components of the proposed PCCF framework.



All experiments use the same 80:20 train-test split. The experiments differ by removing or changing one main component and comparing the results with the complete configuration.



The four experiments are:



| Experiment | Setup |

|---|---|

| E1 — No DP | Federated Learning without Differential Privacy |

| E2 — No FL | Centralized training without Federated Learning and Differential Privacy |

| E3 — No Consistency | Federated Learning without the consistency vector |

| E4 — FL + DP | Complete Federated Learning and Differential Privacy setup |



\---



\## E1 — Without Differential Privacy



E1 removes Differential Privacy while keeping the federated learning setup and the consistency vector. This experiment is used to study the effect of Differential Privacy on the overall performance.



The federated setup uses three clients and FedProx for local training.



\### Results



| Metric | E1 Result |

|---|---:|

| ROC-AUC | 0.6145 |

| F1-score | 0.1721 |

| Accuracy | 0.6640 |

| Precision | 0.1164 |

| Recall | 0.3299 |



\### Confusion Matrix



|  | Predicted Normal | Predicted Anomaly |

|---|---:|---:|

| Actual Normal | 5134 | 2163 |

| Actual Anomaly | 579 | 285 |



\---



\## E2 — Without Federated Learning



E2 removes Federated Learning and uses centralized training instead. The complete training portion of the 80:20 split is used for centralized model training.



Differential Privacy is not used in this experiment, while the consistency vector remains part of the fusion representation.



The final training run used 5 epochs.



\### Results



| Metric | E2 Result |

|---|---:|

| ROC-AUC | 0.5749 |

| F1-score | 0.1504 |

| Accuracy | 0.6552 |

| Precision | 0.1017 |

| Recall | 0.2882 |



\### Confusion Matrix



|  | Predicted Normal | Predicted Anomaly |

|---|---:|---:|

| Actual Normal | 5098 | 2199 |

| Actual Anomaly | 615 | 249 |



\---



\## E3 — Without Consistency Vector



E3 removes the consistency vector from the fusion process while retaining the physical, device, and cyber embeddings.



Each modality embedding has 32 dimensions. Since the 3-dimensional consistency vector is removed, the fusion input becomes:



\*\*32 + 32 + 32 = 96 dimensions\*\*



Federated Learning with three clients and FedProx is retained in this experiment.



\### Results



| Metric | E3 Result |

|---|---:|

| ROC-AUC | 0.6097 |

| F1-score | 0.1922 |

| Accuracy | 0.6745 |

| Precision | 0.1304 |

| Recall | 0.3657 |



\### Confusion Matrix



|  | Predicted Normal | Predicted Anomaly |

|---|---:|---:|

| Actual Normal | 5189 | 2108 |

| Actual Anomaly | 548 | 316 |



\---



\## E4 — Federated Learning + Differential Privacy



E4 represents the complete configuration used in the 80:20 experiment.



It includes the physical, device, and cyber embeddings, the consistency vector, modality gating, fusion autoencoder, Federated Learning, FedProx, and Differential Privacy.



\### Results



| Metric | E4 Result |

|---|---:|

| ROC-AUC | 0.6622 |

| F1-score | 0.2025 |

| Accuracy | 0.6768 |

| Precision | 0.1371 |

| Recall | 0.3877 |



\### Confusion Matrix



|  | Predicted Normal | Predicted Anomaly |

|---|---:|---:|

| Actual Normal | 5188 | 2109 |

| Actual Anomaly | 529 | 335 |



\---



\## Overall Comparison



The results from all four experiments are compared using the same evaluation measures.



| Metric | E1 — No DP | E2 — No FL | E3 — No Consistency | E4 — FL + DP |

|---|---:|---:|---:|---:|

| ROC-AUC | 0.6145 | 0.5749 | 0.6097 | \*\*0.6622\*\* |

| F1-score | 0.1721 | 0.1504 | 0.1922 | \*\*0.2025\*\* |

| Accuracy | 0.6640 | 0.6552 | 0.6745 | \*\*0.6768\*\* |

| Precision | 0.1164 | 0.1017 | 0.1304 | \*\*0.1371\*\* |

| Recall | 0.3299 | 0.2882 | 0.3657 | \*\*0.3877\*\* |



\## Summary



The complete E4 configuration gives the highest ROC-AUC, F1-score, accuracy, precision, and recall among the four experiments.



E1 examines the effect of removing Differential Privacy.



E2 examines the effect of removing Federated Learning and using centralized training.



E3 examines the contribution of the consistency vector by removing it from the fusion representation.



The four configurations are compared using the same 80:20 dataset split and evaluation procedure.

