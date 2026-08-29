\# Dataset



\## Dataset Source



The experiments use the Cyber-Physical Anomaly Detection in Smart Homes dataset introduced by Majib et al. (2023).



The dataset was collected in a real smart-home environment and contains information from multiple sources, including physical environmental sensors, smart devices, and cyber/network traffic.



The original dataset covers a period of four weeks. The dataset contains activities performed by two actors, where the primary actor represents normal behaviour and the second actor represents anomalous behaviour.



\## Data Modalities



Three modalities from the dataset are used in this project:



| Modality | Description |

|---|---|

| Physical | Environmental sensor measurements collected from the smart-home environment |

| Device | Behaviour and observations from connected smart-home devices |

| Cyber | Network communication and traffic information |



The three modalities are generated independently and contain different types of information. They are therefore processed separately and synchronized using a common temporal representation before being used together in the proposed framework.



\## Synchronized Dataset



After preprocessing, feature extraction, and temporal synchronization, the three modalities were combined using common temporal observations.



The resulting synchronized dataset contains:



\- 40,802 common observations

\- 572 physical features

\- 2,780 device features

\- 12 cyber features



These synchronized observations form the input to the subsequent modality-specific representation learning pipeline.



\## Ground Truth



The `GroundTruth` label is used to distinguish normal and anomalous observations.



The labels are maintained separately from the feature representations and are used for the train-test split and final evaluation.



\## 80:20 Train-Test Split



For the randomized 80:20 experiment, the synchronized dataset was divided into:



| Dataset | Samples | Normal | Anomaly |

|---|---:|---:|---:|

| Training | 32,641 | 29,185 | 3,456 |

| Testing | 8,161 | 7,297 | 864 |

| Total | 40,802 | 36,482 | 4,320 |



The split was performed using:



\- 80% training data

\- 20% testing data

\- `random\_state = 42`

\- Stratification using the `GroundTruth` label



The same split indices were applied to the physical, device, and cyber modalities so that the corresponding observations remained aligned.



\## Class Distribution



The dataset is naturally imbalanced, with normal observations hugely outnumbering anomalous observations.



In the test data, there are:



\- 7,297 normal observations

\- 864 anomalous observations



No class balancing or oversampling was applied in the 80:20 experimental pipeline. The original class distribution was therefore retained for evaluation.



\## Preprocessing



Before model training:



1\. Timestamp information not used as a model feature is removed.

2\. Missing values are replaced with zero.

3\. Each modality is standardized separately using `StandardScaler`.

4\. The scaler is fitted using the training data of that modality.

5\. The fitted scaler is then applied to the corresponding test data.



The resulting processed physical, device, and cyber representations are used to train the modality-specific encoders.

