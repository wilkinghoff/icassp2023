# icassp2023

## Description

Anomalous Sound Detection (ASD) system for task 2 "Unsupervised Anomalous Sound Detection for Machine Condition Monitoring Applying Domain Generalization Techniques" of the DCASE challenge 2022 (https://dcase.community/challenge2022/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring) achieving state-of-the-art performance. The system is a conceptually simple ASD system specifically designed for domain generalization and is trained trough an auxiliary task using the sub-cluster AdaCos loss (https://github.com/wilkinghoff/sub-cluster-AdaCos).

## Instructions

The implementation is based on Tensorflow 2.3 (more recent versions can run into problems with the current implementation). Just start the main script for training and evaluation. To run the code, you need to download the development dataset, additional training dataset and the evaluation dataset of the DCASE 2022 ASD dataset, and store the files in an './eval_data' and a './dev_data' folder.

## Reference

When reusing (parts of) the code, a reference to the following paper would be appreciated:

@unpublished{wilkinghoff2023design,
  author = {Wilkinghoff, Kevin},
  title  = {Design Choices for Learning Embeddings from Auxiliary Tasks for Domain Generalization in Anomalous Sound Detection},
  note   = {Accepted for presentation at International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year   = {2022}
}
