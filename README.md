# Comprehensive Evaluation of Explainable AI for Multivariate Time-Series Regression Tasks
This is a repository for our paper, "Comprehensive Evaluation of Explainable AI for Multivariate Time-Series Regression Tasks", submitted to IEEE TPAMI. We employ post-hoc explainability methods; [DeepSHAP](https://github.com/shap/shap) and [TS-MULE](https://github.com/dbvis-ukon/ts-mule) on several deep learning models; CNN, DNN, LSTM and Transformers to compare the fidelity and reliability of feature importance scores from the two methods.

To further evaluate the explanations from DeepSHAP, we use a threefold strategy-
- We perform Perturbation Analysis and Visual Analysis for a comprehensive validation process
- We incorporate feature importance scores into the model training process through feature weighting or feature augmentation

## Installation
This code needs python- 3.9 or higher
''' pip install -r requirements.txt '''

 
