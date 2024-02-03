# Comprehensive Evaluation of Explainable AI for Multivariate Time-Series Regression Tasks
This is a repository for our paper, "Comprehensive Evaluation of Explainable AI for Multivariate Time-Series Regression Tasks", submitted to IEEE TPAMI. We employ post-hoc explainability methods; [DeepSHAP](https://github.com/shap/shap) and [TS-MULE](https://github.com/dbvis-ukon/ts-mule) on several deep learning models; CNN, DNN, LSTM and Transformers to compare the fidelity and reliability of feature importance scores from the two methods.

To further evaluate the explanations from DeepSHAP, we use a threefold strategy-
- We perform Perturbation Analysis and Visual Analysis for a comprehensive validation process
- We incorporate feature importance scores into the model training process through feature weighting or feature augmentation

## Installation
This code needs python- 3.9 or higher

    pip install -r requirements.txt 

## Data Preprocessing and Model Training
Data preprocessing and model training can either be visualised in notebooks or run the following commands for direct execution:

    jupyter nbconvert --to script preprocess_and_train_multi_site.ipynb --execute
    jupyter nbconvert --to script preprocess_and_train_beijing_2.5.ipynb --execute

## Comparative Analysis of DeepSHAP and TS-MULE
the trained model files as well as preprocessed data files are used to compute feature importance scores with DeepSHAP and TS-MULE and the perturbation score metric. Feature scores and their perturbation metric scores can be visualised in the provided notebooks.

## Model Training With Feature Incorporation

To train models with and without feature incorporation in the model training process and then comparing the performance of both models, run the following commands:

    python <feature incorporation method>/<dataset>.py
    
where <feature incorporation method> can be Data Augmentation or Feature Transformation and <dataset> can be beijing_air_multi_site, beijing_air_pm2.5 or synthetic_data

 
