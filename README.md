# Comprehensive Evaluation of Explainable AI for Multivariate Time-Series Regression Tasks
This is a repository for our paper, "Comprehensive Evaluation of Explainable AI for Multivariate Time-Series Regression Tasks", submitted to IEEE TPAMI. We employ post-hoc explainability methods; [DeepSHAP](https://github.com/shap/shap) and [TS-MULE](https://github.com/dbvis-ukon/ts-mule) on several deep learning models; CNN, DNN, LSTM and Transformers to compare the fidelity and reliability of feature importance scores from the two methods.

To further evaluate the explanations from DeepSHAP, we use a threefold strategy-
- We perform Perturbation Analysis and Visual Analysis for a comprehensive validation process
- We incorporate feature importance scores into the model training process through feature weighting or feature augmentation

## Installation
This code needs python- 3.9 or higher

    pip install -r requirements.txt 

## Results Reproduction
In order to reproduce the results depicted in the paper, it is essential to follow a sequence of steps as-
- Preprocessing of data
- Training the models
- Get Feature Importance scores
- Use Feature Importance scores for Comparative, Computational Measures and Visual Analysis

 ### Preprocessing of Data
 Run the following command to preprocess data

     python preprocess.py <dataset> <scalar>

 <dataset> can either be beijing_PM2.5 or beijing-multi_site and <scalar> should be replaced by normalize or standardize, depending on the type of normalization technique to use on data. This will also download files for preprocessed data.

### Training the Models
Run the following command to train LSTM and Transformer models on the chosen dataset

    python train.py
    
This trains and saves model files for the deep learning models.

### Compute Feature Importance Scores and calculate Perturbation Score Metric
To get feature contribtuion scores using various interpretability methods and their perturation score metric, run the command

    python pert_score.py <model>

<model> should either be replcaed by RNN or trans, depending the model for which you want results for.

### Training of Model with Incorporation of Feature Importance Scores
Now we train the model again, first without including feature contribution scores and then with incorporating feature importance scores either through featuring weighting or data augmentation. To Run experiments for feature weighting or data augmentation, the commands to be used are

    python feat_transformation.py <model>
    python data_augmentation.py <model>

<model> should be replaced by eith RNN or trans, depending on the type of model chosen.


  


 
