# Comprehensive Evaluation of Explainable AI for Multivariate Time-Series Regression Tasks
This is a repository for our paper, "Comprehensive Evaluation of Explainable AI for Multivariate Time-Series Regression Tasks", submitted to IEEE TPAMI. We employ post-hoc explainability methods; [DeepSHAP](https://github.com/shap/shap) and [TS-MULE](https://github.com/dbvis-ukon/ts-mule) on several deep learning models; CNN, DNN, LSTM and Transformers to compare the fidelity and reliability of feature importance scores from the two methods.

To further evaluate the explanations from DeepSHAP, we use a threefold strategy-
- We perform Perturbation Analysis and Visual Analysis for a comprehensive validation process
- We incorporate feature importance scores into the model training process through feature weighting or feature augmentation

## Project Structure

    XAI/
    -data/
    -exec/
    -notebooks/
    -perturb/
    -results/
    -src/
    -tsmule/
    -xai_eval/tsmule/
    -README.md
    -requirements.txt

where:

&#8226; `data/` contains datasets used in this project  
&#8226; `exec/` contains executable files for preproecessing and downloading the input data, computation of feature contributions and model training with incorporation of feature contribution and without incorporation of feature contribution  
&#8226; `notebooks/` contains demo execution of the project and visual analysis of feature importance scores in jupyter notebooks  
&#8226; `perturb/` contains perturbation related functions and class files  
&#8226; `results/` contains sample data, model and other output files  
&#8226; `src/` contains class files and other functions related to preprocessing of data, training of models and computation of feature contributions  
&#8226; `tsmule/` contains the TS-MULE repository code  
&#8226; `xai_eval/tsmule/` contains cache files related to the project   


## Installation

This code needs python- 3.9 or higher. For other dependencies of the project, run the command


    pip install -r requirements.txt 

## Results Reproduction
In order to reproduce the results depicted in the paper, it is essential to follow a sequence of steps as-
- Preprocessing of data  
- Training of models
- Computation of Feature Importance scores
- Utilizing Feature Importance scores for Comparative, Computational Measures and Visual Analysis


 ### Preprocessing of Data
 Run the following command to preprocess data

     python -m exec.preprocess_data --dataset <dataset> --url <url> --y_column <y> --drop_col <cols> --scaler <scaler> --window_size <window_size>

 Here, `dataset` can either be beijing_pm2.5 beijing_multi_site, synthetic_data or any other dataset can be imported by providing url to the `url` argument. `y_column` indicates the predicted feature of the dataset. Other arguments like `drop_col`, `scaler` and `window_size` are data preprocessing options. `drop_col` takes names of all columns to be dropped, if any, `scaler` argument takes either normalize or standardize, depending on the type of normalization technique to use on data and `window_size` sets the window length. This will also download files for preprocessed data.

### Training the Models
Run the following command to train a model on the chosen dataset

    python -m exec.train --model <model> --epochs <epochs> --batch_size <batch_size> --test_size <test_size> --windows <windows> --device <device>
    
Here, the `model` argument can be cnn, dnn, rnn or trans (for transformers). This trains and saves model files for the deep learning model with the given training specifications. `windows` argument takes the number of windows of the total windows of complete dataset to train the model on.   

### Compute Feature Importance Scores and calculate Perturbation Score Metric
To get feature contribtuion scores using various interpretability methods and their perturbation score metric, run the command

    python -m exec.feature_contributions --model <model> --windows <windows> --replace_method <repl> --window_length <l> --n_samples <n> --device <device>

`model` argument specifies the type of trained model, `windows` specify the number of windows of the complete dataset to be taken. `replace_method` argument specifies the replaced values in perturbation, its options can be- zeros, global_mean, local_mean, inverse_mean, inverse_max. `window_length` argument specifies the length of each window in dataset. 

### Training of Model with Incorporation of Feature Importance Scores
Now, for training model with incorporating feature importance scores either through featuring weighting or data augmentation to run experiments for feature weighting or data augmentation, the commands to be used are

    python -m exec.train_with_feat_transformation --model <model> --epochs --batch_size <batch_size> --test_size <test_size> --windows <windows> --device <device>
    python exec.train_with_feat_augmentation --model <model> --epochs --batch_size <batch_size> --test_size <test_size> --windows <windows> --device <device>

`model` argument specifies the type of model used for the original training process so that corresponding feature contributions can be incorporated. 


### Results

The Loss Curves for training a model without feature importance scores and with feature importance scores are shown for LSTM and Transformer Models. The
loss curves are plotted as training model with original data against training model with data modified with contribution scores.


![loss_curves](https://github.com/ai4sees/XAI/assets/104296674/fd3d84ed-ae23-4de7-a89e-33d3cf8739dd)

The graphs depict the visual comaprison of shap scores for LSTM and Transformers models for Beijing Air Multi Site dataset. The first graph. In the first graph, Points are highlighted to depict how feature importances are varying at several points with respect to original time stamps. The topmost subplot of the second graph represents the forecast values for each window in Beijing Multi Site Air Quality Dataset. subsequent plots illustrates SHAP feature importances for the selected time period.  

![figure6](https://github.com/ai4sees/XAI/assets/104296674/c12fd90c-af65-4eb5-a083-501533a80354)


![figure7](https://github.com/ai4sees/XAI/assets/104296674/e94d4bbf-52eb-4849-9e20-e53f478610fe)




