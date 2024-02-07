import numpy as np
import pandas as pd
import dill
import shap
import sys
sys.path.append('.')
sys.path.append('./tsmule')
import timeit

import matplotlib.pyplot as plt
import shap
from sklearn import metrics
from sklearn.model_selection import linear_model, train_test_split
from tsmule.xai.lime import LimeTS
from tsmule.sampling.segment import WindowSegmentation, MatrixProfileSegmentation, SAXSegmentation
from tsmule.sampling.perturb import Perturbation
from tsmule.xai.evaluation import PerturbationAnalysis
import torch
from train import RNNModel, TransformerRegressor







def cal_pert_score(orig_score, pert_score, rand_score):
    pert_c = (orig_score - pert_score)/orig_score
    rand_c = (orig_score - rand_score)/orig_score
    score = np.abs(pert_c/rand_c) 
    return score




def add_metadata(contribution_method, orig_score, pert_score, rand_score):
  new_row = {
        "Contribution Method": contribution_method, 
        "Original Score": orig_score, 
        "Perturbation Score": pert_score, 
        "Random Score": rand_score, 
        "Perturbation Score Metric": cal_pert_score(orig_score, pert_score, rand_score)
    }
    
    # Append the new row to the DataFrame and return it
  return metadata.append(new_row, ignore_index=True)




def predict_fn(x):
    if len(x.shape) == 2:
        prediction = model.predict(x[np.newaxis, :, :]).ravel()
    else:
        prediction = model.predict(x).ravel()
    return prediction
  









# MAIN FUNCTION STARTS HERE- 

if len(sys.argv) <= 1:
        print("Usage: python myscript.py arg1 arg2")
        sys.exit(1)

model_ = sys.argv[0]


metadata=pd.DataFrame(columns=["Contribution Method", "Original Score", "Perturbation Score", "Random Score", "Perturbation Score Metric"])



device= 'cuda' if torch.cuda.is_available() else 'cpu'
with open('data.dill', 'rb') as f:
    dataset_test = dill.load(f)

df = dataset_test[0][:10000]
pred_val = dataset_test[1][:10000]


if model_=="RNN":
    model = RNNModel((df.shape[1], df.shape[2]))
    model.load_state_dict(torch.load("rnn_model", map_location=device))

if model_=="trans":
    # Model parameters
    d_model = 128
    nhead = 8
    num_encoder_layers = 3
    dim_feedforward = 512
    output_dim = 1
    input_dim = df.shape[2]
    model = TransformerRegressor(input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim)
    model.load_state_dict(torch.load("transformer_model", map_location = device))






tic=timeit.default_timer()
print("Start Computing Feature Contribution Scores")
shap_exp=shap.DeepExplainer(model, df[:3000]) #expected shape (None, 24, 7)

shap_values=shap_exp.shap_values(df)

print('Total time: ' + str(timeit.default_timer()-tic))

shap_values=np.asarray(shap_values).squeeze()
print("Shap Values.shape: ", shap_values.shape)

#save shap values
with open('shap_values.dill', 'wb') as f:
    dill.dump(shap_values, f)


pa = PerturbationAnalysis()
scores = pa.analysis_relevance(df, pred_val, shap_values,
                        predict_fn=predict_fn,
                        replace_method='zeros',
                        eval_fn=metrics.mean_squared_error,
                        percentile=90
                        )
add_metadata("Shap Values", scores['original'], scores['percentile'], scores['random'])






lasso_classifier = linear_model.Lasso(alpha=0.01)  #faster the model, faster LIME works
per=Perturbation()

#Feature Contribution with LIME and Uniform Segmentation
tic = timeit.default_timer()
print("Start computation of LIME Values")

#segments object, WindowSegmentation object has stationery and exponential segmentations techniques
uniform_seg=WindowSegmentation(partitions=4, win_length=24)
uniform_lime=LimeTS(kernel=lasso_classifier, segmenter=uniform_seg, sampler=per, n_samples=24)
lime_values_uni=[uniform_lime.explain(df[i], predict_fn, segmentation_method='uniform')
                 for i in range(len(df))]

print('Total time: ' + str(timeit.default_timer()-tic)) 

pa = PerturbationAnalysis()
scores = pa.analysis_relevance(df, pred_val, lime_values_uni,
                        predict_fn=predict_fn,
                        replace_method='zeros',
                        eval_fn=metrics.mean_squared_error,
                        percentile=90
                        )

add_metadata("LIME Values with Uniform Segmentation", scores['original'],
             scores['perturbation'], scores['random'])






#LimeTS object for exponential window segmentation
tic = timeit.default_timer()
print("Computation of Lime Values with Exponential Segmentation")

#segment object, WindowSegmentation has stationery and exponentials segmentation techniques
exp_seg=WindowSegmentation(partitions=4, win_length=24)
exp_lime=LimeTS(kernel=lasso_classifier, segmenter=exp_seg, sampler=per, n_samples=24)
#explainer for LimeTS
lime_values_exp=[exp_lime.explain(df[i], predict_fn, segmentation_method='exponential')
                 for i in range(len(df))]

print('Total time: ' + str(timeit.default_timer()-tic))



pa = PerturbationAnalysis()
scores = pa.analysis_relevance(df, pred_val, lime_values_exp,
                        predict_fn=predict_fn,
                        replace_method='zeros',
                        eval_fn=metrics.mean_squared_error,
                        percentile=90
                        )

add_metadata("LIME Values with Exponential Segmentation", scores['original'],
             scores['perturbation'], scores['random'])







#LimeTS object for SAX segmentation
tic=timeit.default_timer()
print("Computation of Lime Values with SAX Segmentation")


#create segment object for SAX Transformation
seg_sax=SAXSegmentation(partitions=4, win_length=10)

lime_sax=LimeTS(kernel=lasso_classifier, segmenter=seg_sax, sampler=per, n_samples=24)
lime_values_sax=[lime_sax.explain(df[i], predict_fn) for i in range(len(df))]

print('Total time: ' + str(timeit.default_timer()-tic))


pa = PerturbationAnalysis()
scores = pa.analysis_relevance(df, pred_val, lime_values_sax,
                        predict_fn=predict_fn,
                        replace_method='zeros',
                        eval_fn=metrics.mean_squared_error,
                        percentile=90
                        )

add_metadata("LIME Values with SAX Segmentation", scores['original'],
             scores['perturbation'], scores['random'])


metadata.to_csv("scores.csv")