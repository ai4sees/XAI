from src.model import *
import pandas as pd
import argparse
import torch
import dill
from src.xai import Xai

with open("data.dill", "rb") as f:
    data = dill.load(f)

total_windows = data[0].shape[0]

parser = argparse.ArgumentParser()


parser.add_argument("--model", type=str, default="trans")
parser.add_argument("--windows", type=int, default=total_windows)
parser.add_argument("--replace_method", type=str,
                    choices=["zeros", "global_mean", "local_mean", "inverse_mean", "inverse_max"], default="zeros")
parser.add_argument("--window_length", type=int, default=10)
parser.add_argument("--n_samples", type=int, default=10)
parser.add_argument("--device", type=str, default='cpu')
args = parser.parse_args()


df=data[0][:args.windows]
labels=data[1][:args.windows]

if args.model == "trans":
    # Model parameters
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_encoder_lay", type=int, default=3)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    args = parser.parse_args()

    d_model = args.d_model
    nhead = args.nhead
    num_encoder_layers = args.num_encoder_lay
    dim_feedforward = args.dim_feedforward
    output_dim = 1
    input_dim = df.shape[2]

    model = TransformerRegressor(input_dim, d_model, nhead,
                                 num_encoder_layers, dim_feedforward,
                                 output_dim)

if args.model == "rnn":
    model = RNNModel((df.shape[1], df.shape[2]))

if args.model == "cnn":
    model = CNNModel((df[0].shape[1], df.shape[2]))

if args.model == "dnn":
    model = DNNModel(df.shape[1], df.shape[2])



model.load_state_dict(torch.load("model.pt", map_location=args.device))



back_data = df[:3000]
xai_ = Xai(model, back_data, df, labels)
pert_scores = {}
shap_values = xai_.get_shap_values()
pert_scores["shap"] = xai_.get_perturbation_score(shap_values).item()
with open("shap_values.dill", "wb") as f:
    dill.dump(shap_values, f)

lime_values_uni = xai_.get_lime_values("uniform segmentation",
                                       win_length=args.window_length,
                                       n_samples=args.n_samples)
pert_scores["lime with uniform segm"] = xai_.get_perturbation_score(lime_values_uni).item()

lime_values_exp = xai_.get_lime_values("exponential segmentation",
                                       win_length=args.window_length,
                                       n_samples=args.n_samples)
pert_scores["lime with exponential segm"] = xai_.get_perturbation_score(lime_values_exp).item()

lime_values_sax = xai_.get_lime_values("sax segmentation",
                                       win_length=args.window_length,
                                       n_samples=args.n_samples)
pert_scores["lime with sax segm"] = xai_.get_perturbation_score(lime_values_sax).item()

scores = pd.DataFrame.from_dict(pert_scores, orient='index', columns=['Score'])
scores.to_csv("scores.csv")
