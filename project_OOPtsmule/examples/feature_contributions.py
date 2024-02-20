import argparse
import torch
import dill
from xai import xai





 with open("data.dill", "rb") as f:
    data = dill.load(f)

total_windows = data[0].shape[0]

parser = argparse.ArgumentParser()

parser.add_argument("--windows", type=int, default=total_windows)
parser.add_argument("--replace_method", type=str,
                    choices=["zeros", "global_mean", "local_mean", "inverse_mean", "inverse_max"], default="zeros")
parser.add_argument("--window_length", type=int, default=24)
parser.add_argument("--n_samples", type=int, default=24)
parser.add_argument("--device", type= str, default= 'cpu')
args = parser.parse_args()


model = torch.load_state_dict(torch.load("model.pt", map_location = args.device))




df = data[0][:args.windows]
labels = data[1][:args]

back_data = df[:3000]
xai_ = xai(model, back_data, df)

shap_values = xai_.get_shap_values()
lime_values_uni = xai_.get_lime_values("uniform segmentation",
                                       win_length = args.window_length,
                                       n_samples = args.n_samples)

lime_values_exp = xai_.get_lime_values("exponential segmentation",
                                       win_length = args.window_length,
                                       n_samples = args.n_samples)

lime_Values_sax = xai_.get_lime_values("sax segmentation",
                                       win_length = args.window_length,
                                       n_samples = args.n_samples)





