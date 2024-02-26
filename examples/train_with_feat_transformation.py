import dill
import argparse
from resources.model import *
from resources.xai import Xai

with open("data.dill", "rb") as f:
    data = dill.load(f)

df = data[0]
labels = data[1]
parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default="trans")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--test_size", type=float, default=0.3)
parser.add_argument("--windows", type=int, default=df.shape[0])
parser.add_argument("--device", type=str, default='cpu')

args = parser.parse_args()

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
    model = CNNModel((df.shape[1], df.shape[2]))

if args.model == "dnn":
    model = DNNModel(df.shape[1], df.shape[2])

df = df[:args.windows]
labels = labels[:args.windows]
with open("shap_values.dill", "rb") as f:
    shap_values = dill.load(f)

feat_cont=shap_values[:args.windows]
back_data=df[:100]

xai_ = Xai(model, back_data, df, labels)
m, trans_train_loss, trans_val_loss = xai_.train_with_feat_transformation(feat_cont,
                  epochs = args.epochs, batch_size = args.batch_size, device = args.device,
                             test_size = args.test_size)



with open("train_loss.dill", "rb") as f:
    orig_train_loss = dill.load(f)

with open("val_loss.dill", "rb") as f:
    orig_val_loss = dill.load(f)

plt.figure(figsize=(20, 20))
plt.plot(range(args.epochs), orig_train_loss, label="train loss with original data")
plt.plot(range(args.epochs), orig_val_loss, label="validation loss with original data")
plt.plot(range(args.epochs), trans_train_loss, label="train loss with transformed data")
plt.plot(range(args.epochs), trans_val_loss, label="validation loss with transformed data")
plt.legend()
plt.ylim(0, 1)
plt.savefig("loss_curves_transformation.png")




