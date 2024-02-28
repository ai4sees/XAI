from src.model import *
import argparse
import dill
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt

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
df = df[:args.windows]
labels = labels[:args.windows]

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

x_train, x_test, y_train, y_test = train_test_split(df, labels,
                                                    test_size=args.test_size, shuffle=True)

train_model = TrainModel(model, x_train, y_train, x_test,
                         y_test, epochs=args.epochs, batch_size=args.batch_size,
                         device=args.device)

model_, train_loss, val_loss = train_model()
print(train_model.test_model())
torch.save(model.state_dict(), "model.pt")
with open("train_loss.dill", "wb") as f:
    dill.dump(train_loss, f)
with open("val_loss.dill", "wb") as f:
    dill.dump(val_loss, f)
plt.figure(figsize=(20, 20))
plt.plot(range(args.epochs), train_loss, label="train loss")
plt.plot(range(args.epochs), val_loss, label="validation loss")
plt.legend()
plt.ylim(0, 1)
plt.savefig("loss_curves.png")
