import dill

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from statistics import mean
from IPython.display import clear_output, display
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import argparse

num_epochs = []
#CREATE LOSS CURVES
def draw_curve(epochs, train_loss, val_loss):
  clear_output(wait=True)
  num_epochs.append(epochs)
  plt.plot(num_epochs, train_loss, label = "Train loss")
  plt.plot(num_epochs, val_loss, label = "Val Loss")
  plt.show()


def save_loss_curves(epochs, train_loss, val_loss, filename, ylim = False):
  plt.plot(range(epochs), train_loss, label = "train loss")
  plt.plot(range(epochs), val_loss, label = "val loss")
  if ylim:
    plt.ylim(0, 1)
  plt.legend()
  plt.savefig(f"{filename}.png", format = "png")

#FUNCTION FOR MODEL TRAINING
def run_model(model, x_train, y_train, x_test, y_test, epochs, CNN, device):
  train_loss=[]
  val_loss=[]

  #hyperparameters
  input_size = 3
  hidden_size = 4
  output_size = 1
  learning_rate = 0.01

  x_train=torch.from_numpy(x_train).float().to(device)
  y_train=torch.from_numpy(y_train).float().to(device)

  x_test=torch.from_numpy(x_test).float().to(device)
  y_test=torch.from_numpy(y_test).float().to(device)

  # Create a DataLoader to handle batching
  batch_size = 32  # Set your desired batch size
  dataset_train = TensorDataset(x_train, y_train)
  dataset_test=TensorDataset(x_test, y_test)

  data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
  data_loader_test=DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

  optimizer = optim.Adam(model.parameters(), lr= learning_rate)
  mse=nn.MSELoss()
 

  for epoch in range(epochs):
      total_loss = 0
      model.train()
      for batch_idx, (batch_data, batch_labels) in enumerate(data_loader_train):  # Iterate through batches
        if CNN==True:
          data=batch_data.transpose(1, 2)
        else:
          data=batch_data
        optimizer.zero_grad()
        # Forward pass
        outputs = model(data)

        # Calculate the custom loss with importance
        batch_labels = batch_labels.view(-1, output_size)  # Ensure target size matches output size
        loss=mse(outputs, batch_labels)

        # Backpropagation and optimization

        loss.backward()
        optimizer.step()


        total_loss += loss.item()*len(batch_data)
   
      avg_loss=total_loss/len(data_loader_train)
      train_loss.append(avg_loss)


      model.eval()
      with torch.no_grad():
        total_val_loss = 0
        for batch_data, batch_labels in data_loader_test:
          if CNN==True:
            data=batch_data.transpose(1, 2)
          else:
            data=batch_data
          outputs = model(data)
          batch_labels = batch_labels.view(-1, output_size)
          loss=mse(outputs, batch_labels)

          total_val_loss+=loss.item()*len(batch_data)
     
      avg_val_loss=total_val_loss/len(data_loader_test)
      val_loss.append(avg_val_loss)
      print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
      draw_curve(epoch, train_loss, val_loss)
  print('Training finished.')
 
  return model, train_loss, val_loss




class RNNModel(nn.Module):
    def __init__(self, input_shape):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_shape[1], hidden_size=32, batch_first=True)  # Increased hidden size and layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * input_shape[0], 100),  # Increased output size
            nn.ReLU(),
            nn.Dropout(0.3),  # Slightly increased dropout
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x



#Transformer model
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim):
        super(TransformerRegressor, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)

        # Define a single Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward
        )

        # Create the Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        self.output = nn.Linear(d_model, output_dim)

    def forward(self, src):
        src = self.embedding(src)
        memory = self.transformer_encoder(src)
        out = torch.mean(memory, dim=1)  # Aggregating across the time dimension
        out = self.output(out)
        return out




if __name__ =="__main__":
  with open("data.dill", "rb") as f:
    df = dill.load(f)
  device= 'cuda' if torch.cuda.is_available() else 'cpu'
  parser = argparse.ArgumentParser()

  parser.add_argument("--epochs", type = int, default = 100)
  parser.add_argument("--test_size", type = float, default = 0.2)
  parser.add_argument("--windows", type = int, default = 1000)
  args = parser.parse_args()

  #taking first 10000 windows for training
  X = df[0][:args.windows]
  y = df[1][:args.windows]
  epochs = args.epochs
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=13)


  #Train RNN Model
  model=RNNModel((X.shape[1], X.shape[2]))
  model.to(device)
  model_, train_loss, val_loss = run_model(model, X_train, y_train, X_test, y_test, epochs, False, device)
  x_test=torch.from_numpy(X_test).float().to(device)
  y_test_=torch.from_numpy(y_test).float().to(device)
  dataset = TensorDataset(x_test, y_test_)

  data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
  mse_ = nn.MSELoss()
  for x_test, y_test_ in data_loader:
    predictions= model_(x_test)
    mse = mse_(predictions, y_test_)

  print("MSE: ", mse)

  #SAVE RNN MODEL
  torch.save(model_.state_dict(), "rnn_model.pt")
  save_loss_curves(epochs, train_loss, val_loss, "rnn_loss_curves")







  #TRAIN TRANSFORMERS MODEL 
  
  #model parameters
  num_epochs = []
  # Model parameters
  parser.add_argument("--d_model", type = int, default = 128)
  parser.add_argument("--nhead", type = int, default = 8)
  parser.add_argument("--num_encoder_lay", type = int, default = 3)
  parser.add_argument("--dim_feedforward", type = int, default = 512)
  args = parser.parse_args()
  d_model = args.d_model
  nhead = args.nhead
  num_encoder_layers = args.num_encoder_lay
  dim_feedforward = args.dim_feedforward
  output_dim = 1
  input_dim = X.shape[2]


  # Create the model
  model = TransformerRegressor(input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim)
  model.to(device)
  model_, train_loss, val_loss = run_model(model, X_train, y_train, X_test, y_test, epochs, False, device)
  x_test=torch.from_numpy(X_test).float().to(device)
  y_test_=torch.from_numpy(y_test).float().to(device)
  dataset = TensorDataset(x_test, y_test_)

  data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
  mse_ = nn.MSELoss()
  for x_test, y_test_ in data_loader:
    predictions= model_(x_test)
    mse = mse_(predictions, y_test_)

  print("MSE: ", mse)


  #SAVE RNN MODEL
  torch.save(model_.state_dict(), "transformer_model.pt")
  save_loss_curves(epochs, train_loss, val_loss, "trans_loss_curves", ylim = True)