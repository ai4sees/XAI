import numpy as np
import matplotlib.pyplot as plt
import dill
import shap
import pandas as pd
from statistics import mean
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers, Sequential, Model
from IPython.display import clear_output, display
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

path='/content/drive/MyDrive/final_codes/'

def load_data(dataset):
  
  if dataset=='beijing 2.5':
    with open(path+'Beijing_Air_Quality_models/beijing_air_2_5_test_data.dill', 'rb') as f:
      dataset_test = dill.load(f)

  elif dataset=='multi site':
    with open(path+'beijing_air_multi_site_models/beijing_air_multi_site_test_data.dill', 'rb') as f:
      dataset_test = dill.load(f)

  elif dataset=='metro traffic':
    with open(path+'Metro_Interstate_models/metro_traffic_test_data.dill', 'rb') as f:
      dataset_test = dill.load(f)

  return dataset_test[0], dataset_test[1]

multi_site_data, multi_site_labels=load_data('multi site')

def save_file(list_, filename):
  with open(path+ f"Table_3/data_augmentation/{filename}", 'wb') as f:
    dill.dump(list_, f)


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




#RNN Model
class customRNN(nn.Module):
    def __init__(self, time_steps, features):
        super(customRNN, self).__init__()

        # Define the LSTM layer with hidden_size set to 7
        self.lstm = nn.LSTM(input_size=features, hidden_size=7, batch_first=True)

        # Define the fully connected layers
        # The input size for the first fully connected layer is time_steps * hidden_size
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7 * time_steps, 100)  # Adjusted input size based on LSTM output
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        # Apply LSTM
        lstm_out, _ = self.lstm(x)
        x = self.flatten(lstm_out)

        # Apply the fully connected layers
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x




def create_dataset(data, shap_values):
  new_data=[]
  for i, j in zip(data, shap_values):
    new_data.append(np.hstack((i, j)))

  new_data=np.array(new_data)
  return new_data



def run_model(model, x_train, y_train, x_test, y_test, epochs, device):
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

  optimizer = optim.SGD(model.parameters(), lr=learning_rate)
  mse=nn.MSELoss()

  for epoch in range(epochs):
      total_loss=0
      model.train()
      for batch_data, batch_labels in data_loader_train:  # Iterate through batches

        outputs = model(batch_data)

        # Calculate the custom loss with importance
        #batch_labels = batch_labels.view(-1, output_size)  # Ensure target size matches output size
        loss=mse(outputs, batch_labels)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()

    
      avg_loss=total_loss/len(data_loader_train)
      train_loss.append(avg_loss)

      model.eval()
      total_val_loss=0
      for batch_data, batch_labels in data_loader_test:
     
        outputs = model(batch_data)
        loss=mse(outputs, batch_labels)

        total_val_loss+=loss.item()        

      avg_val_loss=total_val_loss/len(data_loader_test)  
      val_loss.append(avg_val_loss)

      torch.cuda.empty_cache()

      print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_val_loss}')

  print('Training finished.')

  return model, train_loss, val_loss


def add_metadata(model, iter_, dataset, epochs, MSE):
  new_row = {
        "Model": model, 
        "No. of Iteration": iter_, 
        "dataset": dataset, 
        "epochs": epochs, 
        "MSE": MSE
    }
    
    # Append the new row to the DataFrame and return it
  return metadata.append(new_row, ignore_index=True)
  
# MAIN FUNCTION STARTS HERE- 

metadata=pd.read_csv(path+"Table_3/data_augmentation/metadata_aug.csv")
mse=nn.MSELoss()
orig_loss=[]
orig_val_loss=[]
mean_orig=[]
i=0

# Model parameters
d_model = 128
nhead = 8
num_encoder_layers = 3
dim_feedforward = 512
output_dim = 1
input_dim=11
device= 'cuda' if torch.cuda.is_available() else 'cpu'



#TRAINING WITHOUT FEATURES TRANSFORMERS
while (i<3):
  # Create the model
  model = TransformerRegressor(input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim)
  model.to(device)
  x_train, x_test, y_train, y_test= train_test_split(multi_site_data, multi_site_labels, test_size=0.3,
                                                     shuffle=False)


  model_, train_loss, val_loss =run_model(model, x_train, y_train, x_test,
                                                            y_test, 100, device)


  orig_loss.append(train_loss)
  orig_val_loss.append(val_loss)
  batch_size = 32  # Set your desired batch size
  x_test=torch.from_numpy(x_test).float().to(device)
  y_test=torch.from_numpy(y_test).float().to(device)
  dataset = TensorDataset(x_test, y_test)

  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

  for x_test, y_test in data_loader:
    predictions = model_(x_test)
    mse_=mse(predictions, y_test)
  print("MSE: ", mse_)
  mean_orig.append(mse_)
  i+=1


  metadata=add_metadata("Transformers: Training without feat_imp", i, "Beijing Multi Site", 100, mse_.item())


save_file(orig_loss, 'multi_site_trans_loss.dill')
save_file(orig_val_loss, 'multi_site_trans_val_loss.dill')
save_file(mean_orig, 'multi_site_trans_mse.dill')     



#training with feature_imps for Transformers
with open(path+'shap_values/multi_site_trans_shap.dill', 'rb') as f:
  shap_values=dill.load(f)


# Model parameters
d_model = 128
nhead = 8
num_encoder_layers = 3
dim_feedforward = 512
output_dim = 1
input_dim=14
device= 'cuda' if torch.cuda.is_available() else 'cpu'




new_loss=[]
new_val_loss=[]

mse_new=[]
i=0

while(i<3):
  model = TransformerRegressor(input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim)
  model.to(device)
  new_data=create_dataset(multi_site_data, shap_values)
  x_train, x_test, y_train, y_test= train_test_split(new_data, multi_site_labels, test_size=0.3,
                                                     shuffle=False)
  feat_imp_train, feat_imp_test=train_test_split(np.array(shap_values), test_size=0.3, shuffle=False)
  model_, train_loss, val_loss =run_model(model, x_train, y_train, x_test, y_test, 100, device)

  

  new_loss.append(train_loss)
  new_val_loss.append(val_loss)
  batch_size = 32  # Set your desired batch size
  x_test=torch.from_numpy(x_test).float().to(device)
  y_test=torch.from_numpy(y_test).float().to(device)
  dataset = TensorDataset(x_test, y_test)

  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
  mse_=[]
  for x_test, y_test in data_loader:
    predictions= model_(x_test)
    mse_=mse(predictions, y_test)

  print("MSE: ", mse_)
  mse_new.append(mse_)
  metadata=add_metadata("Transformers: Training with feat_imp", i, "Beijing Multi Site", 100, mse_.item())


  i+=1


save_file(new_loss, 'multi_site_trans_new_loss.dill')
save_file(new_val_loss, 'multi_site_trans_new_val_loss.dill')
save_file(mse_new, 'multi_site_trans_mse_new.dill')


#training without feat_imp RNN
orig_loss=[]
orig_val_loss=[]

mse_orig=[]

i=0

while(i<3):
  model=customRNN(24, 7)
  x_train, x_test, y_train, y_test= train_test_split(multi_site_data, multi_site_labels, test_size=0.3,
                                                     shuffle=False)
  model.to(device)
  model_, train_loss, val_loss =run_model(model, x_train, y_train, x_test, y_test, 100, device)

  

  orig_loss.append(train_loss)
  orig_val_loss.append(val_loss)
  batch_size = 32  # Set your desired batch size
  x_test=torch.from_numpy(x_test).float().to(device)
  y_test=torch.from_numpy(y_test).float().to(device)
  dataset = TensorDataset(x_test, y_test)

  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
  for x_test, y_test in data_loader:
    predictions= model_(x_test)
    mse_=mse(predictions, y_test)

  print("MSE: ", mse_)
  mse_orig.append(mse_)
  metadata=add_metadata("RNN: Training without feat_imp", i, "Beijing Air", 100, mse_.item())

  i+=1


save_file(orig_loss, 'multi_site_rnn_loss.dill')
save_file(orig_val_loss, 'multi_site_rnn_val_loss.dill')
save_file(mse_orig, 'multi_site_rnn_mse.dill') 



#training with feat_imps RNN
with open(path+ 'shap_values/multi_site_rnn_shap_values.dill', 'rb') as f:
  shap_values=dill.load(f)

new_loss=[]
new_val_loss=[]

mse_new=[]

i=0

while(i<3):
  model=customRNN(24, 14)
  new_data=create_dataset(multi_site_data, shap_values)
  x_train, x_test, y_train, y_test= train_test_split(new_data, multi_site_labels, test_size=0.3,
                                                     shuffle=False)
  model.to(device)                                                   
  model_, train_loss, val_loss =run_model(model, x_train, y_train, x_test, y_test, 100, device)
  
  

  new_loss.append(train_loss)
  new_val_loss.append(val_loss)
  batch_size = 32  # Set your desired batch size
  x_test=torch.from_numpy(x_test).float().to(device)
  y_test=torch.from_numpy(y_test).float().to(device)
  dataset = TensorDataset(x_test, y_test)

  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
 
  for x_test, y_test in data_loader:
    predictions= model_(x_test)
    mse_=mse(predictions, y_test)
 
  print("MSE: ", mse_)
  mse_new.append(mse_)
  metadata=add_metadata("RNN: Training with feat_imp", i, "Beijing Air", 100, mse_.item())

  i+=1


save_file(new_loss, 'multi_site_rnn_new_loss.dill')
save_file(new_val_loss, 'multi_site_rnn_new_val_loss.dill')
save_file(mse_new, 'multi_site_rnn_mse_new.dill')



metadata.to_csv(path+ "Table_3/data_augmentation/metadata_aug.csv")





