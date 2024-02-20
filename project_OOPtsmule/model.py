import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
from IPython.display import clear_output

class TrainModel():
    def __init__(self, model, x_train, y_train, x_test,
                 y_test, epochs = 100, batch_size = 32,
                 device = 'cpu'):
        self.train_loss = []
        self.val_loss = []
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device



    def __call__(self):
        self.run_model()




    def run_model(self):
        num_epochs = []
        CNN = False
        if isinstance(self.model, CNNModel):
            CNN = True
        # hyperparameters
        input_size = 3
        hidden_size = 4
        output_size = 1
        learning_rate = 0.01

        x_train = torch.from_numpy(self.x_train).float().to(self.device)
        y_train = torch.from_numpy(self.y_train).float().to(self.device)

        x_test = torch.from_numpy(self.x_test).float().to(self.device)
        y_test = torch.from_numpy(self.y_test).float().to(self.device)

        # Create a DataLoader to handle batching

        dataset_train = TensorDataset(x_train, y_train)
        dataset_test = TensorDataset(x_test, y_test)

        data_loader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=False)
        data_loader_test = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        mse = nn.MSELoss()

        for epoch in range(self.epochs):
            total_loss = 0
            self.model.train()
            for batch_idx, (batch_data, batch_labels) in enumerate(data_loader_train):  # Iterate through batches
                if CNN:
                    data = batch_data.transpose(1, 2)
                else:
                    data = batch_data
                optimizer.zero_grad()
                # Forward pass
                outputs = self.model(data)

                # Calculate the custom loss with importance
                batch_labels = batch_labels.view(-1, output_size)  # Ensure target size matches output size
                loss = mse(outputs, batch_labels)

                # Backpropagation and optimization

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch_data)

            avg_loss = total_loss / len(data_loader_train)
            self.train_loss.append(avg_loss)

            self.model.eval()
            with torch.no_grad():
                total_val_loss = 0
                for batch_data, batch_labels in data_loader_test:
                    if CNN:
                        data = batch_data.transpose(1, 2)
                    else:
                        data = batch_data
                    outputs = self.model(data)
                    batch_labels = batch_labels.view(-1, output_size)
                    loss = mse(outputs, batch_labels)

                    total_val_loss += loss.item() * len(batch_data)

            avg_val_loss = total_val_loss / len(data_loader_test)
            self.val_loss.append(avg_val_loss)
            print(f'Epoch [{epoch + 1}/{self.epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            num_epochs.append(epoch)
            self.draw_curve(num_epochs)
        print('Training finished.')

        return self.model, self.train_loss, self.val_loss





    # CREATE LOSS CURVES
    def draw_curve(self, num_epochs):
        clear_output(wait=True)
        plt.plot(num_epochs, self.train_loss, label="Train loss")
        plt.plot(num_epochs, self.val_loss, label="Val Loss")
        plt.ylim(0, 1)
        plt.legend()
        plt.show()







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






#CNN Model
class CNNModel(nn.Module):
    def __init__(self, input_size):
        super(CNNModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size[1], out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(384, 100),  # Adjusted for correct output size
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x)
        return x





# DNN Model
class DNNModel(nn.Module):
  def __init__(self, time_steps, features):
    super(DNNModel, self).__init__()
    flattened_data=time_steps*features
    self.flatten=nn.Flatten()
    self.dense_layer1=nn.Linear(flattened_data, 500)
    self.dropout_layer1=nn.Dropout(0.3)
    self.dense_layer2=nn.Linear(500, 250)
    self.dropout_layer2=nn.Dropout(0.3)

    self.output_layer=nn.Linear(250, 1)

  def forward(self, x):
    x = self.flatten(x)
    x = nn.ReLU()(self.dense_layer1(x))
    x = self.dropout_layer1(x)
    x = nn.ReLU()(self.dense_layer2(x))
    x = self.dropout_layer2(x)
    x = self.output_layer(x)
    return x
