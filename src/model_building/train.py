import torch.nn as nn
import torch
import yaml 
import pandas as pd
import numpy as np
import os
import mlflow
import matplotlib.pyplot as plt 

class Model(nn.Module):
    def __init__(self, input_shape, hidden_shape, output_shape):
        super().__init__()
        self.input = nn.Linear(input_shape, hidden_shape)
        self.hidden = nn.Linear(hidden_shape, hidden_shape)
        self.output = nn.Linear(hidden_shape, output_shape)

    def forward(self, x):
        x = torch.relu(self.input(x))
        x = torch.relu(self.hidden(x))
        x = self.output(x)

        return x

def train(X_train, y_train, epochs, batch_size=10, X_test=None, y_test=None):
    input_shape = X_train.shape[1] # tenspr
    model = Model(input_shape=input_shape, hidden_shape=10, output_shape=1)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    data_rows = X_train.shape[0]
    criterion = nn.MSELoss()

    losses = []

    if X_test != None and y_test != None:
        accuracy = []
    model.train()
    for i in range(epochs):
        for j in range(0, data_rows, batch_size):
            X = X_train[j:j+batch_size]
            y = y_train[j:j+batch_size]

            y_pred = model(X)

            loss = criterion(y_pred, y)

            loss.backward()
            optim.step()

            optim.zero_grad()

        losses.append(loss.item())

        print(f"Epoch: {i}, Loss: {loss.item()}")
        mlflow.log_metric(f"loss: {i}", loss.item())


    torch.save(model.state_dict(), "models/model.pkl")
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)
    return losses

def params_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        params = yaml.safe_load(f)

    input_path = params["train"]["input_path"]
    output_path = params["train"]["output_path"]

    return input_path, output_path

def open_csv(df_path):
    df = pd.read_csv(df_path)
    return df

def save_csv(df, df_path):
    df.to_csv(df_path, index=False)

def main():
    input_path, _ = params_yaml("config.yaml")

    train_path = os.path.join(input_path, "train_processed.csv")

    train_data = open_csv(train_path)

    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values

    X_train_torch = torch.tensor(X_train).to(torch.float32)
    y_train_torch = torch.tensor(y_train).to(torch.float32).unsqueeze(1)

    with mlflow.start_run(run_name="torch_run1"):
        loss = train(X_train_torch, y_train_torch, epochs=10)

        plt.plot(np.array(loss))
        plt.savefig("losses.png")
        mlflow.log_artifact("losses.png")


if __name__ == "__main__":
    main()





