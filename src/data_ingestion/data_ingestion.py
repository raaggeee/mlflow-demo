import pandas as pd
import numpy as np
import os
import yaml
import mlflow

from sklearn.model_selection import train_test_split

##open the raw file and split it in train and test csv files

def params_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        params = yaml.safe_load(f)

    input_path = params["data_ingestion"]["input_path"]
    output_path = params["data_ingestion"]["output_path"]

    return input_path, output_path

def open_csv(df_path):
    df = pd.read_csv(df_path)
    return df

def save_csv(df, df_path):
    df.to_csv(df_path, index=False)
    

def main():
    input_path, output_path = params_yaml("config.yaml")

    input_df = os.path.join(input_path, "antenna_fault.csv")

    df = open_csv(input_df)

    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    os.makedirs(output_path, exist_ok=True)
    train_data_path = os.path.join(output_path, "train.csv")
    test_data_path = os.path.join(output_path, "test.csv")
    save_csv(train_data, train_data_path)
    save_csv(test_data, test_data_path)

if __name__ == "__main__":
    main()




    