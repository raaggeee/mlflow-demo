import pandas as pd
import numpy as np
import os
import yaml
import mlflow
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder

##open the raw file and split it in train and test csv files

def params_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        params = yaml.safe_load(f)

    input_path = params["data_transformation"]["input_path"]
    output_path = params["data_transformation"]["output_path"]

    return input_path, output_path

def open_csv(df_path):
    df = pd.read_csv(df_path)
    return df

def save_csv(df, df_path):
    df.to_csv(df_path, index=False)

def main():
    input_path, output_path = params_yaml("config.yaml")

    train_df_path = os.path.join(input_path, "train.csv")
    test_df_path = os.path.join(input_path, "test.csv")

    train_df = open_csv(train_df_path)
    test_df = open_csv(test_df_path)

    ohe = OneHotEncoder()
    scaler = StandardScaler()

    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]

    train_cat_cols = X_train.select_dtypes(include="object")
    train_cat_ohe = pd.DataFrame(ohe.fit_transform(train_cat_cols).toarray())
    train_int_cols = X_train.select_dtypes(exclude="object")
    train_int_scale = pd.DataFrame(scaler.fit_transform(train_int_cols))

    X_train = pd.concat([train_cat_ohe, train_int_scale], axis=1)
    train_df = pd.DataFrame(X_train)
    train_df["target"] = y_train

    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    test_cat_cols = X_test.select_dtypes(include="object")
    test_cat_ohe = pd.DataFrame(ohe.transform(test_cat_cols).toarray())
    test_int_cols = X_test.select_dtypes(exclude="object")
    test_int_scale = pd.DataFrame(scaler.transform(test_int_cols))

    X_test = pd.concat([test_cat_ohe, test_int_scale], axis=1)
    test_df = pd.DataFrame(X_test)
    test_df["target"] = y_test

    os.makedirs(output_path, exist_ok=True)
    train_save = os.path.join(output_path, "train_processed.csv")
    test_save = os.path.join(output_path, "test_processed.csv")

    save_csv(train_df, train_save)
    save_csv(test_df, test_save)

    ##---WITHOUT PCA---##


if __name__ == "__main__":
    main()


