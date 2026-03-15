from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
import mlflow
import dagshub
import pandas as pd
import mlflow.sklearn

# mlflow.autolog()
dagshub.init(repo_owner="raaggeee", repo_name="mlflow-demo", mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/raaggeee/mlflow-demo.mlflow")

mlflow.set_experiment("antenna-fault-RandomForest-CV")

df = pd.read_csv("/data2/experiment-tracking/mlflow-demo/data/preprocessed/train_processed.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(random_state=42)

params_grdi = {
    "n_estimators": [10, 20, 30],
    "max_depth": [None, 2, 10, 20, 30, 100]
}

grid_search = GridSearchCV(estimator=rf, param_grid=params_grdi, cv=5, n_jobs=-1, verbose=2)

with mlflow.start_run(run_name="test3"):
    grid_search.fit(X_train, y_train)

    for i in range(len(grid_search.cv_results_["params"])):
        with mlflow.start_run(run_name=f"Exp - {i}", nested=True):
            param = grid_search.cv_results_["params"][i]
            acc = grid_search.cv_results_["mean_test_score"][i]
            mlflow.log_param(f"parameter at {i}th combination", param)
            mlflow.log_metric(f"Metric at {i}th combination", acc)


    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(best_params)
    print(best_score)

    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", best_score)

    new_df = X_train
    new_df["outcome"] = y_train
    mlflow.log_input(mlflow.data.from_pandas(new_df), "training data")

    new_df2 = X_test
    new_df2["outcome"] = y_test
    mlflow.log_input(mlflow.data.from_pandas(new_df2), "test data")

    mlflow.log_artifact(__file__)

    mlflow.set_tag("model", "random forest")
    mlflow.set_tag("cv", "grid search")

    mlflow.sklearn.log_model(grid_search.best_estimator_, "random forest")





