import os, sys
import logging
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss
from sklearn.linear_model import SGDClassifier

if __name__ == "__main__":
    mlflow.set_tracking_uri('http://localhost:6028')
    #
    # Logging initialization
    #
    logging.basicConfig(level=logging.DEBUG)
    logging.info("CURRENT_DIR {}".format(os.getcwd()))
    logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
    logging.info("ARGS {}".format(sys.argv[1:]))
    #
    # Read script arguments
    #
    try:
        train_path = sys.argv[1]
        model_param1 = sys.argv[2]
    except:
        logging.critical("Need to pass train path, model_param1 and output_model path")
        sys.exit(1)

    logging.info(f"TRAIN_PATH {train_path}")
    logging.info(f"model_param1 {model_param1}")

    # We create the preprocessing pipelines for both numeric and categorical data.
    numeric_features = ["if"+str(i) for i in range(1, 14)]
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    categorical_features = ["cf"+str(i) for i in range(1, 27)] + ["day_number"]
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    #
    # Dataset fields
    #
    fields = ["id", "label"] + numeric_features + categorical_features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ]
    )
    #
    # Read dataset
    #
    read_table_opts = dict(sep="\t", names=fields, index_col=False)
    df = pd.read_table(train_path, **read_table_opts)
    #split train/test
    train_headers = [0]+[i for i in range(2, len(fields))]
    X_train, X_test, y_train, y_test = train_test_split(
        df.iloc[:,train_headers], df.iloc[:,1], test_size=0.33, random_state=42
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('SGDClassifier', SGDClassifier(max_iter=int(model_param1)))
    ])
    model.fit(X_train, y_train)
    predicted_qualities = model.predict(X_test)

    metric = log_loss(y_test, predicted_qualities)
    # Mlflow logging
    mlflow.log_metric("log_loss", metric)
    mlflow.sklearn.log_model(model, artifact_path="model")
    mlflow.log_param("model_param1", model_param1)
