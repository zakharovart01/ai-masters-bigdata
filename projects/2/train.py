#!/opt/conda/envs/dsenv/bin/python

import os, sys
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from joblib import dump
import numpy as np

#
# Import model definition
#
from model import model, fields

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
    proj_id = sys.argv[1] 
    train_path ='/home/users/datasets/criteo/criteo_train500'
except:
    logging.critical("Need to pass both project_id and train dataset path")
    sys.exit(1)


logging.info(f"TRAIN_ID {proj_id}")
logging.info(f"TRAIN_PATH {train_path}")

        #
        # Read dataset
        #
        #fields = """doc_id,hotel_name,hotel_url,street,city,state,country,zip,class,price,
        #num_reviews,CLEANLINESS,ROOM,SERVICE,LOCATION,VALUE,COMFORT,overall_ratingsource""".replace("\n",'').split(",")

read_table_opts = dict(sep="\t", names=fields, index_col=False)
df = pd.read_table(train_path, **read_table_opts)

        #split train/test
X_train, X_test, y_train, y_test = train_test_split(
                    df.iloc[:100000,2:], df.iloc[:100000,1], test_size=0.33, random_state=42
                    )

        #
        # Train the model
        #
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_test)

model_score = log_loss(y_test, y_pred[:, 1])

logging.info(f"model score: {model_score:.3f}")

        # save the model
dump(model, "{}.joblib".format(proj_id))
