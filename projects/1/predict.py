#!/opt/conda/envs/dsenv/bin/python

import sys, os
import logging
from joblib import load
import pandas as pd

sys.path.append('.')
from model import real_fields, fields

#
# Init the logger
#

logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

# load the model
model = load("1.joblib")

# numeric_features = ["if" + str(i) for i in range(1, 14)]
# categorical_features = ["cf" + str(i) for i in range(1, 27)] + ["day_number"]
# fields = ["id", "label"] + numeric_features + categorical_features

# read and infere
fields.remove("label")
real_fields.remove("label")

read_opts = dict(sep='\t',
                 names=real_fields,
	             usecols=fields,
                 index_col=False,
                 header=None,
                 iterator=True,
                 chunksize=100)

#cols_to_drop = ['id']

for df in pd.read_csv(sys.stdin, **read_opts):
    pred = model.predict(df)
    out = zip(df.id, pred)
    print("\n".join(["{0}\t{1}".format(*i) for i in out]))
