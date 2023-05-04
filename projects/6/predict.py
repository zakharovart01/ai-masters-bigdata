import os
import sys

SPARK_HOME = "/usr/hdp/current/spark2-client"
PYSPARK_PYTHON = "/opt/conda/envs/dsenv/bin/python"
PYSPARK_DRIVER_PYTHON = "/opt/conda/envs/dsenv/bin/python"
os.environ["PYSPARK_PYTHON"]= PYSPARK_PYTHON
os.environ["PYSPARK_DRIVER_PYTHON"]= PYSPARK_DRIVER_PYTHON
os.environ["SPARK_HOME"] = SPARK_HOME

PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.9.3-src.zip"))
sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))

from pyspark.sql.types import *
from pyspark.ml.feature import *
from pyspark.sql import functions as f
from joblib import load
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pyspark.ml.functions import vector_to_array
import pandas as pd


from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark import SparkContext

sc =SparkContext()
conf = SparkConf()
spark = SparkSession.builder.config(conf=conf).appName("Spark SQL").getOrCreate()
spark.sparkContext.setLogLevel('WARN')

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--test-in", action="store", dest="test_in")
parser.add_argument("--pred-out", action="store", dest="pred_out")
parser.add_argument("--sklearn-model-in", action="store", dest="sklearn_model_in")
args = parser.parse_args()

model_path = args.sklearn_model_in
test_data = args.test_in
output_pred = args.pred_out

model = load(model_path)
clf = sc.broadcast(model)

@f.pandas_udf(FloatType())
def predict(series):
    predictions = clf.value.predict(series.tolist())
    return pd.Series(predictions)

dataset = spark.read.json(test_data)
assembler = VectorAssembler(inputCols=["vote", "verified", "comment_length", "vote"], outputCol="features")
dataset_features = assembler.transform(dataset)
dataset_features = dataset_features.select("id", "features")
predictions = dataset_features.withColumn('predictions', predict(vector_to_array('features')))
predictions.select("id", "predictions").write.mode("overwrite").csv(output_pred)