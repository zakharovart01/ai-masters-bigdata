#!/opt/conda/envs/dsenv/bin/python

import os
import sys

SPARK_HOME = "/usr/lib/spark3"
PYSPARK_PYTHON = "/opt/conda/envs/dsenv/bin/python"
os.environ["PYSPARK_PYTHON"]= PYSPARK_PYTHON
os.environ["PYSPARK_DRIVER_PYTHON"]= PYSPARK_PYTHON
os.environ["SPARK_HOME"] = SPARK_HOME

PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.9.3-src.zip"))
sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))

from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import *

conf = SparkConf()
spark = SparkSession.builder.config(conf=conf).appName("Spark ML Intro").getOrCreate()

from model import pipeline
from pyspark.sql.types import *

input_data = sys.argv[1]
output_model = sys.argv[2]

schema = StructType(fields=[
    StructField("overall", FloatType()),
    StructField("vote", StringType()),
    StructField("verified", BooleanType()),
    StructField("reviewTime", StringType()),
    StructField("reviewerID", StringType()),
    StructField("asin", StringType()),
    StructField("reviewerName", StringType()),
    StructField("reviewText", StringType()),
    StructField("summary", StringType()),
    StructField("unixReviewTime", IntegerType()),
    ])

df = spark.read\
          .schema(schema)\
          .format("json")\
          .load(input_data)

pipeline_model = pipeline.fit(df)

pipeline_model.write().overwrite().save(output_model)
