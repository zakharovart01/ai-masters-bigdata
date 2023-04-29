#!/opt/conda/envs/dsenv/bin/python

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

input_data = sys.argv[1] 
output_model = sys.argv[2]

from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark import SparkContext

sc =SparkContext()
conf = SparkConf()
spark = SparkSession.builder.config(conf=conf).appName("Spark SQL").getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from model import pipeline
from pyspark.sql.types import *

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