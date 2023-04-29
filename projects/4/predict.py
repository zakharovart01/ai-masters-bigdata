#!/opt/conda/envs/dsenv/bin/python

import os
import sys
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from pyspark.ml import Pipeline, PipelineModel

model_path = sys.argv[1] 
test_data = sys.argv[2]
output = sys.args[3]

model = PipelineModel.load(model_path)

from train import schema

df = spark.read\
          .schema(schema)\
          .format("json")\
          .load(test_data)

predictions = model.transform(df)
predictions.write.json(output)