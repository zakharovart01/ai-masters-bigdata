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

start = sys.argv[1] 
finish = sys.argv[2]
input = sys.argv[3] 
output = sys.argv[4]

from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.types import *

sc =SparkContext()
conf = SparkConf()
spark = SparkSession.builder.config(conf=conf).appName("Spark SQL").getOrCreate()

schema = StructType(fields=[
    StructField("user_id", IntegerType()),
    StructField("follower_id", IntegerType())])

df = spark.read\
          .schema(schema)\
          .format("csv")\
          .option("sep", "\t")\
          .load(input)

def bfs(graph, start, end):
    res = []
    queue = []
    queue.append([[start],0])
    level_end = -1
    while queue:
        path_level = queue.pop(0)
        path = path_level[0]
        level = path_level[1]
        if level_end != -1 and level > level_end - 1:
            break
        node = path[-1]
        users_of_node = graph.filter(graph.follower_id == node).select('user_id').rdd.toLocalIterator()
        for adjacent in users_of_node:
            new_path = list(path)
            new_path.append(adjacent.user_id)
            queue.append([new_path,level+1])
            
            if adjacent.user_id == end:
                res.append(new_path)
                level_end = level+1
            else:
                continue
    return res

rep = bfs(df, 12, 34)
df_rep = spark.createDataFrame(data=rep)
df_rep.write.csv(output, sep=',')


spark.stop()