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
from pyspark.ml import Transformer
from pyspark.ml import Pipeline
from pyspark.sql import functions as f
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark import keyword_only
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--path-in", action="store", dest="path_in")
parser.add_argument("--path-out", action="store", dest="path_out")
args = parser.parse_args()

input_data = args.path_in
output_data = args.path_out

from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark import SparkContext

sc =SparkContext()
conf = SparkConf()
spark = SparkSession.builder.config(conf=conf).appName("Spark SQL").getOrCreate()
spark.sparkContext.setLogLevel('WARN')

schema = StructType(fields=[
    StructField("id", StringType()),
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
    
class FillNaTransformer(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
    input_col = Param(Params._dummy(), "input_col", "input column name.", typeConverter=TypeConverters.toString)
    output_col = Param(Params._dummy(), "output_col", "output column name.", typeConverter=TypeConverters.toString)
    val = Param(Params._dummy(), "val", "value for filling nan.", typeConverter=TypeConverters.toInt)
  
    @keyword_only
    def __init__(self, input_col: str = "input", output_col: str = "output", val: int = 0):
        super(FillNaTransformer, self).__init__()
        self._setDefault(input_col=None, output_col=None, val = 0)
        kwargs = self._input_kwargs
        self.set_params(**kwargs)
    
    @keyword_only
    def set_params(self, input_col: str = "input", output_col: str = "output", val: int = 0):
        kwargs = self._input_kwargs
        self._set(**kwargs)
    
    def get_input_col(self):
        return self.getOrDefault(self.input_col)
  
    def get_output_col(self):
        return self.getOrDefault(self.output_col)
  
    def get_val(self):
        return self.getOrDefault(self.val)
    
    def _transform(self, df: DataFrame):
        df = df.drop("reviewTime", "reviewerName", "unixReviewTime").cache()
        df = df.withColumn("vote", df["vote"].cast(IntegerType()))
        df = df.withColumn("comment_length", f.length(df.reviewText))
        df = df.na.fill(0)
        return df
    
custom_transformer = FillNaTransformer()
tokenizer = Tokenizer(inputCol="reviewText", outputCol="text")
hasher = HashingTF(numFeatures=100, binary=True, inputCol=tokenizer.getOutputCol(), outputCol="word_vector")

pipeline = Pipeline(stages=[
    custom_transformer,
    tokenizer,
    hasher
])

pipeline_feature_eng = pipeline.fit(df)
df_feature_eng = pipeline_feature_eng.transform(df)
df_feature_eng = df_feature_eng.drop("reviewText", "summary", "text", "reviewerID", "asin", "word_vector")

df_feature_eng.write.mode("overwrite").json(output_data)