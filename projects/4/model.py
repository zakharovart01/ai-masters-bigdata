#!/opt/conda/envs/dsenv/bin/python

from pyspark.sql.types import *
from pyspark.ml.feature import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Estimator, Transformer
from pyspark.ml import Pipeline
from pyspark.sql import functions as f
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark import keyword_only

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
assembler = VectorAssembler(inputCols=[hasher.getOutputCol(), "comment_length", "vote"], outputCol="features")
lr = LogisticRegression(labelCol="overall", maxIter=25)

pipeline = Pipeline(stages=[
    custom_transformer,
    tokenizer,
    hasher,
    assembler,
    lr
])
