#!/opt/conda/envs/dsenv/bin/python

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import *
from pyspark.ml.feature import *
import pyspark.sql.functions as f


tokenizer = Tokenizer(inputCol="reviewText",
                      outputCol="reviewTextTokenized")

stop_word_remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(),
                                     outputCol="reviewTextFiltered",
                                     stopWords=StopWordsRemover.loadDefaultStopWords("english"))

count_vectorizer = CountVectorizer(inputCol=stop_word_remover.getOutputCol(),
                                   outputCol="reviewVector",
                                   binary=True,
                                   vocabSize=2500)

vector_assembler = VectorAssembler(inputCols=[count_vectorizer.getOutputCol(), 'verified'],
                                   outputCol="features")

lr = LogisticRegression(featuresCol="reviewVector",
                        labelCol="overall",
                        maxIter=10,
                        regParam=0)

evaluator = RegressionEvaluator(labelCol="overall",
                                predictionCol="prediction",
                                metricName='rmse')

pipeline = Pipeline(stages=[
                    tokenizer,
                    stop_word_remover,
                    count_vectorizer,
                    vector_assembler,
                    lr
])
