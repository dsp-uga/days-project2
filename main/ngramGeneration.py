from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row,SparkSession
from pyspark.ml.feature import NGram
from pyspark.sql.types import *


spark = SparkSession\
        .builder\
        .appName("NGramExample")\
        .config("spark.driver.maxResultSize", "3g")\
        .getOrCreate()
sc=spark.sparkContext;
sqlContext=SQLContext(sc)

def createNgram(schemaByte,value):

	ngram = NGram(n=value, inputCol="content", outputCol="n-grams")
    ngramDataFrame = ngram.transform(schemaByte).select("hashcodefile","label","n-grams")
    ngramDataFrameRDD=ngramDataFrame.rddeturn ngramDataFrameRDD