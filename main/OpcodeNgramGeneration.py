from __future__ import print_function
import urllib

from ngramGeneration import createNgram
from pyspark.sql import SQLContext, Row,SparkSession
from pyspark.ml.feature import IDF, Tokenizer,HashingTF
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import *
from pyspark.ml.feature import CountVectorizer
from pyspark.sql.functions import lit
from pyspark.ml.feature import NGram
import sys

import re
import os


spark = SparkSession\
        .builder\
        .appName("MalwareClassification")\
        .config("spark.driver.maxResultSize", "3g")\
        .getOrCreate()
sc=spark.sparkContext;
sqlContext=SQLContext(sc)


def main():
    
    #-------------------------------------
    # Ngram Generation of Testing Data
    #-------------------------------------

    fieldsTest = [StructField("hashcodefile", StringType(), True),
              StructField("n-grams", ArrayType(StringType(), True), True)]

    schemaTest = StructType(fieldsTest)

    #args[0]= PreProcessed parquet file of Opcode Testing Data
    opcodeTestParque = spark.read.parquet(args[0])                                                                                     

    #Unigram
    inputTestN1gram = spark.createDataFrame(opcodeTestParque, schemaTest)

    cvTest = CountVectorizer(inputCol="n-grams", outputCol="opcodeFeatures", vocabSize=256, minDF=1.0,minTF=2.0)
    model1Test = cvTest.fit(inputTestN1gram)
    featurizedTest1Data = model1Test.transform(inputTestN1gram).select("hashcodefile","opcodeFeatures")
    featurizedTest1Data.write.parquet("byte1GramFinalTesting.parquet")
  
    #2Gram
    n2gramTestDataRDD=createNgram(opcodeTestParque,2)
    inputTestN2gram = spark.createDataFrame(n2gramTestDataRDD, schemaTest)

    cvTest1 = CountVectorizer(inputCol="n-grams", outputCol="opcodeFeatures", vocabSize=65536, minDF=1.0,minTF=2.0)
    model2Test = cvTest1.fit(inputTestN2gram)
    featurizedTest2Data = model2Test.transform(inputTestN2gram).select("hashcodefile","opcodeFeatures")
    featurizedTest2Data.write.parquet("byte2GramFinalTesting.parquet")

    #3Gram
    n3gramTestDataRDD=createNgram(opcodeTestParque,3)
    inputTestN3gram = spark.createDataFrame(n3gramTestDataRDD, schemaTest)

    model3Test = cvTest1.fit(inputTestN3gram)
    featurizedTest3Data = model3Test.transform(inputTestN3gram).select("hashcodefile","opcodeFeatures")
    featurizedTest3Data.write.parquet("byte3GramFinalTesting.parquet")

    #4Gram
    n4gramTestDataRDD=createNgram(opcodeTestParque,4)
    inputTestN4gram = spark.createDataFrame(n4gramTestDataRDD, schemaTest)

    model4Test = cvTest1.fit(inputTestN4gram)
    featurizedTest4Data = model4Test.transform(inputTestN4gram).select("hashcodefile","opcodeFeatures")
    featurizedTest4Data.write.parquet("byte4GramFinalTesting.parquet")

    #--------------------------------------
    # Ngram Generation of Testing Data
    #--------------------------------------


    #-------------------------------------
    # Ngram Generation of Training Data
    #-------------------------------------

    fields = [StructField("hashcodefile", StringType(), True),StructField("label", StringType(), True),
              StructField("n-grams", ArrayType(StringType(), True), True)]

    schema = StructType(fields)

    #args[0]= PreProcessed parquet file of Opcode Training Data
    opcodeTrainParque=spark.read.parquet(args[1])

    #Unigram
    inputTrainN1gram = spark.createDataFrame(opcodeTrainParque, schema)

    cv = CountVectorizer(inputCol="n-grams", outputCol="opcodeFeatures", vocabSize=256, minDF=1.0,minTF=2.0)
    model1Train = cv.fit(inputTrainN1gram)
    featurizedTrain1Data = model1Train.transform(inputTrainN1gram).select("hashcodefile","label","opcodeFeatures")
    featurizedTrain1Data.write.parquet("byte1GramFinalTraining.parquet")

    #2gram
    n2gramTrainDataRDD=createNgram(opcodeTrainParque,2)
    inputTrainN2gram = spark.createDataFrame(n2gramTrainDataRDD, schema)

    cv1 = CountVectorizer(inputCol="n-grams", outputCol="opcodeFeatures", vocabSize=65536, minDF=1.0,minTF=2.0)
    model2Train = cv1.fit(inputTrainN2gram)
    featurizedTrain2Data = model2Train.transform(inputTrainN2gram).select("hashcodefile","label","opcodeFeatures")
    featurizedTrain2Data.write.parquet("byte2GramFinalTraining.parquet")

    #3 gran
    n3gramTrainDataRDD=createNgram(opcodeTrainParque,3)
    inputTrainN3gram = spark.createDataFrame(n3gramTrainDataRDD, schema)

    model3Train = cv1.fit(inputTrainN3gram)
    featurizedTrain3Data = model3Train.transform(inputTrainN3gram).select("hashcodefile","label","opcodeFeatures")
    featurizedTrain3Data.write.parquet("byte3GramFinalTraining.parquet")

    # 4 gram
    n4gramTrainDataRDD=createNgram(opcodeTrainParque,4)
    inputTrainN4gram = spark.createDataFrame(n4gramTrainDataRDD, schema)

    model4Train = cv1.fit(inputTrainN4gram)
    featurizedTrain4Data = model4Train.transform(inputTrainN4gram).select("hashcodefile","label","opcodeFeatures")
    featurizedTrain4Data.write.parquet("byte4GramFinalTraining.parquet")

    #---------------------------------------
    # Ngram Generation of Training Data
    #---------------------------------------
    


if __name__ == "__main__":
    main(sys.argv)
