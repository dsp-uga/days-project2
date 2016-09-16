from __future__ import print_function
import urllib

from pyspark.sql import SQLContext, Row,SparkSession
from pyspark.ml.feature import IDF, Tokenizer,HashingTF
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import *
from pyspark.ml.feature import NGram

import re
import os

warehouseLocation = 'file:///home/sharmin/PycharmProjects/MalwareDetection'
#sc = SparkContext(conf=SparkConf().setAppName("MalwareClassifier"))
spark = SparkSession\
        .builder\
        .appName("MalwareClassification")\
        .getOrCreate()
sc=spark.sparkContext;
sqlContext=SQLContext(sc)

#######Method for Preprocessing
def cleanDoc(bytefileData):
    # Removing unwanted items from the list.
    filteredFile = re.sub("\?|\n|\r", "", bytefileData)

    # Removing line pointers.
    removePointer = [word.encode('utf-8') for word in filteredFile.split() if len(word) < 3]
    return removePointer

def uniqueByte(inputByte):
    x= list(set(inputByte))
    temp=[]
    for str in x:
        tempStr=str.replace(" ","")
        temp.append(tempStr)
    return temp

########Main method for Ma
def main():

    fieldsDF = [StructField("hashcodefile", StringType(), True), StructField("label", StringType(), True),
                StructField("4-grams", StringType(), False)]
    schemaDF = StructType(fieldsDF)
    schemaByteParque=spark.read.parquet("cleanFile.parquet")
    print ("Parquet File read completed")
    schemaByteParque.createOrReplaceTempView("byteDataFrame")
    ngram = NGram(n=4, inputCol="content", outputCol="4-grams")
    ngramDataFrame = ngram.transform(schemaByteParque).select("hashcodefile","label","4-grams")
    print ("4 N-gram approach completed")
    ngramRDD=ngramDataFrame.rdd
    ngramRDDUnique = ngramRDD.map(lambda line: (line[0].encode('utf-8'),line[1], uniqueByte(line[4])))
    print ("unique 4 N-Grams extracted")
    #ngramDFUnique=spark.createDataFrame(ngramRDDUnique,schemaDF)
    #ngramDFUnique.show()
    #ngramDataFrameRDDUnique.saveAsTextFile("file:///home/sharmin/unique.txt");
    ngramDataFrameStringRDD=ngramRDDUnique.map(lambda line:(line[1].encode('utf-8'),' '.join(str(x) for x in line[4])))
    print ("Array type to String Type conversion completed")
    fieldTF = [StructField("label", StringType(), True), StructField("4-grams", StringType(), True)]
    schemaTF = StructType(fieldTF)
    inputTFDF=spark.createDataFrame(ngramDataFrameStringRDD,schemaTF)
    #ngramDataFrameStringRDD.saveAsTextFile("file:///home/sharmin/unique.txt");

    count=ngramRDDUnique.flatMap(lambda line:(' '.join(str(x) for x in line[4])).split(" ")).distinct().count()
    print ("Count:",count)
    #count.saveAsTextFile("file:///home/sharmin/unique.txt");

if __name__ == "__main__":
    main()