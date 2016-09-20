from __future__ import print_function
import urllib
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row, SparkSession
from pyspark.ml.feature import IDF, Tokenizer
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.sql.types import *
from pyspark.ml.feature import NGram
from pyspark.mllib.feature import Word2Vec
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from django.utils.encoding import smart_str, smart_unicode
import re
import os
from pyparsing import Word, hexnums, WordEnd, Optional, alphas, alphanums

# sc = SparkContext(conf=SparkConf().setAppName("MalwareClassifier"))
spark = SparkSession \
    .builder \
    .appName("OpcodeExtraction") \
    .getOrCreate()
sc = spark.sparkContext;
sqlContext = SQLContext(sc)


def clean(asmline):
    temp = ""
    file = asmline.split("\r\n")  # Split each line of .asm file into a list by spaces

    hex_integer = Word(hexnums) + WordEnd()  # use WordEnd to avoid parsing leading a-f of non-hex numbers as a hex
    print("1")
    regex = ".text:" + hex_integer + Optional((hex_integer * (1,))("instructions") + Word(alphas, alphanums)("opcode"))
    print("2")
    for source_line in file:
        # print (source_line)
        temp_inp = smart_str(source_line)
        # print (temp_inp)
        try:
            if ".text:" in temp_inp:
                result = regex.parseString(temp_inp)
                if "opcode" in result:
                    if result.opcode not in ['CC', 'align', 'dw', 'db']:
                        #temp += " " + result.opcode
                        temp = " ".join([temp,result.opcode])
        except:
            continue

    return temp


def main():
    # =========================================================================
    # Access Key and secret key necessary to read data from Amazon S3
    # =========================================================================
    sc._jsc.hadoopConfiguration().set('fs.s3n.awsAccessKeyId', 'AKIAJRFT7W3XVW3Z4Z3A')
    sc._jsc.hadoopConfiguration().set('fs.s3n.awsSecretAccessKey', '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')

    # =========================================================================
    # Reading training file from s3
    # =========================================================================
    hashFileData = sc.textFile("s3n://eds-uga-csci8360/data/project2/labels/X_train_small.txt").map(
        lambda doc: doc.encode("utf-8").strip())
    entirehashFileData = hashFileData.zipWithIndex().map(lambda doc: (doc[1], doc[0])).cache()

    # =========================================================================
    # Reading (hashcode).bytes file from s3:
    # 1. Concatenating .bytes extension to each hashcode obtained.
    # 2. Making RDD a string to pass tp wholeTextFile function.
    # 3. Read bytes file from s3 and stored it in RDD format (Filename, FileData)
    # 4. Doing initial cleaning of the data through function cleanDoc()
    # =========================================================================
    asmFile = hashFileData.map(lambda doc: ("s3n://eds-uga-csci8360/data/project2/metadata/" + doc + ".asm"))
    filePath = asmFile.reduce(lambda str1, str2: str1 + "," + str2)
    #asmFileCollect = sc.wholeTextFiles(filePath, 20000)
    asmFileCollect = sc.wholeTextFiles("s3n://eds-uga-csci8360/data/project2/metadata/c2hn9edSNJKmw0OukrBv.asm,s3n://eds-uga-csci8360/data/project2/metadata/hkscPWGaIw0ALpHuNKr8.asm")
    print ("Step 1 done")
    # asmFileCollect.saveAsTextFile("/home/yash/Yash/8.txt")
    # ======
    # Use the below line to test data of byte file
    # byteFileCollect= sc.wholeTextFiles("s3n://eds-uga-csci8360/data/project2/binaries/c2hn9edSNJKmw0OukrBv.bytes",50)
    # ======
    cleanFile = asmFileCollect.map(lambda doc: (doc[0].encode('utf-8'), clean(doc[1]).lstrip(' ')))
    wholeTextFileNameRDD = cleanFile.map(lambda (x, y): (os.path.splitext(os.path.basename(x))[0], y))
    #cleanFile.saveAsTextFile("/home/yash/Yash/clean27.txt")
    print("Step 2 done")

    # =========================================================================
    # Reading label file from s3
    # =========================================================================
    labelData = sc.textFile("s3n://eds-uga-csci8360/data/project2/labels/y_train_small.txt").map(
        lambda doc: doc.encode("utf-8").strip()).cache()
    entireLabelData = labelData.zipWithIndex().map(lambda doc: (doc[1], doc[0]))
    print("Step 3 done")
    # =========================================================================
    # Joining RDD's of HashFile,Label and content
    # =========================================================================

    hashFileLablePair = entirehashFileData.join(entireLabelData, numPartitions=2)
    hashFileLableRDD = hashFileLablePair.values()
    hashFileLableRDDPair = hashFileLableRDD.keyBy(lambda line: line[0]).mapValues(lambda line: line[1])
    dataSet = hashFileLableRDDPair.join(wholeTextFileNameRDD, numPartitions=2)
    finalDataSetRDD = dataSet.map(lambda (x, y): (x, y[0], y[1]))
    #finalDataSetRDD.saveAsTextFile("7.txt")
    print("Step 4 done")
    # =========================================================================
    # creating DATAFRAME
    # =========================================================================
    schemaString = "hashcodefile label content"
    fields = [StructField("hashcodefile", StringType(), True), StructField("label", StringType(), True),
              StructField("content", StringType(), True)]
    schema = StructType(fields)
    schemaOpcode = spark.createDataFrame(finalDataSetRDD, schema)
    schemaOpcodeFinal=schemaOpcode.withColumn("features",schemaOpcode["content"].cast(ArrayType()))
    schemaOpcodeFinal.createOrReplaceTempView("byteDataFrame")

    # =========================================================================
    # Reading and wrir=ting to Parquet file file from s3
    # =========================================================================
    print("Step 5 done")
    schemaOpcodeFinal.write.parquet("cleanOpcodeFile.parquet")
    test = spark.read.parquet("cleanOpcodeFile.parquet")
    test.show(1)
    print("Step 6 done")


if __name__ == "__main__":
    main()
