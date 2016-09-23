from __future__ import print_function
import urllib
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row, SparkSession
import sys
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

###########################################################################
#Method for extracting opcode features
########################################################################
def clean(asmline):
    temp = []
    file = asmline.split("\r\n")  # Split each line of .asm file into a list by spaces
    hex_integer = Word(hexnums) + WordEnd()  # use WordEnd to avoid parsing leading a-f of non-hex numbers as a hex

    regex = ".text:" + hex_integer + Optional((hex_integer * (1,))("instructions") + Word(alphas, alphanums)("opcode"))
    regex1 = ".rdata:" + hex_integer + Optional((hex_integer * (1,))("instructions") + Word(alphas, alphanums)("opcode"))

    for source_line in file:
        temp_inp = smart_str(source_line)

        try:
            if ".text:" in temp_inp:

                result = regex.parseString(temp_inp)
                if "opcode" in result:
                    if result.opcode not in ['CC', 'align', 'dw', 'db']:
                        tempString=re.sub(r'\\x[0-9a-fA-F]+','|',result.opcode)
                        temp.append(tempString)
            if ".rdata:" in temp_inp:
                result1 = regex1.parseString(temp_inp)
                if "opcode" in result1:
                    if result1.opcode not in ['CC', 'align', 'dw', 'db']:
                        temp.append(result1.opcode)
        except:
            continue

    return temp

def geTrainingData(keyId,accessKey,dataset,label):
    # =========================================================================
    # Access Key and secret key necessary to read data from Amazon S3
    # =========================================================================
    sc._jsc.hadoopConfiguration().set('fs.s3n.awsAccessKeyId', keyId)
    sc._jsc.hadoopConfiguration().set('fs.s3n.awsSecretAccessKey', accessKey)

    # =========================================================================
    # Reading training file from s3
    # =========================================================================
    hashFileData = sc.textFile("s3n://eds-uga-csci8360/data/project2/labels/"+dataset+".text").map(
        lambda doc: doc.encode("utf-8").strip())
    entirehashFileData = hashFileData.zipWithIndex().map(lambda doc: (doc[1], doc[0])).cache()

    # =========================================================================
    # Reading (hashcode).bytes file from s3:
    # Cleaning of the data through function cleanDoc()
    # =========================================================================
    asmFile = hashFileData.map(lambda doc: ("s3n://eds-uga-csci8360/data/project2/metadata/" + doc + ".asm"))
    filePath = asmFile.reduce(lambda str1, str2: str1 + "," + str2)
    asmFileCollect = sc.wholeTextFiles(filePath, 40)
    # ======
    #
    # ======
    cleanFile = asmFileCollect.map(lambda doc: (doc[0].encode('utf-8'), clean(doc[1])))
    wholeTextFileNameRDD = cleanFile.map(lambda (x, y): (os.path.splitext(os.path.basename(x))[0], y))

    print("Pre-processing completed for training data")

    # =========================================================================
    # Reading label file from s3
    # =========================================================================
    labelData = sc.textFile("s3n://eds-uga-csci8360/data/project2/labels/y_traning_small.txt").map(
        lambda doc: doc.encode("utf-8").strip())
    entireLabelData = labelData.zipWithIndex().map(lambda doc: (doc[1], doc[0]))
    print("Step 3 done")
    # =========================================================================
    # Joining RDD's of HashFile,Label and content
    # =========================================================================

    hashFileLablePair = entirehashFileData.join(entireLabelData, numPartitions=24)
    hashFileLableRDD = hashFileLablePair.values()
    hashFileLableRDDPair = hashFileLableRDD.keyBy(lambda line: line[0]).mapValues(lambda line: line[1])
    dataSet = hashFileLableRDDPair.join(wholeTextFileNameRDD, numPartitions=24)
    finalDataSetRDD = dataSet.map(lambda (x, y): (x, y[0], y[1]))

    print("Step 4 done")
    # =========================================================================
    # creating DATAFRAME
    # =========================================================================
    fields = [StructField("hashcodefile", StringType(), True), StructField("label", StringType(), True),
              StructField("content", ArrayType(StringType(), False), True)]
    schema = StructType(fields)
    schemaOpcode = spark.createDataFrame(finalDataSetRDD, schema)

    # =========================================================================
    # Reading and wrir=ting to Parquet file file from s3
    # =========================================================================
    print("Step 5 done")
    schemaOpcode.write.parquet("opcodeTraining.parquet")
    print("ASM Features extraction completed for Training data")

def getTestingData(keyId,accesskey,dataset):
    # =========================================================================
    # Access Key and secret key necessary to read data from Amazon S3
    # =========================================================================
    sc._jsc.hadoopConfiguration().set('fs.s3n.awsAccessKeyId', keyId)
    sc._jsc.hadoopConfiguration().set('fs.s3n.awsSecretAccessKey', accesskey)

    # =========================================================================
    # Reading training file from s3
    # =========================================================================
    hashFileData = sc.textFile("s3n://eds-uga-csci8360/data/project2/labels/"+dataset+".text").map(
        lambda doc: doc.encode("utf-8").strip())
    entirehashFileData = hashFileData.zipWithIndex().map(lambda doc: (doc[1], doc[0])).cache()

    # =========================================================================
    # Reading (hashcode).bytes file from s3:
    # Cleaning of the data through function cleanDoc()
    # =========================================================================
    asmFile = hashFileData.map(lambda doc: ("s3n://eds-uga-csci8360/data/project2/metadata/" + doc + ".asm"))
    filePath = asmFile.reduce(lambda str1, str2: str1 + "," + str2)
    asmFileCollect = sc.wholeTextFiles(filePath, 40)
    cleanFile = asmFileCollect.map(lambda doc: (doc[0].encode('utf-8'), clean(doc[1])))
    print("Pre-processing completed for testing data")
    wholeTextFileNameRDD = cleanFile.map(lambda (x, y): (os.path.splitext(os.path.basename(x))[0], y))

    # =========================================================================
    # creating Dataframe
    # =========================================================================
    fields = [StructField("hashcodefile", StringType(), True),
              StructField("content", ArrayType(StringType(), False), True)]
    schema = StructType(fields)
    schemaOpcode = spark.createDataFrame(wholeTextFileNameRDD, schema)
    schemaOpcode.write.parquet("opcodeTesting.parquet")
    print ("ASM feature extarction completed for Testing data")

def main(args):
    if(args[2]=='Testing'):
        getTestingData(keyId=args[0],accesskey=args[1],dataset=args[4])
    if(args[3]=='Training'):
        geTrainingData(keyId=args[0],accessKey=args[1],dataset=args[4],label=args[5])

if __name__ == "__main__":
    main(sys.argv)
