from __future__ import print_function
from pyspark.sql import SQLContext, Row, SparkSession
import sys
from pyspark.sql.types import *
import re
import os


# sc = SparkContext(conf=SparkConf().setAppName("MalwareClassifier"))
spark = SparkSession \
    .builder \
    .appName("OpcodeExtraction") \
    .getOrCreate()
sc = spark.sparkContext;
sqlContext = SQLContext(sc)
##############################################
#Method for preprocessing byte features from testing dataset
##########################################3
def cleanDoc(bytefileData):
    # Removing unwanted items from the list.
    filteredFile = re.sub("\?|\n|\r", "", bytefileData)
    # Removing line pointers.
    removePointer = [word.encode('utf-8') for word in filteredFile.split() if len(word) < 3]
    return removePointer


def getTrainingData(keyId,accessKey,dataset,label):
    # =========================================================================
    # Access Key and secret key necessary to read data from Amazon S3
    # =========================================================================
    sc._jsc.hadoopConfiguration().set('fs.s3n.awsAccessKeyId', keyId)
    sc._jsc.hadoopConfiguration().set('fs.s3n.awsSecretAccessKey', accessKey)

    hashFileData = sc.textFile("s3n://eds-uga-csci8360/data/project2/labels/"+dataset+".txt").map(
        lambda doc: doc.encode("utf-8").strip())
    entirehashFileData = hashFileData.zipWithIndex().map(lambda doc: (doc[1], doc[0])).cache()

    # =========================================================================
    # Reading (hashcode).bytes file from s3:
    # Cleaning of the data through function cleanDoc()
    # =========================================================================

    byteFile = hashFileData.map(lambda doc: ("s3n://eds-uga-csci8360/data/project2/binaries/" + doc + ".bytes"))
    filePath = byteFile.reduce(lambda str1, str2: str1 + "," + str2)
    byteFileCollect = sc.wholeTextFiles(filePath,36)
    cleanFile = byteFileCollect.map(lambda doc: (doc[0].encode('utf-8'), cleanDoc(doc[1])))
    wholeTextFileNameRDD = cleanFile.map(lambda (x, y): (os.path.splitext(os.path.basename(x))[0], y))
    print("Step 2 done")

    # =========================================================================
    # Reading label file from s3
    # =========================================================================
    labelData = sc.textFile("s3n://eds-uga-csci8360/data/project2/labels/"+label+".txt").map(
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
    print("Step 4 done")
    # =========================================================================
    # creating DATAFRAME
    # =========================================================================
    schemaString = "hashcodefile label content"
    fields = [StructField("hashcodefile", StringType(), True), StructField("label", StringType(), True),
              StructField("content", ArrayType(StringType(), False), True)]
    schema = StructType(fields)
    schemaByte = spark.createDataFrame(finalDataSetRDD, schema)

    schemaByte.write.parquet("trainingData.parquet")

    print("Training data preprocessing completed")


#########################################################
#Method for parquet file generation og Testing dataset
#########################################################
def getTestingData(keyId,accessKey,dataset):
    # =========================================================================
    # Access Key and secret key necessary to read data from Amazon S3
    # =========================================================================
    sc._jsc.hadoopConfiguration().set('fs.s3n.awsAccessKeyId', keyId)
    sc._jsc.hadoopConfiguration().set('fs.s3n.awsSecretAccessKey',accessKey)

    # =========================================================================
    # Reading training file from s3
    # =========================================================================
    hashFileTestData = sc.textFile("s3n://eds-uga-csci8360/data/project2/labels/"+dataset+".txt").map(
        lambda doc: doc.encode("utf-8").strip())

    # =========================================================================
    # Reading (hashcode).bytes file from s3
    # Read bytes file from s3 and stored it in RDD format (Filename, FileData)
    # Cleaning of the data through function cleanDoc()
    # =========================================================================

    byteTestFile = hashFileTestData.map(lambda doc: ("s3n://eds-uga-csci8360/data/project2/binaries/" + doc + ".bytes"))
    testFilePath = byteTestFile.reduce(lambda str1, str2: str1 + "," + str2)
    byteTestFileCollect = sc.wholeTextFiles(testFilePath, 36)

    cleanTestFile = byteTestFileCollect.map(lambda doc: (doc[0].encode('utf-8'), cleanDoc(doc[1])))
    wholeTestTextFileNameRDD = cleanTestFile.map(lambda (x, y): (os.path.splitext(os.path.basename(x))[0], y))

    # =========================================================================
    # creating DATAFRAME
    # =========================================================================
    schemaString = "hashcodefile label features"
    fields = [StructField("hashcodefile", StringType(), True),
              StructField("features", ArrayType(StringType(), False), True)]
    schema = StructType(fields)
    schemaTestByte = spark.createDataFrame(wholeTestTextFileNameRDD, schema)
    # =========================================================================
    # Reading and writing to Parquet file file from s3
    # =========================================================================
    schemaTestByte.write.parquet("byteTestFile.parquet")


def main(args):
    if(args[2]=='Testing'):
        getTestingData(keyId=args[0],accesskey=args[1],dataset=args[4])
    if(args[3]=='Training'):
        getTrainingData(keyId=args[0],accessKey=args[1],dataset=args[4],label=args[5])

if __name__ == "__main__":
    main(sys.argv)
