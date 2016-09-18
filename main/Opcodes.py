
from __future__ import print_function
import urllib
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row,SparkSession
from pyspark.ml.feature import IDF, Tokenizer
from pyspark.mllib.feature import HashingTF,IDF
from pyspark.sql.types import *
from pyspark.ml.feature import NGram
from pyspark.mllib.feature import Word2Vec
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
import re
import os

warehouseLocation = 'file:///home/dharamendra/PycharmProjects/MalwareDetection'
#sc = SparkContext(conf=SparkConf().setAppName("MalwareClassifier"))
spark = SparkSession\
        .builder\
        .appName("NGramExample")\
        .config('spark.sql.warehouse.dir',warehouseLocation)\
        .getOrCreate()
sc=spark.sparkContext;
sqlContext=SQLContext(sc)


def clean(asmline):
    temp = ""
    f = asmline.split("\n") #Split each line of .asm file into a list by spaces

    for l in file:
        line = l.split()
        #If nothing or only one element present, return none
        if len(line) is 0 or len(line) is 1:
            temp += ""

        try:
            #Functions outputs the corresponding INTEGER according to the hexadecimal input, otherwise throws exception.
            int(line[1], 16)
            try:
                for i in range(2,10,1): #Enters the loop if 2nd element is a valid Hexadecimal

                    int(line[i],16) #Again checks for the hexadecimal and keep on iterating until it brakes
                    if line[i] == "db" or line[i] == "dd": #"db" and "dd" are valid hexadecimals but garbage for this domain, therefore return none.
                        temp += ""
                    if line[i] == "add":
                        temp += " add"
                    continue
                temp += ""
            except:
                #Exception caught, for the for loop try.
                if line[i] =="dw" or line[i] =="align" or line[i] ==";":
                    temp += ""
                #Return the opcode
                temp += " "+str(line[1])
        except:
            temp += ""
    return temp


def main():
    # =========================================================================
    # Access Key and secret key necessary to read data from Amazon S3
    # =========================================================================
    ***REMOVED***
    ***REMOVED***

    # =========================================================================
    # Reading training file from s3
    # =========================================================================
    hashFileData = sc.textFile("s3n://eds-uga-csci8360/data/project2/labels/X_train_small.txt").map(lambda doc: doc.encode("utf-8").strip())
    entirehashFileData = hashFileData.zipWithIndex().map(lambda doc:(doc[1],doc[0])).cache()

    # =========================================================================
    # Reading (hashcode).bytes file from s3:
    # 1. Concatenating .bytes extension to each hashcode obtained.
    # 2. Making RDD a string to pass tp wholeTextFile function.
    # 3. Read bytes file from s3 and stored it in RDD format (Filename, FileData)
    # 4. Doing initial cleaning of the data through function cleanDoc()
    # =========================================================================
    asmFile = hashFileData.map(lambda doc: ("s3n://eds-uga-csci8360/data/project2/metadata/" + doc + ".asm"))
    filePath = asmFile.reduce(lambda str1, str2: str1 + "," + str2)
    #byteFileCollect = sc.wholeTextFiles(filePath, 20000)
    asmFileCollect = sc.wholeTextFiles("s3n://eds-uga-csci8360/data/project2/metadata/hkscPWGaIw0ALpHuNKr8.asm,s3n://eds-uga-csci8360/data/project2/metadata/G31czXvpnwUfRtdJ4TFs.asm,s3n://eds-uga-csci8360/data/project2/metadata/dETSCuIZDapLP9AlJ7o6.asm,s3n://eds-uga-csci8360/data/project2/metadata/F3Zj217CLRxgi0NyHMY4.asm,s3n://eds-uga-csci8360/data/project2/metadata/c2hn9edSNJKmw0OukrBv.asm")
    # ======
    # Use the below line to test data of byte file
    # byteFileCollect= sc.wholeTextFiles("s3n://eds-uga-csci8360/data/project2/binaries/c2hn9edSNJKmw0OukrBv.bytes",50)
    # ======
    cleanFile = asmFileCollect.map(lambda doc: (doc[0].encode('utf-8'), clean(doc[1]))).cache()
    wholeTextFileNameRDD = cleanFile.map(lambda (x, y): (os.path.splitext(os.path.basename(x))[0], y))