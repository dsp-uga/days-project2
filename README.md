Team Name: DAYS

Malware Classification

Classification Strategy: Random Forest classifier


Technologies Used:
-----------------
- Python 2.7
- Apache Spark 2.0
- Resilient Distributed Datasets (RDDs)
- Dataframes
- Django API
- PyParsing API


Preprocessing of Byte File:
--------------------------
- The hash files on S3 storage were read from "https://s3.amazonaws.com/eds-uga-csci8360/data/project2/metadata/<filename>"
- The next step was to read the label file
- Create a pair resulting from hash file and label file
- Read the contents of all files specified in the training set
- Cleaning of the .bytes files with the following parser
(i)  removal of pointers
(ii) removal of special characters (??)
- Added content to corresponding hash file and label file
- Cleaned data is saved into a parquet file.
- The parquet file is accessed everytime the data is to be read.


Preprocessing of Opcodes from .asm files:
----------------------------------------
- Two librabries, namely Django and pyparsing have been used for the preporcessing. 
- Django: Django is a python library to facilitate rapid development and pragmatic design. It has been used for the removal of special symbols from the opcodes.
- pyparsing: It's a python library for constructing and executing simple grammar. This has been used for opcode extraction.


Flow:
----
- N-grams are generated from the preprocessed byte file and opcodes.
- Convert the N-grams of the byte file and opcodes to vectors of token counts.
- These vectors are brought together by the Vector Assembler.
- The data is then fed to Random Forest Classifier.
- The prediction is computed.


Tuning the accuracy:
-------------------
- 1,2,3,4 grams of byte file and opcodes were generated in order to test which provides a better accuracy.
- Tuning of the parameters of Random Forest classifier, namely maxDepth, no. of trees, maxBin.



Challenges Faced:
----------------
- Memory issues in cluster (Out-of-memory issue)
- Reading parquet file in cluster
- Tuning of Random Forest Classifier in order to increase accuracy
