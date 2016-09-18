Team Name: DAYS

Team Members: 
Shubhi Jain
Sharmin Pathan
Yash Shrivastava
Dharamendra Kumar

Malware Classification

Classification Strategy: Random Forest classifier


Technologies Used:
-----------------
- Python 2.7
- Apache Spark 2.0
- Resilient Distributed Datasets (RDDs)
- Datasets
- Java 1.8


Preprocessing:
-------------
- The hash files were read from S3 storage from "https://s3.amazonaws.com/eds-uga-csci8360/data/project2/metadata/<filename>
- The next step was to read the label file
- Create a pair resulting from hash file and label file
- Read the contents of all the files specified in the training set
- Cleaning of the .bytes files with the following parser
(i)  removal of pointers
(ii) removal of special characters (??)
- Added content to corresponding hash file and label file
- Cleaned data is saved into a parquet file.
- The parquet file is accessed everytime the data is to be read.


Flow:
----
- N-grams: 2,3 and 4 ngrams have been generated from the parquet file.
- TF - IDF applied to the parquet file.
- Extracted unique features from the returned features
- Applied random forest

