# days-project2
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







tf-idf ml

ngram
extract unique features from returned features
tf-idf - brought it to sparse vector
random forest


