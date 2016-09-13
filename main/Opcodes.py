

############
#Author @Yash
############

#Initializing Spark Context
from pyspark import SparkContext
from pyspark import ml
sc = SparkContext("local", "Simple App")


#############################
#Loading the data into the RDD
##############################
asmrdd =sc.textFile("C:\Users\Yash\Desktop\YS\Studies\UGA\Fall 2016\Data Science Practicum\Project 2").map(lambda doc:doc.encode("utf-8").strip())
#asmrdd.saveAsTextFile("C:\\Users\\Yash\\Desktop\\asmfile.txt")

########################################
#-Function to extract the "OPCODES" form .asm file
#-The function works on the observation that opcodes are only present in a line if and only if
#there are address address bytes, therefore the function tries to extract the opcodes corresponding
#the bytes.
#####################################
def clean(asmline):
    line = asmline.split() #Split each line of .asm file into a list by spaces

    #If nothing or only one element present, return none
    if len(line) is 0 or len(line) is 1:
        return None

    try:
        #Functions outputs the corresponding INTEGER according to the hexadecimal input, otherwise throws exception.
        int(line[1], 16)
        try:
            for i in range(2,10,1): #Enters the loop if 2nd element is a valid Hexadecimal

                int(line[i],16) #Again checks for the hexadecimal and keep on iteration until it brakes
                if line[i] == "db" or line[i] == "dd": #"db" and "dd" are valid hexadecimals but garbage for this domain, therefore return none.
                    return None
                if line[i] == "add":
                    return "add"
                continue
            return None
        except:
            #Exception caught, for the for loop try.
            if line[i] =="dw" or line[i] =="align" or line[i] ==";":
                return None
            #Return the opcode
            return line[i]
    except:
        return None

###################
#Calling map on the .asm file RDD to extract the opcodes.
####################
cleanRDD=asmrdd.map(lambda doc: clean(doc))
#new.saveAsTextFile("C:\\Users\\Yash\\Desktop\\asmfile.txt")

######################
#Filtering the opcodes RDD by eliminating all the None objects.
##################
filteredRdd = cleanRDD.filter(lambda doc: doc is not None and len(doc)<6)
filteredRdd.saveAsTextFile("C:\\Users\\Yash\\Desktop\\asmfile.txt")

