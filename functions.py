import time
import numpy as np
from pyspark.sql import Row
from datetime import datetime
from pyspark.sql.types import *
from pyspark.sql.functions import *

def load_data(ss, fileName, hdr=True, col_names=None, verbose=True):
    t1 = time.time()
    if(verbose): print("{0} - Loading {1} file....".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S"), fileName))
    
    df = ss.read.load("tfm-marta/data/"+fileName, format="csv", sep=",", inferSchema=True, header=hdr)
    df = df.na.replace("?", None)

    if hdr==False:
        for i in range(0, len(df.columns)):
            df = df.withColumnRenamed(df.columns[i], col_names[i])

    if(verbose): print("Loaded {0} rows and {1} columns".format(df.count(), len(df.columns)))
    
    df.show(1)
    
    if(verbose):  print("{0} - Load completed. This operation has taken {1:.2f} s".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S"),time.time()-t1))
    
    return df


def load_ts(sc, numFiles, filePrefix, verbose=True):
    t1 = time.time()
        
    for i in range(1,numFiles+1): 
        if(verbose): print("{0} - Loading {1} file....".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S"), filePrefix+str("-%02d"%i)))

        df_aux = sc.textFile('tfm-marta/data/time-series/'+filePrefix+str("-%02d"%i))
        df_aux = df_aux.map(lambda line: line.split("\t"))
        df_aux = df_aux.map(lambda line: Row(date=line[0], time=line[1], code=line[2], value=line[3])).toDF()
        df_aux = df_aux.withColumn('patient_nbr', lit(70125+i))
        df_aux = df_aux.na.replace("\t", None)
        #df_aux.dropna(how='any', subset=df_aux.columns)
        df_aux.na.drop(subset=df_aux.columns)
        
        if(verbose): print("Loaded {0} rows and {1} columns".format(df_aux.count(), len(df_aux.columns)))

        df_aux = df_aux.withColumn('aux',concat(df_aux.date,lit(' '), df_aux.time))
        df_aux = df_aux.withColumn('aux', to_timestamp(unix_timestamp(df_aux.aux, 'MM-dd-yyyy H:mm')))
        df_aux = df_aux.drop(*['date', 'time'])
        df_aux = df_aux.withColumnRenamed('aux', 'date')

        df_aux = df_aux.na.replace("NaT", None)
        df_aux = df_aux.na.drop(subset=df_aux.columns)
    
        if(i==1): df = df_aux
        else: df = df.union(df_aux)
            
    if(verbose): print("{0} - Load completed. This operation has taken {1:.2f} s".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S"),time.time()-t1))
    
    return df


def impute_na(df, perc=0.3, verbose=True):
    t1 = time.time()
    if(verbose): print("\n{0} - Working with missing values...".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
    
    
    if(verbose): print("Removing rows with more than {0}% of missing data...".format(int(100*perc)))
    n1 = df.count()
    df = df.dropna(thresh=int((1-perc)*len(df.columns)), subset=df.columns) # drop rows with <thresh non-null values 
    n2 = df.count()
    if(verbose): print("{0} rows removed".format(n1-n2))

    NAsdf = df.select([count(when(isnan(i) | col(i).isNull(), i)).alias(i) for i in df.columns]) # Number of NAs/Null
    NAsdf_ = NAsdf.toPandas() # The data to be processed can fit into memory -> use Pandas df
    NAsdf_ = NAsdf_.T # Transpose
    NAsdf_.columns = ['count']
    NAsdf_['density'] = np.round(NAsdf_['count']/df.count(),2)
    if(verbose): print("\nShowing columns with missing values: ")
    if(verbose): print(NAsdf_[NAsdf_['count'] > 0])

    if(verbose): print("\nRemoving rows with more than 30% of missing data (", end ="")
    idx_to_remove = list(filter(lambda x: NAsdf.collect()[0][x]>int(np.floor(0.3*df.count())), list(range(0, len(df.columns)))))
    columns_to_drop = [NAsdf.columns[i] for i in idx_to_remove]

    if(verbose): print(*columns_to_drop, sep=", ", end =""); print(") ...")
    df = df.drop(*columns_to_drop)

    if(verbose): print("\n{0} - Columns removed. \nNew shape: ({1} rows x {2} columns)"\
          .format(datetime.now().strftime("%d-%m-%Y %H:%M:%S"), df.count(), len(df.columns)))

    
    if(verbose): print("\n{0} - Replacing missing values...".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
    
    idx_to_replace = list(filter(lambda x: NAsdf.collect()[0][x] > 0 and NAsdf.collect()[0][x]<=int(np.floor(0.3*df.count())), list(range(0, len(df.columns)))))
    columns_to_replace = [NAsdf.columns[i] for i in idx_to_replace]

    for i in columns_to_replace:
        mode = df.groupBy(i).count().sort("count", ascending=False).first()[0]
        if(verbose): print("Replacing NAs values of attribute {} by value {}".format(i, mode))
        df = df.na.fill({i: mode})

    if(verbose): print("\n{0} - Missing values removed.  This operation has taken {1:.2f} s".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S"), time.time()-t1))

    return df

def df_split(df, weight1):
    splits = df.randomSplit([weight1, 1.0-weight1], 24)
    return splits[0], splits[1]