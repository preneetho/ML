#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 17:18:57 2020

@author: preneeth
"""

import pandas as pd
import numpy as np
import pickle


#dataframe to hold input data from a excel. This is for testing the moldel with data from excel (1 row)
df = pd.DataFrame()
#this df holds the processed input data.
final_df = pd.DataFrame()
#this df holds the actual data for preprocessing inout data.
masterDataDF = pd.read_csv('../data/innercity.csv')

    
# Derive Age of the house at the time of sale from dayhours.
def processDayhours():
    df["yr_sold"] = df["dayhours"].apply(lambda x:x[:4]).astype(int)
    df["age_sold"] = df.yr_sold - df.yr_built
    
    
# Set Categorical columns
def setCategoricalColumns():
    df.coast = pd.Categorical(df.coast)
    df.condition = pd.Categorical(df.condition, ordered=True)
    df.quality = pd.Categorical(df.quality, ordered=True)
    df.furnished = pd.Categorical(df.furnished)
    df.zipcode = pd.Categorical(df.zipcode)
    df.yr_built = pd.Categorical(df.yr_built)
    df.yr_sold = pd.Categorical(df.yr_sold)

# Method to Mean encoding of categorical columns
def addMeanEncodedFeature (indFeatureName):
    global df
    
    grpDF = pd.read_csv('../data/'+indFeatureName+'.csv')
    
    grpDF.rename(columns = {indFeatureName:'key', 0:"val"}, inplace = True) 
    grpDF.set_index("key", inplace = True) 
   
    
    lookup = str(df.loc[0,indFeatureName])
    if ((indFeatureName == 'furnished') or (indFeatureName == 'coast')):
        lookup = df.loc[0,indFeatureName]
    
    
    lookupval = grpDF.at[lookup, 'val'] 
    df.loc[:,indFeatureName+'_enc'] = lookupval


# Method for processing Age Sold Feature
def binAgeSold():
    df['age_sold_quantile_bin'] = df.apply(lambda val: round((val['age_sold'] / 10))*10, axis=1 )
    #masterDataDF['age_sold_quantile_bin'] = masterDataDF.apply(lambda val: round((val['age_sold'] / 10))*10, axis=1 )
    addMeanEncodedFeature(df['age_sold_quantile_bin'].name)

# Method for processing Lat and long Feature
def binLatLong():
    
    lat_long_df = pd.read_csv('../data/lat_long_df.csv')
    lat_long_df.rename(columns = {0:"val"}, inplace = True) 
    lat_long_df.set_index("key", inplace = True) 
    lat_long_df["val"]= lat_long_df["val"].astype(str) 
    
    
    longmin = float(lat_long_df.at['longmin', 'val'] )
    latmin = float(lat_long_df.at['latmin', 'val'] )
    
    
    df['long_bin'] = df['long'].apply(lambda val: round(( abs(longmin) - abs(val)) /.2))
    df['lat_bin'] = df['lat'].apply(lambda val: round(( abs(val) - abs(latmin) )/.2))
    df['region'] = df.apply (lambda row: str(row['long_bin'])+'-'+str(row['lat_bin']), axis=1)
    #df.region = pd.Categorical(df.region).codes
    df['Region_name']= df.apply (lambda row: "Region"+'-'+str(row['region']), axis=1)
    addMeanEncodedFeature(df['Region_name'].name)
  
# Method for processing condition Feature
def binCondition():
    
    conditions_df = [
    df['condition'] == 1,
    df['condition'] == 2,
    df['condition'] == 3,
    df['condition'] == 4,
    df['condition'] == 5
    ]

    choices = ['Bad', 'Bad', 'Average', 'Average', 'Good']
   
    df['condition_bin'] = np.select(conditions_df, choices)
    addMeanEncodedFeature(df['condition_bin'].name)
  
# Method for processing Quality Feature
def binQuality():
    
    conditions_df = [
    df['quality'] < 7,
    df['quality'] == 7,
    df['quality'] == 8,
    df['quality'] == 9,
    df['quality'] == 10,
    df['quality'] > 10
    ]
    
    
    choices = ['Bad', 'Average','Average','Average','Average', 'Good']
   
    df['quality_bin'] = np.select(conditions_df, choices)
    
    addMeanEncodedFeature(df['quality_bin'].name)

# Method for processing bed room Feature    
def binBedRooms():
    
    
    conditions_df = [
    df['room_bed'] < 3,
    df['room_bed'] == 3,
    df['room_bed'] == 4,
    df['room_bed'] == 5,
    df['room_bed'] == 6,
    df['room_bed'] > 6]

    choices = ['Small','Average','Average','Large','Large','Large']
    
    df['room_bed_bin'] = np.select(conditions_df, choices)
    addMeanEncodedFeature(df['room_bed_bin'].name)

#method to return bath type
def getBathType(x):
   
    if (x < 2):
        return "1_Bath"
    elif (x >= 2 and x <3):
        return "2_Bath"
    elif (x >= 3):
        return "3_Bath"
    else :
        return
    
    
# Method for processing bath room Feature    
def binBath():
    
    df['room_bath_bin'] = df['room_bath'].apply(lambda val: getBathType(val))
    addMeanEncodedFeature(df['room_bath_bin'].name)
    
    
def getCeilType(x):
    if (x <= 1):
        return "1_Floor"
    elif (x > 1 and x <= 2):
        return "2_Floor"
    elif (x > 2):
        return "3_Floor"
    else :
        return

def binCeil():
    
    df['ceil_bin'] = df['ceil'].apply(lambda val: getCeilType(val))
    addMeanEncodedFeature(df['ceil_bin'].name)
    
def getSightType(x):
    if (x == 0):
        return "No_Visits"
    elif (x >= 1 and x <= 3):
        return "Few_Visits"
    elif (x > 3):
        return "More_Visits"
    else :
        return
    
def binSight():
    
    df['sight_bin'] = df['sight'].apply(lambda val: getSightType(val))
    addMeanEncodedFeature(df['sight_bin'].name)
    
    
def getYrBuilt(val):
    if str(val).find("1875, 1900") > 0:
        return "1900s"
    elif str(val).find("1900, 1925") > 0:
        return "1925s"
    elif str(val).find("1925, 1950") > 0:
        return "1950s"
    elif str(val).find("1950, 1975") > 0:
        return "1975s"
    elif str(val).find("1975, 2000") > 0:
        return "2000s"
    elif str(val).find("2000, 2025") > 0:
        return "2025s"
    else :
        return "Others"
        
def binYrBuilt():
    
    
    df['yr_built_tmpbin'] = pd.cut(df.yr_built, bins=[1875,1900,1925,1950,1975,2000,2025])
    df['yr_built_bin'] = df['yr_built_tmpbin'].apply(lambda val: getYrBuilt(val) )
    df.drop(['yr_built_tmpbin'], axis=1, inplace=True)
    
    
    addMeanEncodedFeature(df['yr_built_bin'].name)
    
def getYrRenovated(val, year):
    
    if (year == 0 or year == 1890):
        return "Not Renovated"
    elif str(val).find("1875, 1900") > 0:
        return "1900s"
    elif str(val).find("1900, 1925") > 0:
        return "1925s"
    elif str(val).find("1925, 1950") > 0:
        return "1950s"
    elif str(val).find("1950, 1975") > 0:
        return "1975s"
    elif str(val).find("1975, 2000") > 0:
        return "2000s"
    elif str(val).find("2000, 2025") > 0:
        return "2025s"
    else:
        return "Others"

def binYrRenovated() :   
   
    df.loc[(masterDataDF.yr_renovated == 0), "yr_renovated"]=1890    
    df['yr_renovated_tmpbin'] = pd.cut(df.yr_renovated, bins=[1875,1900,1925,1950,1975,2000,2025])
    df['yr_renovated_bin'] = df.apply(lambda val: getYrRenovated(val['yr_renovated_tmpbin'], val['yr_renovated']), axis=1 )


    addMeanEncodedFeature(df['yr_renovated_bin'].name)

def getZipcode(val):
    

    if str(val).find("98000, 98025") > 0:
        return "ZIPGRP1"
    elif str(val).find("98025, 98050") > 0:
        return "ZIPGRP2"
    elif str(val).find("98050, 98075") > 0:
        return "ZIPGRP3"
    elif str(val).find("98075, 98100") > 0:
        return "ZIPGRP4"
    elif str(val).find("98100, 98125") > 0:
        return "ZIPGRP5"
    elif str(val).find("98125, 98150") > 0:
        return "ZIPGRP6"
    elif str(val).find("98150, 98175") > 0:
        return "ZIPGRP7"
    elif str(val).find("98175, 98199") > 0:
        return "ZIPGRP8"
    else:
        return "Others"

def binZipcode():
    global df
   
    
    df['zipcode_tmpbin'] = pd.cut(df.zipcode, bins=[98000, 98025, 98050,98075, 98100,98125, 98150, 98175, 98199])
    df['zipcode_bin'] = df.apply(lambda val: getZipcode(val['zipcode_tmpbin']), axis=1 )
    df.drop(['zipcode_tmpbin'], axis=1, inplace=True)
    encoded_columns = pd.get_dummies(df['zipcode_bin'], prefix="zipcode")
    df = df.join(encoded_columns)
    
def binFurnished():
    addMeanEncodedFeature (df.furnished.name)
    
def binCoast():
    addMeanEncodedFeature (df.coast.name)
    
def dataLogTransformation():
    df['lot_measure_log'] = (df['lot_measure']+1).transform(np.log)
    df['ceil_measure_log'] = (df['ceil_measure']+1).transform(np.log)
    df['basement_log'] = (df['basement']+1).transform(np.log)
    
#Function to drop attributes
def dropAttributes (columns_list):
    for col in columns_list:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
            print ("Dropped Attribute : "+ col)
            
def dropFeatures():
    dropCols = ['living_measure15','lot_measure15']
    dropAttributes (dropCols)
    dropCols = ['cid','living_measure','total_area', 'dayhours']
    dropAttributes (dropCols)
    dropCols = ['room_bed','room_bath', 'lot_measure', 'ceil', 'coast', 'sight', 'condition',\
            'quality', 'ceil_measure', 'basement', 'yr_built', 'yr_built_bin', 'zipcode_bin',\
            'yr_renovated', 'yr_renovated_bin', 'zipcode', 'lat', 'long', 'furnished', 'yr_sold', \
            'long_bin', 'lat_bin', 'region', 'Region_name', 'condition_bin', 'quality_bin', \
            'room_bed_bin', 'room_bath_bin', 'ceil_bin', 'sight_bin', 'age_bin','age_sold', \
            'age_sold_bin','age_sold_quantile_bin'
           ]
    dropAttributes (dropCols)
    dropCols=['zipcode_ZIPGRP1','zipcode_ZIPGRP7','zipcode_ZIPGRP6', 'zipcode_ZIPGRP4']
    dropAttributes (dropCols)
    
#Function to replace outliers lying outside IQR range with median value.
def fixOutlier (col):
    global masterDataDF
    Q1 = col.quantile(0.25)
    Q3 =col.quantile(0.75)
    IQR = Q3 - Q1
    max_value = Q3+(1.5*IQR)
    min_value = Q1-(1.5*IQR) 
    masterDataDF.loc[( col < min_value) | (col > max_value), col.name] = col.median()

def fixOutliers ():
    global masterDataDF
    fixOutlier(masterDataDF.basement)
    fixOutlier(masterDataDF.lot_measure)
    fixOutlier(masterDataDF.ceil_measure)
    fixOutlier(masterDataDF.room_bath)


#Method to set the data types of all features
def setDataTypes():
    global df
    convert_dict = {'cid': object, 
                'dayhours': object,
                'room_bed': float,
                'room_bath': float,
                'living_measure': float,
                'lot_measure': float,
                'ceil': float,
                'coast': int,
                'sight': int,
                'condition': int,
                'quality': int,
                'ceil_measure': float,
                'basement': float,
                'yr_built': int,
                'yr_renovated': int,
                'zipcode': int,
                'lat': float,
                'long': float,
                'living_measure15': float,
                'lot_measure15': float,
                'furnished': int,
                'total_area': float
               }  
    df = df.astype(convert_dict) 

#Method for pre processing and feature engg of input data
def preProcessing(inputData): 
    
    global df 
    
    cols = ['cid', 'dayhours', 'room_bed', 'room_bath', 'living_measure',\
       'lot_measure', 'ceil', 'coast', 'sight', 'condition', 'quality',\
       'ceil_measure', 'basement', 'yr_built', 'yr_renovated', 'zipcode',\
       'lat', 'long', 'living_measure15', 'lot_measure15', 'furnished',\
       'total_area']
    
    df = pd.DataFrame([inputData], columns=cols)  

    
    setDataTypes()
    print ("DataTypes Set for all Features")
    processDayhours()
    print ("Derived age and year sold feature.")
    fixOutliers()    
    print ("Fixed outliers")
    binAgeSold ()
    print ("Age Sold - Processed")
    binLatLong ()
    print ("Lat & Long - Processed")
    binCondition ()
    print ("Condition- Processed")
    binQuality() 
    print ("Quality- Processed")
    binBedRooms ()
    print ("Bed Rooms- Processed")
    binBath ()
    print ("Bath Rooms- Processed")
    binCeil()
    print ("Ceil- Processed")
    binSight ()
    print ("Sight- Processed")
    binYrBuilt()
    print ("Yr Built- Processed")
    binYrRenovated()
    print ("Yr Renovated- Processed")
    binZipcode()
    print ("Zipcode- Processed")
    binFurnished()
    print ("Furnished- Processed")
    binCoast()
    print ("Coast- Processed")
    dataLogTransformation()
    print ("Data Log Transformation completed")
    setCategoricalColumns()
    print ("Set Categorical Features ")
    dropFeatures()
    
    model_cols=['furnished_enc', 'Region_name_enc', 'quality_bin_enc',
       'ceil_measure_log', 'lot_measure_log', 'sight_bin_enc', 'basement_log',
       'coast_enc', 'yr_built_bin_enc', 'yr_renovated_bin_enc',
       'zipcode_ZIPGRP3', 'age_sold_quantile_bin_enc', 'room_bed_bin_enc',
       'room_bath_bin_enc', 'zipcode_ZIPGRP5', 'ceil_bin_enc',
       'zipcode_ZIPGRP2', 'condition_bin_enc']
    
    
    for col in model_cols:
        if col in df.columns:
            final_df[col]=df[col]
        else:
            final_df[col]=0
    
    processedData = list(final_df.loc[0])
    return processedData
 
    
#Method to test the code as standalone
def test():
    #A simple method to read a input data and pass it to the 
    #processing method
    tempDf = pd.DataFrame()
    tempDf = pd.read_csv('../data/input.csv')
    inputData = list(tempDf.loc[0])
    model = pickle.load(open('../model/HousePrediction.pkl', 'rb')) 
    print (inputData)
    processedData = preProcessing(inputData)
    output = np.round(model.predict([processedData]),2)
    print ("Predicted Value ===> "+ str(output))

    
test()


