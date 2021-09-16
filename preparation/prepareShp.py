"""
Script for merging shapefiles, making new fields, filling them and removing unnecessary fields

Note: old shapefile will have both old and new fields, new shapefile will only have new fields

written by Samantha Wittke
"""


import subprocess
import shapefile
from osgeo import ogr
import os

#works in gisthings environment
#invalid geometries and geometries outside of Finland still need to be removed


#outputfilename = '/u/58/wittkes3/unix/Documents/AICropPro/croploss/merged_py7.shp'
#finalfilename = '/u/58/wittkes3/unix/Documents/AICropPro/croploss/merged_py8.shp'
#inputfilenames = ['/u/58/wittkes3/unix/Documents/AICropPro/croploss/2013/rap_2013.shp','/u/58/wittkes3/unix/Documents/AICropPro/croploss/2014/rap_2014.shp','/u/58/wittkes3/unix/Documents/AICropPro/croploss/2015/rap_2015.shp']

#directory with shapefiles with original fields
ofilenames = os.listdir('/u/58/wittkes3/unix/Documents/AICropPro/15y_barley/barley_new')
outputfilenames = [os.path.join('/u/58/wittkes3/unix/Documents/AICropPro/15y_barley/barley_new',x) for x in ofilenames if x.endswith('.shp')]

# to merge yearly shapefiles into one
#command = 'ogrmerge.py -o ' + outputfilename +' ' +  ' '.join(inputfilenames) + ' -single' 
#subprocess.call(command, shell=True)

def createField(layer,name,ogrtype,width):

    field = ogr.FieldDefn(name, ogrtype)
    field.SetWidth(width)
    layer.CreateField(field)


for outputfilename in outputfilenames:
    print(outputfilename)

    finalfilename = outputfilename.replace('barley_new','barley_newfields')

    #creating new fields in the old shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(outputfilename, 1)
    thelayer = dataSource.GetLayer()
    
    #giving it new field names
    createField(thelayer,"new_ID",ogr.OFTInteger,10)
    createField(thelayer,"orig_ID",ogr.OFTString,15)
    createField(thelayer,"year",ogr.OFTInteger,5)
    createField(thelayer,"full_cl",ogr.OFTReal,10)
    createField(thelayer,"partial_cl",ogr.OFTReal,10)
    createField(thelayer,"area",ogr.OFTReal,10)
    createField(thelayer,"plantcode",ogr.OFTInteger,10)
    createField(thelayer,"loss",ogr.OFTInteger,10)
    createField(thelayer,"speciescod",ogr.OFTString,10)
    createField(thelayer,"farmid",ogr.OFTInteger,10)

    #getting fields from old fieldnames
    for i,feature in enumerate(thelayer):
    
        #geom = feature.GetGeometryRef()
        #area = geom.GetArea() 
        #feature.SetField("Area", area)
        
        year = feature.GetField('vuosi')

        if year == 2015:
            myid = feature.GetField('lohkonro')
            area = feature.GetField('pintaala')
        else:
            myid = feature.GetField('plk_perusl')
            area = feature.GetField('pinta_ala')
        fullcl = feature.GetField('tays_tuho')
        partcl = feature.GetField('ositt_tuho')
        #test via print that no value is None and nothing else so that following line works
        loss = 1 if fullcl or partcl else 0  
        plantcode = feature.GetField('kasvikoodi')
        farmid = feature.GetField('tunnus')
        speciescode = feature.GetField('lajikekood')
        newid = feature.GetField('ID')
        
        #filling new fields
        feature.SetField("new_ID",newid)
        feature.SetField("year",year)
        feature.SetField("orig_ID",myid)
        feature.SetField("full_cl",fullcl)
        feature.SetField("partial_cl",partcl)
        feature.SetField("loss",loss)
        feature.SetField("area",area)
        feature.SetField("plantcode",plantcode)
        feature.SetField("speciescod",speciescode)
        feature.SetField("farmid",farmid)

        thelayer.SetFeature(feature)

    newfields= ["new_ID","year","orig_ID","full_cl","partial_cl","loss","area","plantcode","speciescod","farmid"]
    fieldnames = [afield.name for afield in thelayer.schema]
    print(fieldnames)

    dataSource = None

    #with select option all other fields are being deleted
    
    #creating new shapefile with only new fields
    command2 = 'ogr2ogr -select ' +','.join(newfields) + ' ' + finalfilename + ' ' + outputfilename
    subprocess.call(command2,shell=True)
