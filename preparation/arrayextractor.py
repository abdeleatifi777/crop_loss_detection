import os
from rasterstats import zonal_stats
import csv
import h5py
import subprocess
from osgeo import gdal,osr
import sys


#direct array extraction

def main(shapedir,bandpathtxt):
    bandpathlist = makelist(bandpathtxt)
    for bandpath in bandpathlist:
        print(bandpath)
        bandpath = bandpath[:-1]
        print(bandpath)
        print(os.path.split(bandpath))
        #LC08_L1GT_189017_20150601_20170408_01_T2_sr_band2.tif
        path = os.path.split(bandpath)[-1].split('_')[2][0:3]
        row = os.path.split(bandpath)[-1].split('_')[2][4:6]
        pathrow = str(path) + str(0) + str(row)
        year = os.path.split(bandpath)[-1].split('_')[3][0:4]
        shapefile = os.path.join(shapedir,'barley_newfields_' + pathrow + '_' + year+ '.shp')
        shapefile = reproject(shapefile,bandpath)
        #print(shapefile)
        extractarray(bandpath,shapefile,pathrow,year)


def extractarray(rasterpath ,shpfile,pathrow,year):

    date = os.path.split(rasterpath)[-1].split('_')[3]
    band = os.path.splitext(os.path.split(rasterpath)[-1])[0].split('_')[-1]
    

    print('arrayextractor started')

    a=zonal_stats(shpfile, rasterpath, stats=['mean'], band=1, geojson_out=True, all_touched=True, raster_out=True)

    for x in a:
        myarray = x['properties']['mini_raster_array']
        print(x['properties']['mini_raster_nodata'])
        print(myarray)
        myarray = myarray.filled(-9999)
        print(myarray)
        print(type(myarray))
        myid = x['properties']['new_ID']
        mystat = x['properties']['mean'] 
        tohdf(myid,date,band,myarray,pathrow,year)

def tohdf(myid,date,band,myarray,pathrow,year):

    myid = str(myid)
    date = str(date)

    
    hdfpath = '/scratch/cs/ai_croppro/data/hdfs'
    hdfname = os.path.join(hdfpath,pathrow + '_' + year + '.hdf')

    print(myarray)

    with h5py.File(hdfname, 'a') as hf:
        if hf.get(myid) is None:
            idgroup = hf.create_group(myid)
        else:
            idgroup = hf.get(myid)
        if idgroup.get(date) is None:
            dategroup = idgroup.create_group(date)
        else:
            dategroup = idgroup.get(date)
        dategroup.create_dataset(band,data= myarray)


def makelist(mytxt):
    #readtxt and put in list
    with open(mytxt) as f: mylist = f. readlines()
    return mylist


def getbandlist(mydir):

    #make list of paths to all bands
    bandpathlist = []
    for lsdir in os.listdir(mydir):
        for bandfile in os.listdir(os.path.join(mydir,lsdir)):
            if bandfile.endswith('pixel_qa.tif') or 'band' in bandfile and bandfile.endswith('.tif'):
                bandpath = os.path.join(mydir,lsdir,bandfile)
                bandpathlist.append(bandpath)

    return bandpathlist

def reproject(shpfile,raster):
    print('INFO: checking the projection dependent on '+ raster +' of the inputfile now')
    head, tail = os.path.split(shpfile)
    root, ext = os.path.splitext(tail)
    openrasterfile = gdal.Open(raster)
    rasterprojection = openrasterfile.GetProjection()
    rasterrs = osr.SpatialReference(wkt=rasterprojection)
    rasterepsg = rasterrs.GetAttrValue('AUTHORITY',1)
    #rasterepsg = '32634'
    ##reproject the shapefile according to projection of Sentinel2/raster image
    reprojectedshape = os.path.join(head, root + '_reprojected_'+ rasterepsg+ ext)
    if not os.path.exists(reprojectedshape):
        reprojectcommand = 'ogr2ogr -t_srs EPSG:'+rasterepsg+' ' + reprojectedshape + ' ' + shpfile
        subprocess.call(reprojectcommand, shell=True)
        print('INFO: ' + shpfile + ' was reprojected to EPSG code: ' + rasterepsg + ' based on the projection of ' + raster)
    return reprojectedshape

bandpathtxt = sys.argv[1]
shapedir = '/scratch/cs/ai_croppro/data/shp'
main(shapedir,bandpathtxt)