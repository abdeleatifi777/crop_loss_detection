import os
from rasterstats import zonal_stats
from osgeo import osr, gdal
import subprocess
import csv
import numpy as np
import h5py
import datetime
import sys

yeardir = sys.argv[1]
shpdir = '/scratch/cs/ai_croppro/data/shp'
hdfdir = '/scratch/cs/ai_croppro/meteohdf'
fillarray = np.full((1,366),-9999)
    
hf = h5py.File(os.path.join(hdfdir,os.path.split(yeardir)[1] +'.hdf'),'a')    

shpfile = os.path.join(shpdir, 'barley_newfields_5tiles_' + str(os.path.split(yeardir)[1]) + '.shp')

for raster in os.listdir(yeardir):

    year = str(yeardir)
    print(raster)
    mtype = raster.split('_')[0]
    print(mtype)
    dateobj = datetime.datetime.strptime(raster.split('_')[1][0:8], '%Y%m%d')
    print(dateobj)
    doy = (dateobj - datetime.datetime(dateobj.year, 1, 1)).days + 1
    print(doy)
    rasterpath = os.path.join(yeardir,raster)	
        
    a=zonal_stats(shpfile, rasterpath, stats=['mean'], band=1, geojson_out=True, all_touched=True)         

    for x in a:
        myid = x['properties']['new_ID']
        if not str(myid) in hf:
            hf.create_group(str(myid))
        idgroup = hf.get(str(myid))
        mtypegroup = str(myid) + '/' + mtype
        if not mtypegroup in hf:
            idgroup.create_dataset(mtype, data= fillarray)
        data = hf[mtypegroup][()]
        #print(data)
        data[0,doy-1] = x['properties']['mean']
        #print(data)
        del hf[mtypegroup]
        idgroup.create_dataset(mtype,data=data)
                
