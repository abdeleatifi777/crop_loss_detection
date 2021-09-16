"""
extracting mean value for each polygon
"""

import os
from rasterstats import zonal_stats
import csv
import numpy as np
import sys



rasterpath = sys.argv[1]

shpfile = sys.argv[2]

csvdir = '/scratch/cs/ai_croppro/data/meteo/meteo_csv'

date = rasterpath.split('.')[0].split('_')[-1]    
year = date[0:4]            
mtype = rasterpath.split('/')[-3]
thename = mtype + '_' + str(date) +'.csv'                 

if not os.path.exists(os.path.join(csvdir,year)):
    os.mkdir(os.path.join(csvdir,year))

csvname = os.path.join(csvdir,year, thename)
                    
if not os.path.exists(csvname):

    a=zonal_stats(shpfile, rasterpath, stats=['mean'], band=1, geojson_out=True, all_touched=True)

    statlinex = ['ID','value']
   
    with open (csvname,'w') as sh:
        shWriter=csv.writer(sh)
        shWriter.writerow(statlinex)
        
    with open (csvname,'a') as sf:
        sfWriter=csv.writer(sf)
        for x in a:
            row = [x['properties']['ID'], x['properties']['mean']]
            sfWriter.writerow(row)
   