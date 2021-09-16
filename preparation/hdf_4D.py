# stacking bands and timepoints per ID from hdf with id/date/bands/data to id/data
# also adding date information as one-hot encoded information to attribute file

import h5py
import pandas as pd
import numpy as np
from datetime import datetime

#hdf = '/u/58/wittkes3/unix/Documents/AICropPro/ordering_hdf/187018_2010.hdf'
#newhdf = '/u/58/wittkes3/unix/Documents/AICropPro/ordering_hdf/187018_2010_stacked.hdf'

#corresponding attribute file

#attr = '/u/58/wittkes3/unix/Documents/AICropPro/attributes_15y/barley_newfields_187018_2010_attributes.csv'

hdfdir = '/scratch/cs/ai_croppro/data/newerhdfs'
newhdfdir = '/scratch/cs/ai_croppro/data/stacked_hdf'

for hdf in os.listdir(hdfdir):
    name = os.path.splitext(hdf)
    hdf = os.path.join(hdfdir,hdf)
    newhdf = os.path.join(newhdfdir,name+ 'stacked.hdf')
    tile = name.split('_')[0]
    year = name.split('_')[1]
    attr = '/scratch/cs/ai_croppro/data/attributes_15y/barley_newfields_'+ tile +'_' + year + '_attributes.csv'
    newattr = '/scratch/cs/ai_croppro/data/attributes_15y/barley_newfields_'+ tile +'_' + year + '_attributes_doy.csv'
# read hdf

    doylist = list(range(1,366))

    with h5py.File(hdf, 'r') as f:
        with h5py.File(newhdf,'w') as nf:
            a = pd.read_csv(attr)
            a = a.reindex(columns = a.columns.tolist() + doylist)

            # per ID key
            for x in f.keys():
                #print(x)
            #get dates for each id
                dates = list(f[x])
                #print(list(f[x]))

            #get the doy
                zeroarray = np.zeros((365,),dtype = int)
                datearraylist = []
                for date in dates: 

                    dateobj = datetime.strptime(date, '%Y%m%d')
                    doy = (dateobj - datetime(dateobj.year, 1, 1)).days + 1
                    doyindex = doy - 1
                    zeroarray[doyindex] = 1
            # save it one hot encoded as addition to attribute file
            # find corresponding ID


        # per doy, stack the bands
                    #print(list(f[x][date]))
                    bands = list(f[x][date])
                    bandarraylist = []
                    for b in bands:
                        #print(list(f[x][date][b]))
                        bandarray = f[(x + '/' + date + '/' + b)]
                        #print(bandarray)
                        bandarraylist.append(bandarray)
                    bandstack = np.stack(bandarraylist)
                    datearraylist.append(bandstack)

                stackedstacks = np.stack(datearraylist)
                #print(stackedstacks.shape)

                d = dict(zip(doylist,zeroarray))
                a.loc[x] = pd.Series(d)

                nf.create_dataset(x, data =stackedstacks)

                #safe csv
            a.to_csv(newattr)