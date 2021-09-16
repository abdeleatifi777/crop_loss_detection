"""
script for extracting txt files from all available landsat tiles/timepoints
"""
import os


def getbandlist(mydir):

    #make list of paths to all bands
    bandpathlist = []
    for lsdir in os.listdir(mydir):
        for bandfile in os.listdir(os.path.join(mydir,lsdir)):
            if bandfile.endswith('pixel_qa.tif') or 'band' in bandfile and bandfile.endswith('.tif'):
                bandpath = os.path.join(mydir,lsdir,bandfile)
                bandpathlist.append(bandpath)

    return bandpathlist

def writetofile(mytxtname,bandpathlist):
    with open(mytxtname,'w') as txt:
        for bandpath in bandpathlist:
            txt.write(bandpath +'\n')



ls7dir = '/scratch/cs/ai_croppro/data/ls7'

for prdirx in os.listdir(ls7dir):
    prdir = os.path.join(ls7dir,prdirx)
    for yeardirx in os.listdir(prdir):
        yeardir = os.path.join(prdir,yeardirx)
        for tilex in os.listdir(yeardir):
            tile = os.path.join(yeardir,tilex)
            mytxtfile = '/scratch/cs/ai_croppro/data/ls7lists/'+ prdirx 
            bandpathlist = getbandlist(tile)
            writetofile(mytxtfile,bandpathlist)

